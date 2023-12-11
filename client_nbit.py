import hashlib
import cv2
import numpy as np
import mss
import pyautogui as pag
import json
import math
from pykeyboard import PyKeyboard
import time
from numba import jit
import multiprocessing as mp
import os

from server_nbit import WarpServer, get_file_md5


@jit()
def find_anchor(image, xrange, yrange, color, find_last=False):
    if find_last:
        axis_y = range(yrange[1], yrange[0], -1)
        axis_x = range(xrange[1], xrange[0], -1)
    else:
        axis_y = range(yrange[0], yrange[1], 1)
        axis_x = range(xrange[0], xrange[1], 1)
    for y in axis_y:
        for x in axis_x:
            diff = np.abs(image[y, x] - np.array(color)).sum()
            if diff < 30:
                return x, y
    return None


def process_data(image, pt1, pt2, pixels_per_line):
    (x1, y1), (x2, y2) = pt1, pt2
    H, W = y2 - y1 - 1, x2 - x1 - 1
    dh = H / pixels_per_line
    dw = W / pixels_per_line

    data = image[pt1[1] + 1 : pt2[1], pt1[0] + 1 : pt2[0]]  # noqa:E203
    mesh_x = [int(i * dw + dw * 0.5) for i in range(pixels_per_line)]
    mesh_y = [int(i * dh + dh * 0.5) for i in range(pixels_per_line)]
    data = data[mesh_x][:, mesh_y]
    return data


def pixel_to_bytes(data, num_bits):
    psize = 8 // num_bits
    data_range = 2**num_bits
    n = int(math.ceil(math.sqrt(data_range)))
    scaler = (n - 1) / 255
    k = np.round(data[..., 0] * scaler).astype(int)
    b = np.round(data[..., 2] * scaler).astype(int)
    p = (k * n + b).reshape(-1, psize)
    vfunc = np.vectorize(lambda i: "{0:0{k}b}".format(i, k=num_bits))
    p = vfunc(p)
    ret = []
    for data in p:
        data = int("".join((str(x) for x in data)), 2)
        ret.append(data)
    return ret


def screenshot_stream(dsize=416, exit_key="q", color_format=cv2.COLOR_RGBA2RGB):
    if isinstance(dsize, int):
        dsize = (dsize, dsize)
    h, w = dsize
    capture_range = {"top": 0, "left": 0, "width": w, "height": h}
    cap = mss.mss()

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(exit_key):
            break
        x, y = pag.position()  # 返回鼠标的坐标
        capture_range["top"] = int(y - capture_range["height"] // 2)
        capture_range["left"] = int(x - capture_range["width"] // 2)
        frame = cap.grab(capture_range)
        frame = np.array(frame)
        # frame = cv2.resize(frame, (w, h))
        if color_format:
            frame = cv2.cvtColor(frame, color_format)
        yield frame


class WarpClient:
    def __init__(self):
        self.num_bits = WarpServer.num_bits
        self.page_size = WarpServer.page_size
        self.field_sizes = WarpServer.field_sizes
        self.head_size = WarpServer.head_size
        self.qr_size = WarpServer.qr_size
        self.pixels_per_line = WarpServer.pixels_per_line
        self.max_file_bytes = WarpServer.max_file_bytes
        self.k = PyKeyboard()
        self.loaded_pages = -1
        self.time_ckpt = time.time()
        self.mouse_position = None
        self.anchor_cache = None
        self.num_process = os.cpu_count()
        self.pool = mp.Pool(self.num_process)

    def align_mesh(self, image):
        x, y = pag.position()
        if self.anchor_cache is not None and self.mouse_position == (x, y):
            pt1, pt2 = self.anchor_cache
        else:
            pt1 = find_anchor(image, (0, 30), (0, 30), color=(0, 255, 0), find_last=True)
            pt2 = find_anchor(image, (-31, -1), (-31, -1), color=(0, 255, 0), find_last=False)
            self.anchor_cache = (pt1, pt2)
            self.mouse_position = (x, y)
        data = None
        if pt1 is not None and pt2 is not None:
            IW, IH = image.shape[0], image.shape[1]
            pt2 = (IW + pt2[0], IH + pt2[1])
            data = process_data(image, pt1, pt2, self.pixels_per_line)
            image = cv2.rectangle(image, pt1, pt2, color=(0, 255, 0))
        image = cv2.resize(image, (self.qr_size // 2, self.qr_size // 2), interpolation=cv2.INTER_AREA)
        cv2.imshow("qr-warp:client", image)
        cv2.waitKey(1)
        return data

    def pixel_to_bytes(self, data):
        chunk_size = len(data) // self.num_process
        tasks = []
        for i in range(self.num_process):
            chunk = data[i * chunk_size : (i + 1) * chunk_size]
            k = self.pool.apply_async(pixel_to_bytes, args=(chunk, self.num_bits))
            tasks.append(k)

        ret = []
        for i in range(self.num_process):
            task_ret = tasks[i].get()
            ret.extend(task_ret)
        return ret

    def read(self):
        for image in screenshot_stream(self.qr_size + 24):
            try:
                data = self.align_mesh(image)
                if data is not None:
                    data = self.pixel_to_bytes(data)

                    header = data[: self.head_size]
                    header = bytes(header).decode("utf-8")
                    splits = [0] + self.field_sizes.copy()
                    header_splits = []
                    for i in range(1, len(splits)):
                        splits[i] += splits[i - 1]
                        header_splits.append(header[splits[i - 1] : splits[i]])
                    page_id = int(header_splits[0])
                    page_dsize = int(header_splits[1])
                    page_hash = header_splits[2]

                    data = data[self.head_size : self.head_size + page_dsize]
                    data = bytes(data)

                    assert hashlib.md5(data).hexdigest() == page_hash
                    finished = self.load_page(header_splits, page_id, data)
                    if finished:
                        break
            except Exception:
                pass
        self.pool.close()
        print("md5:", get_file_md5(self.m["file_name"]))

    def load_page(self, header_splits, page_id, data):
        if page_id > self.loaded_pages + 1:
            raise ValueError(f"page {self.loaded_pages + 1} lost, current page: {page_id}")
        if page_id == self.loaded_pages + 1:
            if page_id == 0:
                m = json.loads(data)
                self.m = m
                m: dict
                print("INFO file_size=", m["file_size"])
                print("INFO file_md5=", m["file_md5"])
                self.n_pages = int(math.ceil(int(m["file_size"]) / int(m["page_size"])))
                print("================================== starting warping, num pages:", self.n_pages)
                print("ready to warp, press 'N' to continue")
                # initialize file
                with open(m["file_name"], "w"):
                    pass
            else:
                if page_id == self.loaded_pages + 1:
                    with open(self.m["file_name"], "ab") as f:
                        f.write(data)
            self.loaded_pages += 1
            now = time.time()
            delta = now - self.time_ckpt
            self.time_ckpt = now
            v = (self.page_size / 1000) / delta
            print(f"RECV {page_id}/{self.n_pages}", header_splits[2], f"{v:.4f}kb/s")
            self.k.tap_key("N")
            if page_id >= self.n_pages:
                return True
        return False


if __name__ == "__main__":
    c = WarpClient()
    c.read()
