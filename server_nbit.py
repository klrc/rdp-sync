from collections import namedtuple
import os
import hashlib
import cv2
import math
import numpy as np
import json
import sys

WarpMeta = namedtuple("WarpMeta", ["file_size", "file_name", "file_md5", "page_size"])


def get_file_name(file):
    return file.split("/")[-1]


def get_file_md5(file):
    with open(file, "rb") as f:
        data = f.read()
    file_md5 = hashlib.md5(data).hexdigest()
    return file_md5


class WarpServer:
    num_bits = 4
    page_size = 8000 * num_bits
    max_file_bytes = 100 * 10**6 // page_size
    field_sizes = [len(str(max_file_bytes)), len(str(page_size)), len("43ac2b807baa04d47e86099e98e88cc5")]
    head_size = sum(field_sizes)
    pixels_per_line = 256
    qr_size = pixels_per_line * 3

    def __init__(self):
        print("INIT qr-warp:server")
        print("INFO page_size=", self.page_size)
        print("INFO max_file_bytes=", self.max_file_bytes)
        print("INFO head_size=", self.head_size)
        print("INFO pixels_per_line=", self.pixels_per_line)
        print("INFO qr_size=", self.qr_size)
        assert 8 % self.num_bits == 0
        self.n_pages = 1

    def post(self, file):
        zip_file = f"{file}.tar.gz"
        os.system(f"tar -zcvf {zip_file} {file}")
        data_pages = self.load_file(zip_file)
        for i, page in enumerate(data_pages):
            self.display_data(i, page)
            self.receive_next()
        os.system(f"rm {zip_file}")
        print("md5:", self.m.file_md5)

    def get_header(self, i, chunk):
        stri = str(i).rjust(len(f"{self.max_file_bytes}"), "0")
        lc = str(len(chunk)).rjust(len(str(self.page_size)), "0")
        chunk_md5 = hashlib.md5(chunk).hexdigest()
        header = f"{stri}{lc}{chunk_md5}"
        print(f"SEND {i}/{self.n_pages} {chunk_md5}")
        return header

    def bytes_to_bin(self, bytes_data):
        return [str(bin(x))[2:].rjust(8, "0") for x in bytes_data]

    def bin_to_pixel(self, bin_data):
        data = []
        psize = 8 // self.num_bits
        data_range = 2**self.num_bits
        n = int(math.ceil(math.sqrt(data_range)))
        scaler = 255 / (n - 1)
        for x in bin_data:
            # depth=2 for BGR, avoid G channel for green screens
            for i in range(psize):
                xs = int(x[self.num_bits * i : self.num_bits * (i + 1)], 2)
                # x[num_bits] -> y
                k = xs // n
                b = xs % n
                data.append((int(k * scaler), 0, (b * scaler)))
        return np.asarray(data, dtype=np.uint8)

    def pixel_to_image(self, data):
        S = self.pixels_per_line
        assert S**2 >= len(data), (len(data), S, S**2)

        data = np.pad(data, ((0, S**2 - len(data)), (0, 0)), "constant", constant_values=0)
        data.resize((S, S, 3))

        data = cv2.resize(data, (self.qr_size, self.qr_size), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        data = cv2.copyMakeBorder(data, 16, 16, 16, 16, borderType=cv2.BORDER_CONSTANT, value=0)
        data[:16, :16, 1] = 255
        data[-16:, -16:, 1] = 255
        return data

    def load_file(self, file):
        file_size = os.path.getsize(file)
        file_name = get_file_name(file)
        file_md5 = get_file_md5(file)
        page_size = self.page_size

        # display meta information
        m = WarpMeta(file_size, file_name, file_md5, page_size)
        self.m = m
        yield bytes(json.dumps(m._asdict()), encoding="utf-8")

        # display data
        with open(file, "rb") as f:
            n_pages = math.ceil(m.file_size / m.page_size)
            self.n_pages = n_pages
            print("================================== starting warping, num pages:", n_pages)
            for _ in range(n_pages):
                chunk = f.read(m.page_size)
                yield chunk

    def display_data(self, i, data):
        header = self.get_header(i, data)
        head = bytes(header, encoding="utf-8")
        byte_data = head + data
        if i > 0:
            assert len(head) == self.head_size
        bin_data = self.bytes_to_bin(byte_data)
        pixel = self.bin_to_pixel(bin_data)
        image = self.pixel_to_image(pixel)
        cv2.imshow("QR", image)

    def receive_next(self):
        key = None
        while key is None or key != ord("n"):
            key = cv2.waitKey(0) & 0xFF


if __name__ == "__main__":
    s = WarpServer()
    s.post(sys.argv[1])
