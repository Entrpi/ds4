#!/usr/bin/env python3
"""Regenerate the synthetic fixture images (license-clean, no external
dependencies): gradients, bars and checkers at the sizes the Vision-Exp
preprocess cares about (tiny, odd, typical, portrait, 8:1 wide, large).
Content is irrelevant to the encoder bit-exact gate; sizes and formats are
what it exercises.  The JPEG twin of the typical image is made with
`sips -s format jpeg -s formatOptions 85` on macOS (or any encoder)."""
import math, os, struct, zlib

D = os.path.dirname(os.path.abspath(__file__))

def png(path, w, h, pix):
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw.extend(pix(x, y))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    open(path, "wb").write(data)
    return len(data)

def grad(w, h):
    def f(x, y):
        r = int(255 * x / max(w - 1, 1)); g = int(255 * y / max(h - 1, 1))
        cx, cy = w * 0.35, h * 0.5; d = math.hypot(x - cx, y - cy) / max(min(w, h) * 0.3, 1)
        b = int(255 * max(0.0, 1.0 - d)) if d < 1 else int(40 + 40 * math.sin(x / 17.0) * math.cos(y / 23.0))
        return (r, g, max(0, min(255, b)))
    return f

def checker(w, h, c=8):
    return lambda x, y: ((235, 235, 235) if ((x // c) + (y // c)) % 2 == 0 else (20, 20, 20))

def bars(w, h):
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
    return lambda x, y: (30, 30, 30) if abs(y - (x * h) // max(w, 1)) < 2 else cols[(x * 8) // max(w, 1)]

def bands(w, h):
    return lambda x, y: (int(255 * ((x * 12) // w) / 11), int(255 * y / (h - 1)), 200 if (y // 150) % 2 == 0 else 40)

SIZES = [("tiny_16x16.png", 16, 16, checker(16, 16, 4)), ("odd_17x9.png", 17, 9, grad(17, 9)),
         ("typical_532x280.png", 532, 280, grad(532, 280)), ("portrait_600x900.png", 600, 900, bars(600, 900)),
         ("wide_8to1_2000x250.png", 2000, 250, bars(2000, 250)), ("large_2400x1800.png", 2400, 1800, bands(2400, 1800)),
         ("checker_640x480.png", 640, 480, checker(640, 480, 16))]

if __name__ == "__main__":
    total = 0
    for name, w, h, f in SIZES:
        n = png(os.path.join(D, name), w, h, f); total += n
        print("%-26s %5dx%-5d %8d B" % (name, w, h, n))
    print("total", total, "bytes")
