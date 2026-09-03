#!/usr/bin/env python3
"""Compare two dump_vision_embedding outputs (fork vs upstream oracle).

usage: compare_vision_dumps.py A.bin B.bin
Prints the header fields, the fingerprint, and for the [tokens x 4096] F32
block: bit-exact (yes/no), max |a-b|, max rel, count of differing floats and
the first differing row.  Exit 0 iff byte-identical (the inc-2 gate)."""
import struct, sys

def load(p):
    d = open(p, "rb").read()
    h = struct.unpack("<8I", d[:32]); fp = d[32:64]; body = d[64:]
    n = h[0] * 4096
    assert len(body) == n * 4, (p, len(body), n * 4)
    return h, fp, struct.unpack("<%df" % n, body), body

if len(sys.argv) != 3:
    print(__doc__); sys.exit(2)
a, b = sys.argv[1], sys.argv[2]
ha, fa, va, ba = load(a); hb, fb, vb, bb = load(b)
names = ["tokens", "layout", "grid_w", "grid_h", "width", "height", "content_w", "content_h"]
print("A:", dict(zip(names, ha)), fa[:6].hex())
print("B:", dict(zip(names, hb)), fb[:6].hex())
if ha != hb or fa != fb:
    print("HEADER/FINGERPRINT MISMATCH"); sys.exit(1)
exact = ba == bb
diffs = 0; mx = 0.0; mrel = 0.0; first = None
if not exact:
    for i, (x, y) in enumerate(zip(va, vb)):
        if x != y:
            diffs += 1; d = abs(x - y); mx = max(mx, d)
            m = max(abs(x), abs(y)); mrel = max(mrel, d / m if m > 1e-6 else d)
            if first is None: first = (i // 4096, i % 4096, x, y)
print("bit-exact:", "YES" if exact else "NO", "| floats:", len(va), "| differing:", diffs,
      "| max_abs: %.6g | max_rel: %.6g" % (mx, mrel), "| first diff (row, col, a, b):", first)
sys.exit(0 if exact else 1)
