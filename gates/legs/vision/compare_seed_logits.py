#!/usr/bin/env python3
"""inc 6: batched vs serial PREFILL state at the seed position.
usage: compare_seed_logits.py SERIAL.bin BATCHED.bin N_VOCAB
Records: u32 a, u32 prompt_len, float[N_VOCAB].  Matched by prompt_len (the
probe's prompts have distinct lengths; the batched file may hold extra
admissions -- warm/restored -- which are skipped: only the first record per
length, i.e. the cold admission, is compared)."""
import sys, struct, array
# numpy-free on purpose (the GB10 boxes carry no numpy): array + sorted indices; 8 records x 129280 floats is instant.
def load(path, nv):
    out = {}
    with open(path, 'rb') as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            a, L = struct.unpack('<II', h)
            v = array.array('f'); raw = f.read(4 * nv)
            if len(raw) < 4 * nv: break
            v.frombytes(raw)
            out.setdefault(L, (a, v))
    return out
nv = int(sys.argv[3]); BAND = float(sys.argv[4]) if len(sys.argv) > 4 else 0.34
S, B = load(sys.argv[1], nv), load(sys.argv[2], nv)
worst = 0.0; pairs = 0; argmax_eq = 0; near_ties = 0; fails = []
for L in sorted(S):
    if L not in B: print(f"len={L}: no batched record"); continue
    (i, s), (b, v) = S[L], B[L]
    d = [abs(x - y) for x, y in zip(s, v)]
    top_s = sorted(range(nv), key=lambda k: -s[k])[:20]; top_b = sorted(range(nv), key=lambda k: -v[k])[:20]
    ss = sorted(s)[-2:]; sv = sorted(v)[-2:]
    gap_s = float(ss[1] - ss[0]); gap_b = float(sv[1] - sv[0])
    ov = len(set(top_s) & set(top_b))
    worst = max(worst, float(max(d)))
    pairs += 1
    if top_s[0] == top_b[0]: argmax_eq += 1
    elif gap_s <= BAND: near_ties += 1
    else: fails.append((i, L, gap_s))
    print(f"prompt={i} len={L} bank={b} argmax_equal={int(top_s[0]==top_b[0])} top20_overlap={ov}/20 "
          f"max_abs_delta={max(d):.4f} mean_abs_delta={sum(d)/len(d):.5f} top1top2_gap serial={gap_s:.3f} batched={gap_b:.3f}")
print(f"worst_max_abs_delta={worst:.4f}")
print(f"SEED_TWIN_{'PASS' if pairs and not fails else 'FAIL'} pairs={pairs} argmax_equal={argmax_eq} near_ties={near_ties} (serial gap <= {BAND}) fails={fails}")
