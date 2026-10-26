#!/usr/bin/env python3
"""Compare two DS4_REPL_TOKEN_DUMP files (fork vs upstream oracle).

Each file holds one record per REPL turn: a `turn prompt_len=N images=K`
line, the N ids, then one `image i start= count= fingerprint=` line per
span.  The CLI constructs everything except the tokens the model generated;
with `-n 1` each earlier turn contributed exactly one generated token, at
the position where that turn's prompt ended.  Those positions are the only
places the two transcripts may legitimately differ (the kernels differ
numerically); everything else -- ids, image spans, fingerprints -- must be
byte-identical for the gate to pass."""
import sys

def parse(path):
    turns = []
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        hdr = lines[i]; assert hdr.startswith("turn "), hdr
        n = int(hdr.split("prompt_len=")[1].split()[0])
        k = int(hdr.split("images=")[1])
        ids = [int(x) for x in lines[i + 1].split()]
        assert len(ids) == n, (len(ids), n)
        spans = lines[i + 2:i + 2 + k]
        turns.append((n, ids, spans))
        i += 2 + k
    return turns

a, b = parse(sys.argv[1]), parse(sys.argv[2])
ok = True
if len(a) != len(b):
    print(f"FAIL turn count {len(a)} vs {len(b)}"); sys.exit(1)
gen_positions = []   # where earlier turns' generated tokens sit
gen_equal = 0
gen_checked = 0
for t, ((na, ia, sa), (nb, ib, sb)) in enumerate(zip(a, b), 1):
    if na != nb:
        print(f"FAIL turn {t}: prompt_len {na} vs {nb}"); ok = False; continue
    diffs = [p for p in range(na) if ia[p] != ib[p]]
    bad = [p for p in diffs if p not in gen_positions]
    if bad:
        print(f"FAIL turn {t}: {len(bad)} constructed positions differ, first at {bad[0]} ({ia[bad[0]]} vs {ib[bad[0]]})"); ok = False
    if sa != sb:
        print(f"FAIL turn {t}: image spans differ\n  {sa}\n  {sb}"); ok = False
    gen_equal += sum(1 for p in gen_positions if p < na and ia[p] == ib[p])
    gen_checked += sum(1 for p in gen_positions if p < na)
    print(f"turn {t}: prompt_len={na} images={len(sa)} constructed-identical={'yes' if not bad else 'NO'} "
          f"spans-identical={'yes' if sa == sb else 'NO'} generated-diffs-at={[p for p in diffs if p in gen_positions]}")
    gen_positions.append(na)   # this turn's generated token lands here
print(f"generated-token comparisons equal across trees: {gen_equal} of {gen_checked}")
print("REPL_TOKENS_PASS" if ok else "REPL_TOKENS_FAIL")
sys.exit(0 if ok else 1)
