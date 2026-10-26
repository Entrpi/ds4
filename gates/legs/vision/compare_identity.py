#!/usr/bin/env python3
"""Compare the identity-leg outputs: oracle{1,2} vs fork{1,2} jsonl in DIR."""
import json, sys, os
rd = sys.argv[1]
EOS = 1
def trunc(g):
    return g[:g.index(EOS)] if EOS in g else g
def load(n):
    out = {}
    for line in open(os.path.join(rd, n + ".jsonl")):
        line = line.strip()
        if line.startswith("{"):
            d = json.loads(line)
            base = "text" if d["image"] == "-" else "+".join(os.path.basename(p) for p in d["image"].split("+"))
            key = f"{base}#{d['prompt_len']}"          # one key per (images, prompt)
            d["generated_full"] = d["generated"]; d["generated"] = trunc(d["generated"])
            out[key] = d
    return out
def bh(d): return d.get("block_fnv1a64", d.get("block_sha256"))
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
o1, o2, f1, f2 = load("oracle1"), load("oracle2"), load("fork1"), load("fork2")
imgs = sorted(set(o1) | set(o2) | set(f1) | set(f2))
print(f"{'case (images#prompt_len)':44s} {'o-twin':>7s} {'f-twin':>7s} {'ids':>5s} {'span':>5s} {'block':>5s} {'lcp/len-to-EOS':>14s}")
allok = True
for im in imgs:
    a, b, c, d = o1.get(im), o2.get(im), f1.get(im), f2.get(im)
    if not (a and b and c and d):
        print(f"{im:44s} MISSING oracle={bool(a)},{bool(b)} fork={bool(c)},{bool(d)}"); allok = False; continue
    otwin = "ok" if a["generated"] == b["generated"] else f"lcp{lcp(a['generated'], b['generated'])}"
    ftwin = "ok" if c["generated"] == d["generated"] else f"lcp{lcp(c['generated'], d['generated'])}"
    ids = "ok" if a["prompt_ids"] == c["prompt_ids"] else "DIFF"
    sp = "ok" if a["span"] == c["span"] else "DIFF"
    bl = "ok" if bh(a) == bh(c) else "DIFF"
    gl = lcp(a["generated"], c["generated"]); same = a["generated"] == c["generated"]
    g = f"{gl}/{len(a['generated'])}" + ("" if same else "*")
    if not (ids == "ok" and sp == "ok" and bl == "ok" and same): allok = False
    print(f"{im:44s} {otwin:>7s} {ftwin:>7s} {ids:>5s} {sp:>5s} {bl:>5s} {g:>14s}")
print()
for im in imgs:
    a, c = o1.get(im), f1.get(im)
    if a and c and a["generated"] != c["generated"]:
        i = lcp(a["generated"], c["generated"])
        ot = a.get("top3", [[None]*3]*len(a["generated"]))[i] if i < len(a.get("top3", [])) else None
        ft = c.get("top3", [[None]*3]*len(c["generated"]))[i] if i < len(c.get("top3", [])) else None
        tie = ""
        if ot and ft:
            tie = " NEAR-TIE (each side's top-1 is in the other's top-3)" if (a["generated"][i] in ft and c["generated"][i] in ot) else " NOT a top-3 swap"
        gap = ""
        t8 = c.get("top8")
        if t8 and i < len(t8):
            row = t8[i]; ids = [x[0] for x in row]; ot1 = a["generated"][i]
            if ot1 in ids:
                r = ids.index(ot1); gap = f"; fork logits: top1 {row[0][0]}={row[0][1]:.3f}, oracle's pick {ot1} rank {r+1} at {row[r][1]:.3f} (gap {row[0][1]-row[r][1]:.3f})"
            else:
                gap = f"; fork logits: top1 {row[0][0]}={row[0][1]:.3f}, oracle's pick {ot1} NOT in fork top-8 (8th {row[7][0]}={row[7][1]:.3f})"
        print(f"{im}: diverge @{i}: oracle {a['generated'][i:i+6]} vs fork {c['generated'][i:i+6]}; top3 oracle={ot} fork={ft}{tie}{gap}")
print("VERDICT:", "IDENTICAL on all images" if allok else "NOT identical (see rows marked * or DIFF)")
