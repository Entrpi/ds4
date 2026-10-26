#!/usr/bin/env bash
# battery: perf_tripwire tier=2 stop=0 server=33 driver=15 trigger=always est=5 seq=30
# verdict: ^TRIPWIRE_(PASS|FLAG|FAIL)
# catches: a gross prefill speed regression (ds4-bench rows at 2048 and 65536 against the recorded band; informational: FLAG below band - 3%, FAIL below - 10%); ABBA (perf/batched_lane_abba.sh) is the statistical leg
# cmd: R=$RB_SERVER BINDIR=$RB_TREE MODEL=$RB_VBASE OUT="$RB_OUT" bash gates/legs/perf/tripwire.sh
# gates/legs/perf/tripwire.sh — the release battery's perf tripwire (release-process §3.0, 2026-09-04).
# One pure-prefill ds4-bench sweep with two rows (2048 and 65536 fresh tokens) on the server box, parsed
# from the bench's own CSV header (ctx_tokens, prefill_tps), compared with the record band at a loose
# tolerance.  Informational and always disclosed: it makes a speed regression VISIBLE; ABBA (§3e) stays the
# instrument for any <5% claim.  Verdict line: TRIPWIRE_PASS | TRIPWIRE_FLAG (a row below band - TOL%) |
# TRIPWIRE_FAIL (no rows).  Note: the 64k row of a two-row sweep prefills 63488 fresh tokens, which is not
# the record sweep's cumulative shape, so its band is the 2k..64k record range, not the 64k point.
# Env: R (ssh name of the server box, default ${DS4_BOX33:?}), BINDIR, MODEL (default: the tree's ds4flash.gguf),
#      OUT (dir for the csv + bench log), TOL (percent, default 3), BAND_2K (default 959), BAND_64K (default 933)
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
R=${R:-${DS4_BOX33:?}}; BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}; MODEL=${MODEL:-}
OUT=${OUT:-/tmp/perf_tripwire_$$}; mkdir -p "$OUT"; TOL=${TOL:-3}; BAND_2K=${BAND_2K:-959}; BAND_64K=${BAND_64K:-933}
m=${MODEL:+-m $MODEL}
ssh -o BatchMode=yes "$R" "cd $BINDIR && ./ds4-bench $m --cuda --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 65536 --step-incr 63488 --gen-tokens 0 --csv /tmp/perf_tripwire.csv" > "$OUT/bench.out" 2>&1; rc=$?
scp -q "$R:/tmp/perf_tripwire.csv" "$OUT/tripwire.csv" 2>/dev/null
python3 - "$OUT/tripwire.csv" "$TOL" "$BAND_2K" "$BAND_64K" "$rc" <<'PY'
import csv, os, sys
path, tol, b2k, b64k, rc = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
band = {2048: b2k, 65536: b64k}
flag = False; seen = 0
for r in rows:
    try:
        ctx = int(float(r.get('ctx_tokens', 0))); tps = float(r.get('prefill_tps', 0))
    except (TypeError, ValueError):
        continue
    if ctx not in band: continue
    seen += 1; dev = (tps - band[ctx]) / band[ctx] * 100.0
    below = dev < -tol; flag |= below
    print(f"   ctx={ctx} prefill={tps:.1f} t/s record={band[ctx]:.0f} dev={dev:+.1f}%{' BELOW-BAND' if below else ''}")
if seen == 0: print(f"TRIPWIRE_FAIL no rows parsed (bench rc={rc}; see bench.out)"); sys.exit(1)
print(("TRIPWIRE_FLAG" if flag else "TRIPWIRE_PASS") + f" rows={seen} tol={tol:.0f}% (informational: ABBA is the instrument for claims)")
PY
