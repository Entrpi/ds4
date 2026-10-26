#!/bin/bash
# battery: vision_members tier=1 stop=1 server=33 driver=33 trigger=always est=25 seq=20
# verdict: ^\[.*\] 3V_LEG_(PASS|FAIL)
# catches: the six Vision-Exp bars in one chain: identity vs the upstream oracle, the needle battery (the model no longer sees the image), CLI transcript identity, agent view_image A/B/C/D, the API surfaces, the seed-logit twin
# cmd: V8="$RB_OUT" bash gates/legs/vision/members.sh
# Ported into gates/legs/vision/ on 2026-09-04 (release-process §3.0): the Vision-Exp release-battery member.
# Runs ON the GB10 box-side (the tree is the rsync mirror at $T); helper scripts live beside this file.
# V8 step 3 = the release battery's new 3v leg (V1+V2+V3+V4+V7 in one script, runs ON .33,
# detached, one model process at a time).  Chains the arc's own gate drivers, staged under
# ~/gguf/, each in its own run dir under $V8, and adjudicates every member by its OWN
# verdict line (no member is re-scored here):
#   vision/identity.sh  V2 token identity vs the 110afdd oracle, SHIPPING config (packed FP8
#                      comp KV), oracle twin first (A10)  -> "VERDICT: IDENTICAL on all images"
#   vision/needles.sh  V4 needle battery, fork total == oracle total (24/24)
#   vision/cli.sh       V3 CLI /read transcript identity                  -> CLI_LEG_PASS
#   vision/agent.sh     V3 agent view_image A/B/C/D (tells adjudicated mechanically below)
#   vision/api.sh       V7 API surfaces (identity twins, surfaces, refusals, concurrency,
#                      needles x3, metrics)                              -> API_LEG_PASS
#   vision/seed.sh      seed-logit twin records (compared on the Mac: compare_seed_logits.py)
# Sentinel $V8/leg.DONE; summary $V8/leg.log ends with 3V_LEG_PASS or 3V_LEG_FAIL (n).
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
HERE=$(cd "$(dirname "$0")" && pwd)
G=${DS4_GGUF_DIR:?}
V8=${V8:-${DS4_BOX_SCRATCH:?}/vision_members_$(date +%Y%m%d_%H%M)}; mkdir -p "$V8"
OUT=$V8/leg.log; SENT=$V8/leg.DONE; rm -f "$SENT"; : > "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }
fails=0; miss(){ log "   MISS: $*"; fails=$((fails+1)); }
strip(){ tr -d '\r' < "$1" | sed 's/\x1b\[[0-9;?]*[a-zA-Z]//g'; }
[ -z "$(pgrep -x ds4-server)" ] || { log "FAIL: a ds4-server is up"; log "3V_LEG_FAIL (precondition)"; touch "$SENT"; exit 1; }
[ -z "$(pgrep -f '[v]ision_serial_probe|[v]ision_cont_probe|[d]s4_test')" ] || { log "FAIL: a probe/test is running"; log "3V_LEG_FAIL (precondition)"; touch "$SENT"; exit 1; }
log "== 3v leg out=$V8 tree md5: $(cd ${DS4_BOX_TREE:?} && md5sum ds4.c ds4_cuda.cu ds4_server.c | cut -c1-12 | tr '\n' ' ')"

# 1. V2 identity, shipping config (FORK_ENV empty = packed FP8 comp KV), oracle twin rebuilt + rerun
log "== identity (V2, shipping config)"
RD=$V8/identity FORK_ENV="" bash $HERE/identity.sh > $V8/identity.out 2>&1
# compare_identity.py prints to the driver's stdout ($V8/identity.out), not leg.log.  Bar (RECEIPT-identity-20260903):
# structure exact = ids + span + block "ok" on every row (block DIFF is adjudicated by A10 when the oracle twin
# itself differs on that row: the fork's block must equal one of the oracle runs'), continuations = every
# divergence a near-tie inside the text path's own drift band (fork logit gap <= 0.35).
python3 - $V8/identity $V8/identity.out <<'PY' | tee -a "$OUT"
import json, os, re, sys
rd, out = sys.argv[1], open(sys.argv[2]).read()
rows = [l for l in out.splitlines() if re.match(r'^\S+#\d+\s', l) and 'diverge' not in l]
bad = []
for l in rows:
    f = l.split(); name, otwin, ftwin, ids, span, block = f[0], f[1], f[2], f[3], f[4], f[5]
    if ids != 'ok' or span != 'ok' or ftwin != 'ok': bad.append(name + ':structure')
    if block != 'ok':
        img = name.split('#')[0]; blocks = {}
        for n in ('oracle1', 'oracle2', 'fork1', 'fork2'):
            for line in open(os.path.join(rd, n + '.jsonl')):
                if line.startswith('{'):
                    d = json.loads(line)
                    if os.path.basename(d['image']) == img: blocks[n] = d.get('block_fnv1a64')
        if not (blocks.get('fork1') == blocks.get('fork2') and blocks.get('fork1') in (blocks.get('oracle1'), blocks.get('oracle2')) and otwin != 'ok'):
            bad.append(name + ':block')
        else: print('   A10: %s block == %s (oracle twin differs on it)' % (name, [k for k in ('oracle1','oracle2') if blocks.get(k) == blocks.get('fork1')]))
gaps = [(m.group(1), float(m.group(2))) for m in re.finditer(r'^(\S+#\d+): diverge.*?\(gap ([0-9.]+)\)', out, re.M)]
for name, g in gaps:
    if g > 0.35: bad.append('%s:gap %.3f' % (name, g))
print('   identity rows %d, gaps max %.3f, %s' % (len(rows), max([g for _, g in gaps] + [0.0]), 'IDENTITY_PASS' if rows and not bad else 'IDENTITY_FAIL ' + ' '.join(bad)))
PY
grep -q "IDENTITY_PASS" "$OUT" || miss "identity bar"
grep -q "LEG_FAIL" $V8/identity/leg.log && miss "identity LEG_FAIL"

# 2. V4 needles (fork on the shipping config; the PASS band is the oracle's own total)
log "== needles (V4)"
RD=$V8/needle FORK_ENV="" bash $HERE/needles.sh > $V8/needle.out 2>&1
ot=$(grep -o 'total: [0-9]*/[0-9]*' $V8/needle/oracle.score.txt 2>/dev/null | tail -1)
ft=$(grep -o 'total: [0-9]*/[0-9]*' $V8/needle/fork.score.txt 2>/dev/null | tail -1)
log "   oracle $ot | fork $ft"
[ -n "$ft" ] && [ "$ft" = "$ot" ] || miss "needle totals (oracle '$ot' fork '$ft')"
grep -q "LEG_FAIL" $V8/needle/leg.log && miss "needle LEG_FAIL"

# 3. V3 CLI
log "== CLI (V3 G1)"
STAMP=v8_$(basename $V8) OUT=$V8/cli bash $HERE/cli.sh > $V8/cli.out 2>&1   # 09-04 reorg fix: the member's dir is passed, never derived (the chain read $G/vision_cli_* while the member wrote $DS4_BOX_SCRATCH/vision_cli_*)
CLI=$V8/cli
log "   $(grep -E 'CLI_LEG_(PASS|FAIL)|REPL_TOKENS' $CLI/leg.log | tail -2 | tr '\n' ' ')"
grep -q "CLI_LEG_PASS" $CLI/leg.log || miss "CLI verdict"

# 4. V3 agent (tells from the inc 5 run-3 adjudication)
log "== agent (V3 G2)"
STAMP=v8_$(basename $V8) OUT=$V8/agent bash $HERE/agent.sh > $V8/agent.out 2>&1
AG=$V8/agent
a_edit=$(grep -c "A_EDIT_OK" $AG/leg.log)
a_eng=$(strip $AG/ptyA.log | grep -c "image prefill engaged")   # the engine prints it on stderr (pty), not the trace
b_sw=$(strip $AG/ptyB.log | grep -c "switched to session")
b_ext=$(grep -oE "prefill tool_round=[0-9]+ transcript=[0-9]+ prompt=[0-9]+ cached=[1-9][0-9]* suffix=[1-9][0-9]*" $AG/traceB.log 2>/dev/null | head -1)
b_noemb=$(grep -c "has no embedding" $AG/traceB.log $AG/ptyB.log 2>/dev/null | awk -F: '{s+=$2} END{print s+0}')
c_ret=$(grep -c "compaction retained" $AG/traceC.log 2>/dev/null)
c_ans=$(python3 - "$AG/traceC.log" <<'PYX'
import re,sys
# the agent's generated tokens after the LAST assistant marker, joined: the TUI redraws can split "hello" and
# ".c" across cursor moves so a screen-scrape misses the contiguous string (09-05 confirmation run)
toks=[]
for l in open(sys.argv[1], errors="replace"):
    m=re.search(r'text="((?:[^"\\]|\\.)*)"', l)
    if not m: continue
    t=m.group(1)
    if "Assistant" in t: toks=[]
    else: toks.append(t)
print(1 if "hello.c" in "".join(toks) else 0)
PYX
)
[ "$c_ans" -ge 1 ] 2>/dev/null || c_ans=$(strip $AG/ptyC.log | grep -ci "hello\.c")
d_img=$(grep -c "image prefill engaged\|view_image" $AG/traceD.log 2>/dev/null)
log "   A edit=$a_edit engaged=$a_eng | B switched=$b_sw extend='${b_ext:-none}' noemb=$b_noemb | C retained=$c_ret hello.c=$c_ans | D image-lines=$d_img"
[ "$a_edit" -ge 1 ] || miss "agent A edit"
[ "$a_eng" -ge 1 ] || miss "agent A image prefill engaged"
[ "$b_sw" -ge 1 ] || miss "agent B switch"
[ -n "$b_ext" ] || miss "agent B checkpoint extended (cached>0 suffix>0)"
[ "$b_noemb" = 0 ] || miss "agent B 'has no embedding'"
[ "$c_ret" -ge 1 ] || miss "agent C compaction retained"
[ "$c_ans" -ge 1 ] || miss "agent C post-compaction answer"
[ "$d_img" = 0 ] || miss "agent D text-only control saw image lines"
echo "AGENT_LEG_$([ $fails = 0 ] && echo PASS || echo CHECK)" | tee -a $AG/leg.log >> "$OUT"

# 5. V7 API surfaces (server boot --no-spec --no-mtp, run-4 client)
log "== API (V7 surfaces)"
RD=$V8/api bash $HERE/api.sh > $V8/api.out 2>&1
log "   $(grep -E 'API_LEG_(PASS|FAIL)' $V8/api/leg.log | tail -1)"
grep -q "API_LEG_PASS" $V8/api/leg.log || miss "API verdict"

# 6. seed-logit twin records (Mac compares)
log "== API seed twin"
RD=$V8/apiseed bash $HERE/seed.sh > $V8/apiseed.out 2>&1
log "   $(grep -E 'records:|SEED_LEG' $V8/apiseed/leg.log | tail -2 | tr '\n' ' ' | cut -c1-200)"
grep -q "SEED_LEG_FAIL" $V8/apiseed/leg.log && miss "seed leg FAIL"
grep -q "records: server=8 serial=8" $V8/apiseed/leg.log || miss "seed records 8/8"
grep -q "SEED_TWIN_PASS" $V8/apiseed/leg.log || miss "seed twin verdict"

log "   servers left: $(pgrep -cx ds4-server) probes left: $(pgrep -fc '[v]ision_serial_probe|[v]ision_cont_probe')"
[ $fails = 0 ] && log "3V_LEG_PASS" || log "3V_LEG_FAIL ($fails)"
log "== LEG_DONE"; touch "$SENT"
[ $fails = 0 ]   # rc follows the verdict (review 09-05, Grok 5)
