#!/bin/bash
# speed-bench/toolcall_depth_gate.sh — deep-context tool-call protocol gate
# (field issue #18; first PASS expected on the fix lineage, 2026-08-25).
#
# Guards the failure the field found: at depth, the completion-report tool
# call degrades into plain text (or frayed DSML the parser rejects). The
# needle harness grows a capture-shaped tool conversation to ~112K prompt
# tokens and triggers completion off a TOOL RESULT (no user cue). PASS =
# tool_call at EVERY depth checkpoint, both with the OpenAI reasoning_effort
# field sent ("high" — compat-mapped, must render prefix-free) and without.
#
# Laws:
#   * GREEDY leg is the gate (deterministic); the field-sampling leg is an
#     A/B instrument, not a gate — sampled flips are probabilistic.
#   * reasoning_effort=high vs omitted must land the SAME prompt_tokens per
#     checkpoint (the issue-#18 compat clamp: client fields never inject the
#     effort preamble unless --reasoning-effort-native).
#   * Each boot kills any prior gate server on the host; the run kills its
#     own server on exit (PASS or FAIL).
#
# Env: TG_HOST (sync-192_168_88_33)  BINDIR (/home/ent/code/ds4-phase0)
#      GGUF (/home/ent/gguf)  SEEDS (1,2,3)  PORT (8078)
#      TG_EXTRA_ENV — extra server env (e.g. "DS4_REASONING_REPLAY=drop
#      DS4_TOOL_SLIP_RESAMPLE=1" for the v0.6.4 knob-armed arm)
set -u
R=${TG_HOST:-sync-192_168_88_33}
BINDIR=${BINDIR:-/home/ent/code/ds4-phase0}
GGUF=${GGUF:-/home/ent/gguf}
SEEDS=${SEEDS:-1,2,3}
PORT=${PORT:-8078}
EXTRA_ENV=${TG_EXTRA_ENV:-}
WORK=/tmp/toolcall_depth_gate.$$
MODEL="$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"
DRAFTER="$GGUF/DSpark-drafter-Q2K-Q8-0731.gguf"
HERE=$(cd "$(dirname "$0")" && pwd)

fail() { echo "TOOLCALL-DEPTH GATE: FAIL — $*"; cleanup; exit 1; }
cleanup() {
    # TG_KEEP=<local dir>: keep the per-draw jsonl (flake_retry_first_draw
    # disclosures) as receipts before the work dir goes away.
    if [ -n "${TG_KEEP:-}" ]; then mkdir -p "$TG_KEEP"; scp -q "$R:$WORK/gate_*.jsonl" "$TG_KEEP/" 2>/dev/null; fi
    ssh "$R" "p=\$(cat $WORK/srv.pid 2>/dev/null); [ -n \"\$p\" ] && kill \$p 2>/dev/null; sleep 2; [ -n \"\$p\" ] && kill -9 \$p 2>/dev/null; rm -rf $WORK" 2>/dev/null
}

ssh "$R" "mkdir -p $WORK" || fail "ssh/mkdir"
scp -q "$HERE/needle_toolcall_harness.py" "$R:$WORK/" || fail "scp harness"

ssh "$R" "setsid --fork sh -c 'env DS4_SESSION_LAZY_GRAPH=0 DS4_SERVER_FORK=0 \
  DS4_SERVER_COALESCE_MAX=2 DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 \
  DS4_DSPARK_MODEL=$DRAFTER DS4_BATCH_VMM_BUDGET_MB=4096 $EXTRA_ENV \
  $BINDIR/ds4-server --cuda -m $MODEL --dspark $DRAFTER --no-mtp -c 131072 \
  --kv-disk-dir $WORK/kv --kv-disk-space-mb 16384 --port $PORT \
  > $WORK/srv.log 2>&1 & echo \$! > $WORK/srv.pid' </dev/null >/dev/null 2>&1"

for i in $(seq 1 180); do
    code=$(ssh "$R" "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/stats" 2>/dev/null)
    [ "$code" = 200 ] && break
    alive=$(ssh "$R" "kill -0 \$(cat $WORK/srv.pid) 2>/dev/null && echo up" 2>/dev/null)
    [ "$alive" = up ] || fail "BOOT-DIED: $(ssh "$R" "tail -3 $WORK/srv.log" 2>/dev/null | tr '\n' ' ')"
    [ "$i" = 180 ] && fail "BOOT-TIMEOUT"
    sleep 5
done

ssh "$R" "python3 $WORK/needle_toolcall_harness.py $PORT gate-high high $SEEDS $WORK/gate_high.jsonl greedy 1 tool" \
    || fail "harness (effort=high) errored"
ssh "$R" "python3 $WORK/needle_toolcall_harness.py $PORT gate-omit omit $SEEDS $WORK/gate_omit.jsonl greedy 1 tool" \
    || fail "harness (effort omitted) errored"

VERDICT=$(ssh "$R" "python3 - $WORK" <<'EOF'
import json, sys
w = sys.argv[1]
bad = []
pt = {}
for arm in ("high", "omit"):
    for line in open("%s/gate_%s.jsonl" % (w, arm)):
        r = json.loads(line)
        key = (r["seed"], r["depth_target"])
        if r["outcome"] != "tool_call":
            bad.append((arm, key, r["outcome"], (r.get("content_head") or r.get("detail") or "")[:80]))
        pt.setdefault(key, {})[arm] = r.get("prompt_tokens")
mismatch = [(k, v) for k, v in pt.items()
            if v.get("high") is not None and v.get("omit") is not None
            and v["high"] != v["omit"]]
for b in bad:
    print("NONTOOL", b)
for m in mismatch:
    print("PT-MISMATCH", m)
print("VERDICT", "PASS" if not bad and not mismatch else "FAIL")
EOF
)
echo "$VERDICT"
cleanup
case "$VERDICT" in
    *"VERDICT PASS"*) echo "TOOLCALL-DEPTH GATE: ALL DEPTHS PASS"; exit 0;;
    *) echo "TOOLCALL-DEPTH GATE: FAIL"; exit 1;;
esac
