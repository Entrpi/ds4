#!/bin/bash
# gates/legs/agent/teb.sh — standing tool-eval-bench release gate for the
# batched serving stack (v0.2 charter, workstream 3: robustness gates).
#
# Drives a GB10 over SSH: boots the ship-config ds4-server FRESH per leg and
# runs tool-eval-bench from this machine through an SSH tunnel, health-gating
# every leg on the server log. Reuses a live weight server on the host if one
# is up (never starts or stops one); otherwise the server self-loads weights.
# NOTE: each boot kills any running ds4-server on the host — do not point this
# at a box that is serving real traffic.
#
# Legs (select with LEGS="crash fast think"; default all, in this order):
#   crash — fresh boot PINNED at max_seq=4; two stages on the SAME boot:
#             crash1: TC-36 TC-37 TC-38  (category L — builds deep warm trunks)
#             crash2: TC-01 TC-07        (shallow admits beside the trunks)
#           Placement-deterministic repro for the tokentile comp-mirror OOB
#           class (fixed at f16820c). Health-gated; no score floor.
#   fast  — fresh boot; FULL 69-scenario suite (includes category L);
#           deepseek-chat = cont+DSpark ship path. Band 81-86; floor SCORE_MIN.
#   think — fresh boot; full suite; deepseek-v4-flash = thinking. Since
#           spec+tools, thinking+tools rides cont+DSpark (engagement gate =
#           accept lines, not serial starts). Band 81-86; floor SCORE_MIN.
#
# Opt-in legs (NOT in default LEGS until banded twice — health+engagement
# gated only, no score floor; record bands here once stamped):
#   hard    — fresh boot; Hard Mode only (--hardmode-only, TC-70..84,
#             15 adversarial/stateful/recovery scenarios).
#             Stamp 1 (2026-07-14, 32e1dab): 73. Band: TBD (needs stamp 2).
#   errrate — fresh boot; full suite with --error-rate $ERR_RATE injected
#             tool failures (per-scenario RNG seeded from SEED+scenario
#             digest → deterministic at fixed seed).
#             Stamp 1 (2026-07-14, 32e1dab, rate 0.2): 82 — inside the
#             normal 81-86 band. Band: TBD (needs stamp 2).
#   passk   — fresh boot; full suite --trials $TRIALS on ONE boot; parses
#             trial_statistics from the JSON (final_score_mean, pass@k,
#             pass^k, reliability_gap). Bands: TBD.
#             Stamp 1 (2026-07-14) INVALID for banding: the :$TUNNEL_PORT
#             ssh tunnel died mid-trial-3 → 15 consecutive "All connection
#             attempts failed" zeros (server healthy throughout). Valid
#             signal: trials 1-2 scored 82/86 with only single-scenario
#             flips. The conn-cascade guard below now fails the leg as
#             ENVIRONMENT instead of recording a poisoned score.
#
# Laws (earned 2026-07-13 — do not relax):
#   * Crash legs MUST run at max_seq=4. Batch fit tracks MemAvailable, so the
#     boot pins DS4_SERVER_COALESCE_MAX and ASSERTS the booted max_seq: a
#     drifted bank count changes placement and the repro silently evaporates.
#   * Any warm-off control leg must include category L (TC-36..40): the
#     64-scenario subset skips the deep trunks and proves nothing here.
#   * Health gate per leg: 0 'illegal', 0 'continuous batch failed', rc=0.
#     Engagement is part of the gate: every leg must log CONT_MTP_ACCEPT(DSpark)
#     (thinking+tools is batchable since spec+tools; 'prompt start' = serial
#     fallback marker, reported per leg) — a leg that "passes" without
#     engaging the path under test proved nothing.
#   * Full-suite temp-0 scores flip ±1 scenario run-to-run (band 81-86):
#     don't chase single flips; SCORE_MIN catches real regressions.
#
# Env overrides (defaults in parens):
#   TEB_GATE_HOST   (${DS4_BOX33:?})   SSH alias of the GB10 gate box
#   TEB_DIR         (${DS4_TEB_DIR:?}) bench checkout with .venv
#   TEB_GATE_OUT    (/tmp/teb_gates_<ts>)  local artifact dir
#   LEGS            (crash fast think)     legs to run, in order
#   SCORE_MIN       (79)                   hard floor for fast/think legs
#   TEB_SAFETY_KNOWN (TC-60)               space-separated scenario ids whose
#                                          CRITICAL model-safety warnings are
#                                          known/tracked. All criticals surface
#                                          loudly; NEW ones (off the allowlist)
#                                          get an extra banner. None FAIL the
#                                          gate — model-content safety is
#                                          model-alignment, not an engine fix.
#   MAXSEQ          (4)                    pinned+asserted bank count
#   CTX (49152)  PORT (8000)  TUNNEL_PORT (18000)  HEADROOM_MB (6272)  SEED (7)
#   BINDIR (${DS4_BOX_TREE:?})     server tree on the host
#   ERR_RATE (0.2)   injected tool-error rate for the errrate leg
#   TRIALS (3)       trial count for the passk leg
#   EXTRA_ARGS ("")  extra ds4-server flags spliced after -c (e.g. "--vision FILE")
#   EXTRA_ENV ("")   extra server env spliced into every boot, e.g.
#     EXTRA_ENV="DS4_CUDA_FP8_KV=1 DS4_CUDA_FP4_INDEX=1"
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${TEB_GATE_HOST:-${DS4_BOX33:?}}
TEB=${TEB_DIR:-${DS4_TEB_DIR:?}}
OUT=${TEB_GATE_OUT:-/tmp/teb_gates_$(date +%Y%m%d_%H%M%S)}
LEGS=${LEGS:-crash fast think}
SCORE_MIN=${SCORE_MIN:-79}
MAXSEQ=${MAXSEQ:-4}
CTX=${CTX:-49152}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
HEADROOM_MB=${HEADROOM_MB:-6272}
SEED=${SEED:-7}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
GGUF=${GGUF:-${DS4_GGUF_DIR:?}}
BASE=$GGUF/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
MTP=${MTP-$GGUF/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf}   # MTP="" boots --no-mtp (MTP-droppable legs; blocks launch-default auto-attach)
DRAFTER=$GGUF/DSpark-drafter-Q2K-Q8.gguf
MAN=${MAN:-/tmp/ds4_weights_hermes.manifest}
RWORK=/tmp/teb_gates
FAST_MODEL=${FAST_MODEL:-deepseek-chat}
THINK_MODEL=${THINK_MODEL:-deepseek-v4-flash}
ERR_RATE=${ERR_RATE:-0.2}
TRIALS=${TRIALS:-3}
EXTRA_ENV=${EXTRA_ENV:-}
EXTRA_ARGS=${EXTRA_ARGS:-}   # extra ds4-server flags spliced after -c (e.g. "--vision FILE", v0.7.0 3d restamp)

mkdir -p "$OUT"
DRV=$OUT/driver.log
SENT=$OUT/sentinel.log
rm -f "$SENT"; : > "$DRV"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$DRV"; }
fail(){ log "FAIL: $*"; echo "RUN_DONE_tebgates exit=1" >> "$SENT"; exit 1; }

# v0.2.x counter-based health (PRIMARY): per-leg /metrics deltas mirror the
# stderr greps below, which stay as fallback for one release.  msnap fails on
# an unreachable/empty endpoint — the endpoint ships with this release, so a
# leg that cannot fetch it is itself a failure.
msnap(){ curl -s -m 10 "http://127.0.0.1:$TUNNEL_PORT/metrics" > "$1" && [ -s "$1" ]; }
mval(){ awk -v k="$2" '$1 == k {v=$2} END {printf "%.0f", v+0}' "$1" 2>/dev/null; }
mdelta(){ echo $(( $(mval "$2" "$3") - $(mval "$1" "$3") )); }

SRV=""          # remote server log of the current boot
SUMMARY=""

# Stability wait: MemAvailable delta < 512 MB between 5 s samples (and >= $1 GiB).
wait_mem(){ ssh "$R" "min=$1; prev=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo); i=0
    while [ \$i -lt 60 ]; do sleep 5
      cur=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
      d=\$((cur - prev)); [ \$d -lt 0 ] && d=\$((-d))
      if [ \$d -lt 512000 ] && [ \$((cur/1048576)) -ge \$min ]; then echo \"mem stable: \$((cur/1048576)) GiB\"; exit 0; fi
      prev=\$cur; i=\$((i+1)); done
    echo \"mem NOT stable/low: \$((cur/1048576)) GiB\"; exit 1"
}

# Fresh ship-config server boot. Uses the host's weight server iff one is live
# with a manifest; else the server self-loads the same GGUFs.
boot_server(){
  local leg=$1
  SRV=$RWORK/srv_${leg}.log
  log "boot($leg): killing old ds4-server (weight server untouched)"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 0 >/dev/null || log "boot($leg): WARN mem not stable after kill (continuing)"
  local ipc=""
  if ssh "$R" "pgrep -x ds4_weight_serv >/dev/null && [ -s $MAN ]"; then
    ipc="DS4_CUDA_WEIGHT_IPC_SCOPE=base DS4_CUDA_WEIGHT_IPC_MANIFEST=$MAN"
    log "boot($leg): weight server live — IPC mode"
  else
    log "boot($leg): no weight server — self-load mode"
  fi
  ssh "$R" ": > $SRV; cd $BINDIR; env $ipc $EXTRA_ENV \
      DS4_CUDA_NO_HBM_CACHE=1 DS4_SESSION_LAZY_GRAPH=0 \
      DS4_BATCH_FIT_HEADROOM_MB=$HEADROOM_MB DS4_SERVER_COALESCE_MAX=$MAXSEQ \
      DS4_CONT_PREFILL_CHUNK=2048 DS4_CONT_MTP_MODE=2 DS4_CONT_DSPARK=1 \
      DS4_DSPARK_MODEL=$DRAFTER DS4_CONT_CAPTURE=1 DS4_SERVER_DEFAULT_TEMP=0 \
      setsid nohup ./ds4-server -m $BASE ${MTP:+--mtp} ${MTP:---no-mtp} --cuda -c $CTX $EXTRA_ARGS --port $PORT \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3   # re-check: a too-eager liveness poll can false-negative mid-exec
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "boot($leg) BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot($leg) timeout"
  done
  local ms
  ms=$(ssh "$R" "grep -m1 -oE 'max_seq[= ][0-9]+' $SRV" 2>/dev/null | grep -oE '[0-9]+$'); ms=${ms:-0}
  [ "$ms" -eq "$MAXSEQ" ] || fail "boot($leg): max_seq=$ms != $MAXSEQ — placement drifted (MemAvailable?); crash gate invalid"
  # Tunnel: reuse if live, else start one (idempotent — bind fails if held).
  curl -s -m 5 "http://127.0.0.1:$TUNNEL_PORT/v1/models" >/dev/null 2>&1 || {
    ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
    sleep 2
    curl -s -m 10 "http://127.0.0.1:$TUNNEL_PORT/v1/models" >/dev/null || fail "boot($leg): tunnel/:$TUNNEL_PORT unreachable"
  }
  log "boot($leg): up, max_seq=$ms"
}

# run_teb <name> <model> <need:accepts|serial> [--scenarios ...]
# Runs one bench invocation against the current boot; pulls the srv-log segment
# it produced and applies the health+engagement gate.
run_teb(){
  local name=$1 model=$2 need=$3; shift 3
  local off
  off=$(ssh "$R" "wc -l < $SRV")
  msnap "$OUT/m_${name}_pre.txt" || fail "$name: /metrics unreachable pre-run"
  log "$name: bench starting (model=$model, srv offset $off)"
  ( cd "$TEB" && .venv/bin/python -m tool_eval_bench \
      --model "$model" --backend vllm --base-url "http://127.0.0.1:$TUNNEL_PORT/v1" \
      --no-probe-engine --seed "$SEED" --no-live \
      --json-file "$OUT/teb_${name}.json" "$@" \
      > "$OUT/teb_${name}.out" 2>&1 )
  local rc=$?
  local seg=$OUT/teb_${name}_srv.log
  ssh "$R" "tail -n +$((off+1)) $SRV" > "$seg"
  msnap "$OUT/m_${name}_post.txt" || fail "$name: /metrics unreachable post-run"
  # PRIMARY health: registry counter deltas over this leg.
  local mfb mser mrej mhits
  mfb=$(mdelta "$OUT/m_${name}_pre.txt" "$OUT/m_${name}_post.txt" ds4_cont_batch_failures_total)
  mser=$(mdelta "$OUT/m_${name}_pre.txt" "$OUT/m_${name}_post.txt" ds4_requests_serial_total)
  mrej=$(mdelta "$OUT/m_${name}_pre.txt" "$OUT/m_${name}_post.txt" ds4_cont_admit_rejects_total)
  mhits=$(mdelta "$OUT/m_${name}_pre.txt" "$OUT/m_${name}_post.txt" ds4_spec_hits_total)
  # FALLBACK health (one release): the stderr grep twins of the counters.
  local ill fb acc ser rej score conn
  ill=$(grep -c 'illegal' "$seg" || true)
  fb=$(grep -c 'continuous batch failed' "$seg" || true)
  acc=$(grep -c 'CONT_MTP_ACCEPT(DSpark)' "$seg" || true)
  ser=$(grep -c 'prompt start' "$seg" || true)
  rej=$(grep -c 'cont admit rejected' "$seg" || true)
  # Environment guard (learned 2026-07-14): a mid-run tunnel death scores every
  # remaining scenario 0 with connection errors while the server log stays
  # clean — that must fail the leg as ENVIRONMENT, never band as a score.
  conn=$(cat "$OUT/teb_${name}.json" 2>/dev/null "$OUT/teb_${name}.out" | grep -c 'All connection attempts failed' || true)
  score=$(grep -o '"final_score": *[0-9]*' "$OUT/teb_${name}.out" | tail -1 | grep -o '[0-9]*$'); score=${score:-none}
  # Safety surface (2026-07-17, thread post #37; policy set same day): a
  # CRITICAL bench safety warning must never ride a green gate SILENTLY, but
  # it must not FAIL the gate either. This engine owns well-formed tool
  # handling — a tool call parses under the structural sampler or is never
  # emitted — NOT refusing adversarial/injected actions. Whether the model
  # obeys an instruction injected in a prior turn is model-alignment
  # behavior, out of the engine's fix domain, so we surface it, we don't
  # block a release on it and we do not bolt red-team-blocking heuristics
  # onto the serving layer. Every CRITICAL is printed loudly; NEW ones (not
  # on TEB_SAFETY_KNOWN) get an extra banner. Do NOT inspect the scenario
  # payloads — they are red-team prompts (TC-60 = cross-turn sleeper).
  local crit c known_crit new_crit all_crit
  crit=$(grep -oE 'TC-[0-9]+ \([^)]+\): CRITICAL' "$OUT/teb_${name}.json" 2>/dev/null | grep -oE '^TC-[0-9]+' | sort -u)
  known_crit=""; new_crit=""
  for c in $crit; do
    case " ${TEB_SAFETY_KNOWN:-TC-60} " in
      *" $c "*) known_crit="$known_crit$c " ;;
      *)        new_crit="$new_crit$c " ;;
    esac
  done
  all_crit="$known_crit$new_crit"
  log "$name: rc=$rc score=$score counters[fail=$mfb serial=$mser rejects=$mrej spec_hits=$mhits] greps[illegal=$ill fallbacks=$fb accepts=$acc serial_starts=$ser admit_rejects=$rej]"
  [ -z "$known_crit" ] || log "$name: NOTE — CRITICAL model-safety warning(s), KNOWN tracked (allowlist TEB_SAFETY_KNOWN): $known_crit"
  [ -z "$new_crit" ]   || log "$name: WARNING — NEW CRITICAL model-safety warning(s): $new_crit(not on TEB_SAFETY_KNOWN — surface to the user; model-alignment class, not an engine gate-fail)"
  SUMMARY="$SUMMARY
  $name: score=$score rc=$rc cnt_fail=$mfb cnt_serial=$mser cnt_rejects=$mrej cnt_spec_hits=$mhits illegal=$ill fallbacks=$fb accepts=$acc serial_starts=$ser admit_rejects=$rej safety_critical=${all_crit:-none}"
  [ "$mfb" -eq 0 ] || fail "$name: ds4_cont_batch_failures_total +$mfb (counter)"
  [ "$ill" -eq 0 ] || fail "$name: illegal memory access in server log ($seg)"
  [ "$fb" -eq 0 ] || fail "$name: continuous-batch fallback in server log ($seg)"
  [ "$conn" -eq 0 ] || fail "$name: $conn connection-failure scenario(s) — ENVIRONMENT (tunnel/:$TUNNEL_PORT died mid-run?), score invalid"
  [ $rc -eq 0 ] || fail "$name: bench rc=$rc (see $OUT/teb_${name}.out)"
  case $need in
    accepts) [ "$mhits" -gt 0 ] || fail "$name: ds4_spec_hits_total +0 — cont/DSpark path not engaged (counter)"
             [ "$acc" -gt 0 ] || fail "$name: no CONT_MTP_ACCEPT(DSpark) lines — cont/DSpark path not engaged" ;;
    serial)  [ "$mser" -gt 0 ] || fail "$name: ds4_requests_serial_total +0 — serial path not engaged (counter)"
             [ "$ser" -gt 0 ] || fail "$name: no 'prompt start' lines — serial path not engaged" ;;
  esac
  LAST_SCORE=$score
}

check_floor(){
  local name=$1
  [ "$LAST_SCORE" != "none" ] || fail "$name: no final_score in bench output"
  [ "$LAST_SCORE" -ge "$SCORE_MIN" ] || fail "$name: score $LAST_SCORE < floor $SCORE_MIN"
  if [ "$LAST_SCORE" -lt 81 ] || [ "$LAST_SCORE" -gt 86 ]; then
    log "$name: NOTE score $LAST_SCORE outside historical band 81-86"
  fi
}

log "teb_gates: host=$R legs=[$LEGS] max_seq=$MAXSEQ ctx=$CTX headroom=${HEADROOM_MB}MB seed=$SEED out=$OUT"
ssh "$R" "true" || fail "cannot ssh to $R"
[ -x "$TEB/.venv/bin/python" ] || fail "tool-eval-bench venv not found at $TEB/.venv"

for legname in $LEGS; do
  case $legname in
    crash)
      boot_server crash
      run_teb crash1 "$FAST_MODEL" accepts --scenarios TC-36 TC-37 TC-38
      run_teb crash2 "$FAST_MODEL" accepts --scenarios TC-01 TC-07
      ;;
    fast)
      boot_server fast
      run_teb fast "$FAST_MODEL" accepts
      check_floor fast
      ;;
    think)
      boot_server think
      run_teb think "$THINK_MODEL" accepts
      check_floor think
      ;;
    hard)
      boot_server hard
      run_teb hard "$FAST_MODEL" accepts --hardmode-only
      log "hard: opt-in leg, no floor until banded (Hard Mode is ceiling-breaking by design)"
      ;;
    errrate)
      boot_server errrate
      run_teb errrate "$FAST_MODEL" accepts --error-rate "$ERR_RATE"
      log "errrate: opt-in leg rate=$ERR_RATE, no floor until banded"
      ;;
    passk)
      boot_server passk
      run_teb passk "$FAST_MODEL" accepts --trials "$TRIALS"
      PKSTATS=$(python3 - "$OUT/teb_passk.json" <<'PYEOF'
import json, sys
t = json.load(open(sys.argv[1])).get("trial_statistics", {})
print(f"mean={t.get('final_score_mean')} stddev={t.get('final_score_stddev')} "
      f"pass@k={t.get('pass_at_k')} pass^k={t.get('pass_hat_k')} "
      f"gap={t.get('reliability_gap')}")
PYEOF
) || fail "passk: no trial_statistics in $OUT/teb_passk.json"
      log "passk: trials=$TRIALS $PKSTATS (no floor until banded)"
      SUMMARY="$SUMMARY
  passk trials=$TRIALS: $PKSTATS"
      ;;
    *) fail "unknown leg '$legname'" ;;
  esac
done

log "teb_gates: ALL LEGS GREEN$SUMMARY"
echo "RUN_DONE_tebgates exit=0" >> "$SENT"
