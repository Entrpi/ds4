#!/bin/bash
# frontier_bytes_gate.sh — issue #18 option E receipt: the tool-call reminder
# on OUTPUT-ONLY Responses continuations (rendered-byte frontier bookkeeping,
# commit 1a037d7 + the receipt-instrumentation commit on top of it).
#
# INVOCATION (from the Mac, over SSH, like the other gates).  The box owner
# runs it: .33 box-lock ownership stays with the ds4 parent session, never
# run two gates against one box.
#   speed-bench/frontier_bytes_gate.sh                  # this gate only
#   WITH_SIBLINGS=1 speed-bench/frontier_bytes_gate.sh  # FIRST run
#       speed-bench/inc5_close_gate.sh and speed-bench/inc6_tool_gate.sh
#       unchanged (the existing output-only legs t2_stream / rt2_stream /
#       prot_t2 = no-regression), THEN this gate
#   Detached:  setsid nohup speed-bench/frontier_bytes_gate.sh > run.log 2>&1 < /dev/null &
#       and watch for the END sentinel line "frontier_bytes_gate END rc=" --
#       silence is not still-running.  Never invoke through `| tail`.
#   Env overrides: R (sync-192_168_88_33) BINDIR (/home/ent/code/ds4-phase0)
#       PORT (8000) TUNNEL_PORT (18000) CTX (65536) DEEP_BYTES (140000)
#       CTRL_BYTES (24000) OUT (local/docs/issue18/frontier_bytes_gate_<ts>
#       under this checkout)
#
# PRECONDITION: BINDIR's ds4-server is built from the receipt branch
# (1a037d7 + instrumentation).  The first T2 FAILS with a clear message if
# its srv.log line lacks the frontier_bytes=/tail_bytes=/reminder= fields
# (a build that predates the branch) -- the gate never guesses.
#
# What it proves.  One boot, standard ship launch shape plus --trace
# (--tool-call-reminder default on; gate DS4_TOOL_CALL_REMINDER_MIN_BYTES
# default 98304 RENDERED bytes).  Every T2 sends ONLY the function_call_output
# (no history), the shape that never carried the reminder before 1a037d7.
# The T1 user text is filler + the tool instruction: CTRL_BYTES (~24 KB,
# well below the gate) for the control, DEEP_BYTES (~140 KB, past the gate)
# for the deep legs.  Both T2s use the SAME tool-output text, so their fed
# tails differ by exactly the reminder bytes (REM_BYTES) or not at all.
#
#   serial lane (buffered Responses tool turns stay serial; the output-only
#   T2 resolves through responses_live_continuation_prompt):
#     s_ctrl_t1  control T1 -> function_call; stayed serial (LANE-ENTRY TRAP)
#     s_ctrl_t2  output-only T2 -> 200; srv.log "responses live continuation
#                RESPPROTO match=tool-output-ids ... frontier_bytes=N
#                tail_bytes=B reminder=0" with 0 < N < gate; the trace's
#                "--- live continuation suffix ---" section (the FED tail)
#                holds the tool output and NOT the reminder text; resolved +1
#     s_deep_t1  deep T1 -> function_call
#     s_deep_t2  output-only T2 -> 200; frontier_bytes=N >= gate;
#                reminder=1; the fed tail in the trace CONTAINS "[Reminder:
#                respond with exactly one tool call" (bytes, not inference);
#                tail_bytes = control tail_bytes + REM_BYTES exactly;
#                resolved +1
#     s_deep_t3  second hop: T2's function_call answered output-only again ->
#                frontier_bytes ADVANCED past N + tail (the continued record's
#                frontier, not a re-seed) and reminder=1 again
#   bank lane (STREAMING tool turns ride cont; the output-only T2 claims its
#   bank in place; the cont lane has no trace, so the receipt is the srv.log
#   "cont bank continuation admit bank=.. cached=.. suffix=.. frontier_bytes=N
#   tail_bytes=B reminder=R" line):
#     b_ctrl_t1 / b_ctrl_t2 / b_deep_t1 / b_deep_t2 / b_deep_t3  the same
#                shapes streamed, the same N / B / R assertions, plus the
#                cont-lane entry and bank-continuation decision counters
#
# Receipts: per-leg bodies + responses, srv.log, trace.log, and summary.txt
# (one PASS/FAIL line per leg with the parsed numbers) in $OUT.
# End state: ds4-server killed, box left free.
set -uo pipefail

R=${R:-sync-192_168_88_33}
BINDIR=${BINDIR:-/home/ent/code/ds4-phase0}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-65536}
DEEP_BYTES=${DEEP_BYTES:-140000}
CTRL_BYTES=${CTRL_BYTES:-24000}
GATE_BYTES=${GATE_BYTES:-98304}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$ROOT/local/docs/issue18/frontier_bytes_gate_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
RWORK=/tmp/frontier_bytes_gate
BASE="http://127.0.0.1:$TUNNEL_PORT"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
note(){ log "$*"; echo "$*" >> "$SUMMARY"; }
fail(){ note "FAIL: $*"; exit 1; }
RC=1
trap 'echo "frontier_bytes_gate END rc=$RC receipts=$OUT"' EXIT

# The reminder text byte-for-byte (ds4_server.c DS4_TOOL_CALL_REMINDER_TEXT)
# and the EOS marker: the deep tail must exceed the control tail by exactly
# REM_BYTES, and a second hop's frontier must advance by more than the fed
# tail minus the EOS the tail re-emits.
REM_HEAD='[Reminder: respond with exactly one tool call'
REM_BYTES=$(printf '\n\n[Reminder: respond with exactly one tool call in the required format. Plain-text replies are not accepted by this harness.]' | wc -c | tr -d ' ')
EOS_BYTES=$(printf '<｜end▁of▁sentence｜>' | wc -c | tr -d ' ')

# ---- no-regression siblings first (their own boots; unchanged) ------------
if [ "${WITH_SIBLINGS:-0}" = "1" ]; then
  for sib in inc5_close_gate.sh inc6_tool_gate.sh; do
    log "sibling: $sib"
    "$ROOT/speed-bench/$sib" > "$OUT/$sib.log" 2>&1
    rc=$?
    [ $rc -eq 0 ] || fail "sibling $sib rc=$rc (see $OUT/$sib.log)"
    note "sibling $sib PASS (no-regression on the existing output-only legs)"
  done
fi

tunnel_up(){
  curl -s -m 5 "$BASE/v1/models" >/dev/null 2>&1 && return 0
  ssh -f -N -L "$TUNNEL_PORT:127.0.0.1:$PORT" "$R" 2>/dev/null || true
  sleep 2
  curl -s -m 10 "$BASE/v1/models" >/dev/null 2>&1
}

wait_mem(){
  local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge "$1" ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable ${got:-?}G never reached ${1}G"
    sleep 5
  done
}

boot(){
  SRV=$RWORK/srv.log
  TRACE=$RWORK/trace.log
  log "boot: killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; : > $TRACE; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT --trace $TRACE \
      > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV 2>/dev/null; exit \$?" 2>/dev/null; do
    if ! ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null; then
      sleep 3
      ssh "$R" "pgrep -x ds4-server >/dev/null; exit \$?" 2>/dev/null || \
        fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" 2>/dev/null | tr '\n' ' ')"
    fi
    sleep 10; n=$((n+10)); [ $n -ge 1200 ] && fail "boot timeout"
  done
  tunnel_up || fail "tunnel :$TUNNEL_PORT unreachable"
  note "boot: up (BINDIR=$BINDIR rev=$(ssh "$R" "cd $BINDIR && git rev-parse --short HEAD 2>/dev/null || echo unknown") ctx=$CTX trace=$TRACE)"
}

post(){ # $1=name $2=path $3=body-file -> writes $OUT/$1.json, echoes http code
  curl -s -m 900 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d @"$3"
}
sse(){ # $1=name $2=path $3=body-file -> writes $OUT/$1.sse, echoes http code
  curl -s -m 900 --no-buffer -o "$OUT/$1.sse" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d @"$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
shas(){ grep -q "$2" "$OUT/$1.sse" || fail "$1: missing [$2] in the stream"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1".* 2>/dev/null))"; }
# METRICS-HELPER TRAP: extract the number BEFORE head -1.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
srv_count(){ ssh "$R" "grep -cF \"$1\" $RWORK/srv.log" 2>/dev/null; }
# The LAST srv.log line holding the marker: requests are synchronous, so it
# is this leg's line.
srv_last(){ ssh "$R" "grep -F \"$1\" $RWORK/srv.log | tail -n 1" 2>/dev/null; }
field(){ # $1=line $2=name -> integer value or empty
  printf '%s\n' "$1" | grep -oE "(^| )$2=[0-9]+" | grep -oE '[0-9]+$' | head -1
}
# Parse a continuation line's receipt fields into F (frontier_bytes), B
# (tail_bytes), REM (reminder); the missing-field case is the build-predates
# failure, named as such.
parse_receipt(){ # $1=leg $2=line
  [ -n "$2" ] || fail "$1: continuation log line missing in srv.log"
  F=$(field "$2" frontier_bytes); B=$(field "$2" tail_bytes); REM=$(field "$2" reminder)
  [ -n "$F" ] && [ -n "$B" ] && [ -n "$REM" ] || \
    fail "$1: line lacks frontier_bytes/tail_bytes/reminder fields -- BINDIR build predates the receipt branch (1a037d7 + instrumentation): [$2]"
}
# The FED tail on the serial lane: the last "--- live continuation suffix ---"
# section of the trace, pulled to $OUT/$1.tail.txt (bytes, for grep).
fed_tail(){ # $1=leg
  scp -q "$R:$RWORK/trace.log" "$OUT/trace_after_$1.log" || fail "$1: trace scp failed"
  python3 - "$OUT/trace_after_$1.log" "$OUT/$1.tail.txt" <<'PY' || fail "$1: no live continuation suffix section in the trace"
import sys
data = open(sys.argv[1], encoding="utf-8", errors="replace").read()
head, tail_mark = "--- live continuation suffix ---\n", "\n--- end live continuation suffix ---"
i = data.rfind(head)
if i < 0: sys.exit(1)
j = data.find(tail_mark, i)
if j < 0: sys.exit(1)
open(sys.argv[2], "w", encoding="utf-8").write(data[i + len(head):j])
PY
}

TOOLS_RESP='[{"type":"function","name":"list_files","description":"List the files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'
# One tool-output text for EVERY T2 (control and deep, both lanes) so tails
# differ by the reminder alone; it asks for a second call so the deep T3 hop
# has a function_call to answer.  It never quotes the reminder text.
T2_OUTPUT=$'file1.txt\nfile2.txt\n\nNow list /var as well: call list_files again for /var.'

# Body builders (python, stdlib).  t1_body runs inside $(...) so it must NOT
# call fail (that would exit the subshell only): it prints the user-text
# byte count and the CALLER checks its status.
# t1_body: $1=out-file $2=user-bytes $3=stream(0/1)
t1_body(){
  python3 - "$1" "$2" "$3" "$TOOLS_RESP" <<'PY'
import json, sys
out, nbytes, stream, tools = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", json.loads(sys.argv[4])
line = "note %05d: the quick brown fox jumps over the lazy dog near the riverbank at dawn.\n"
buf, n = [], 0
while sum(len(s) for s in buf) < nbytes:
    buf.append(line % n); n += 1
text = "".join(buf) + "\nThe notes above are filler.  Use the list_files tool to list the files in /tmp. Call the tool."
body = {"model": "m", "max_output_tokens": 1200, "temperature": 0,
        "input": [{"role": "user", "content": text}], "tools": tools}
if stream: body["stream"] = True
open(out, "w").write(json.dumps(body))
print(len(text))
PY
}
t2_body(){ # $1=out-file $2=call_id $3=stream(0/1)
  python3 - "$1" "$2" "$3" "$TOOLS_RESP" "$T2_OUTPUT" <<'PY' || fail "t2 body build failed"
import json, sys
out, cid, stream, tools, text = sys.argv[1], sys.argv[2], sys.argv[3] == "1", json.loads(sys.argv[4]), sys.argv[5]
body = {"model": "m", "max_output_tokens": 1200, "temperature": 0,
        "input": [{"type": "function_call_output", "call_id": cid, "output": text}],
        "tools": tools}
if stream: body["stream"] = True
open(out, "w").write(json.dumps(body))
PY
}
call_id_of(){ # $1=file (json or sse) -> first call_id
  grep -o '"call_id":"[^"]*"' "$1" | head -1 | cut -d'"' -f4
}

# ==================== boot =================================================
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot
note "gate=$GATE_BYTES REM_BYTES=$REM_BYTES EOS_BYTES=$EOS_BYTES ctrl_user_bytes=$CTRL_BYTES deep_user_bytes=$DEEP_BYTES"

# ==================== serial lane (buffered) ================================
# s_ctrl_t1: buffered tool turn stays serial (engagement check, as
# resp_buffered_serial in inc6) and mints call_id.
n=$(t1_body "$OUT/s_ctrl_t1.req" "$CTRL_BYTES" 0) || fail "s_ctrl_t1: body build failed"
dec0=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
ser0=$(m ds4_requests_serial_total)
c=$(post s_ctrl_t1 /v1/responses "$OUT/s_ctrl_t1.req")
code_is s_ctrl_t1 "$c" 200
has s_ctrl_t1 '"type":"function_call"'
S_CTRL_ID=$(call_id_of "$OUT/s_ctrl_t1.json"); [ -n "$S_CTRL_ID" ] || fail "s_ctrl_t1: no call_id"
dec1=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
ser1=$(m ds4_requests_serial_total)
[ "${dec1:-0}" -gt "${dec0:-0}" ] || fail "s_ctrl_t1: publish-hold decision never recorded"
[ "${ser1:-0}" -gt "${ser0:-0}" ] || fail "s_ctrl_t1: serial entries unmoved (${ser0:-?} -> ${ser1:-?})"
note "s_ctrl_t1 PASS (buffered tool turn stayed serial; user_bytes=$n call_id=$S_CTRL_ID)"

# s_ctrl_t2: output-only, below the gate: frontier known, NO reminder in the
# fed tail (trace bytes), resolved through the call-id lane.
t2_body "$OUT/s_ctrl_t2.req" "$S_CTRL_ID" 0
res0=$(m ds4_continuation_resolved_total)
c=$(post s_ctrl_t2 /v1/responses "$OUT/s_ctrl_t2.req")
code_is s_ctrl_t2 "$c" 200
res1=$(m ds4_continuation_resolved_total)
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "s_ctrl_t2: registry never resolved (${res0:-?} -> ${res1:-?})"
line=$(srv_last 'responses live continuation RESPPROTO')
printf '%s\n' "$line" | grep -q 'match=tool-output-ids' || fail "s_ctrl_t2: not the call-id lane: [$line]"
parse_receipt s_ctrl_t2 "$line"
[ "$F" -gt 0 ] || fail "s_ctrl_t2: frontier_bytes=$F, want > 0 (record published no frontier)"
[ "$F" -lt "$GATE_BYTES" ] || fail "s_ctrl_t2: control frontier_bytes=$F is not below the gate $GATE_BYTES (raise the gap: CTRL_BYTES)"
[ "$REM" = "0" ] || fail "s_ctrl_t2: reminder=$REM below the gate"
fed_tail s_ctrl_t2
grep -qF 'file1.txt' "$OUT/s_ctrl_t2.tail.txt" || fail "s_ctrl_t2: fed tail lacks the tool output"
! grep -qF "$REM_HEAD" "$OUT/s_ctrl_t2.tail.txt" || fail "s_ctrl_t2: reminder bytes present BELOW the gate"
S_CTRL_F=$F; S_CTRL_B=$B
note "s_ctrl_t2 PASS (frontier_bytes=$F tail_bytes=$B reminder=0; fed tail clean)"

# s_deep_t1: deep buffered tool turn (past the gate once rendered).
n=$(t1_body "$OUT/s_deep_t1.req" "$DEEP_BYTES" 0) || fail "s_deep_t1: body build failed"
c=$(post s_deep_t1 /v1/responses "$OUT/s_deep_t1.req")
code_is s_deep_t1 "$c" 200
grep -q '"type":"function_call"' "$OUT/s_deep_t1.json" || \
  fail "s_deep_t1: deep T1 answered without a tool call (PRECONDITION slip at ~$n user bytes, not a frontier failure): $(head -c 300 "$OUT/s_deep_t1.json")"
S_DEEP_ID=$(call_id_of "$OUT/s_deep_t1.json"); [ -n "$S_DEEP_ID" ] || fail "s_deep_t1: no call_id"
note "s_deep_t1 PASS (user_bytes=$n call_id=$S_DEEP_ID)"

# s_deep_t2: output-only past the gate: reminder=1 AND the reminder bytes in
# the FED tail (trace), tail longer than the control tail by exactly REM_BYTES.
t2_body "$OUT/s_deep_t2.req" "$S_DEEP_ID" 0
res0=$(m ds4_continuation_resolved_total)
c=$(post s_deep_t2 /v1/responses "$OUT/s_deep_t2.req")
code_is s_deep_t2 "$c" 200
res1=$(m ds4_continuation_resolved_total)
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "s_deep_t2: registry never resolved (${res0:-?} -> ${res1:-?})"
line=$(srv_last 'responses live continuation RESPPROTO')
printf '%s\n' "$line" | grep -q 'match=tool-output-ids' || fail "s_deep_t2: not the call-id lane: [$line]"
parse_receipt s_deep_t2 "$line"
[ "$F" -ge "$GATE_BYTES" ] || fail "s_deep_t2: frontier_bytes=$F below the gate $GATE_BYTES (raise DEEP_BYTES)"
[ "$REM" = "1" ] || fail "s_deep_t2: reminder=$REM past the gate (frontier_bytes=$F)"
[ "$B" -eq $((S_CTRL_B + REM_BYTES)) ] || fail "s_deep_t2: tail_bytes=$B, want control $S_CTRL_B + REM_BYTES $REM_BYTES = $((S_CTRL_B + REM_BYTES))"
fed_tail s_deep_t2
grep -qF "$REM_HEAD" "$OUT/s_deep_t2.tail.txt" || fail "s_deep_t2: reminder bytes ABSENT from the fed tail (log said reminder=1)"
grep -qF 'file1.txt' "$OUT/s_deep_t2.tail.txt" || fail "s_deep_t2: fed tail lacks the tool output"
S_DEEP_F=$F; S_DEEP_B=$B
note "s_deep_t2 PASS (frontier_bytes=$F tail_bytes=$B reminder=1; reminder bytes in the fed tail)"

# s_deep_t3: the chain's second hop.  The reminder-bearing T2 must itself
# call the tool again (the output asked for /var); its output-only T3
# resolves against the ADVANCED frontier: N3 > N2 + (tail - EOS), i.e. the
# continued record advanced by the fed tail plus the visible suffix, not
# re-seeded from the outputs-only prompt.
grep -q '"type":"function_call"' "$OUT/s_deep_t2.json" || \
  fail "s_deep_t3: the reminder-bearing T2 answered without a tool call -- a slip ON the reminder (finding, not a frontier failure): $(head -c 300 "$OUT/s_deep_t2.json")"
S_T3_ID=$(call_id_of "$OUT/s_deep_t2.json"); [ -n "$S_T3_ID" ] || fail "s_deep_t3: no call_id on T2"
t2_body "$OUT/s_deep_t3.req" "$S_T3_ID" 0
res0=$(m ds4_continuation_resolved_total)
c=$(post s_deep_t3 /v1/responses "$OUT/s_deep_t3.req")
code_is s_deep_t3 "$c" 200
res1=$(m ds4_continuation_resolved_total)
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "s_deep_t3: registry never resolved (${res0:-?} -> ${res1:-?})"
line=$(srv_last 'responses live continuation RESPPROTO')
printf '%s\n' "$line" | grep -q 'match=tool-output-ids' || fail "s_deep_t3: not the call-id lane: [$line]"
parse_receipt s_deep_t3 "$line"
[ "$F" -gt $((S_DEEP_F + S_DEEP_B - EOS_BYTES)) ] || fail "s_deep_t3: frontier_bytes=$F did not advance past $S_DEEP_F + $S_DEEP_B - $EOS_BYTES"
[ "$REM" = "1" ] || fail "s_deep_t3: reminder=$REM on the second hop"
fed_tail s_deep_t3
grep -qF "$REM_HEAD" "$OUT/s_deep_t3.tail.txt" || fail "s_deep_t3: reminder bytes absent from the fed tail"
note "s_deep_t3 PASS (frontier advanced $S_DEEP_F -> $F; reminder=1 on hop 2)"

# ==================== bank lane (streamed) ==================================
# b_ctrl_t1: streamed tool turn rides cont (LANE-ENTRY TRAP: cont entry AND
# serial unmoved) and publishes a BANK record.
n=$(t1_body "$OUT/b_ctrl_t1.req" "$CTRL_BYTES" 1) || fail "b_ctrl_t1: body build failed"
pub0=$(m ds4_continuation_records_published_total)
lane0=$(m 'ds4_route_requests_total{surface="openai_responses",lane="continuous"}')
ser0=$(m ds4_requests_serial_total)
c=$(sse b_ctrl_t1 /v1/responses "$OUT/b_ctrl_t1.req")
code_is b_ctrl_t1 "$c" 200
shas b_ctrl_t1 'function_call_arguments'
B_CTRL_ID=$(call_id_of "$OUT/b_ctrl_t1.sse"); [ -n "$B_CTRL_ID" ] || fail "b_ctrl_t1: no call_id in the stream"
pub1=$(m ds4_continuation_records_published_total)
lane1=$(m 'ds4_route_requests_total{surface="openai_responses",lane="continuous"}')
ser1=$(m ds4_requests_serial_total)
[ "${pub1:-0}" -gt "${pub0:-0}" ] || fail "b_ctrl_t1: no bank record published"
[ "${lane1:-0}" -gt "${lane0:-0}" ] || fail "b_ctrl_t1: never entered the cont lane"
[ "${ser1:-0}" -eq "${ser0:-0}" ] || fail "b_ctrl_t1: serial entries moved (${ser0:-?} -> ${ser1:-?})"
note "b_ctrl_t1 PASS (streamed tool turn rode cont; user_bytes=$n call_id=$B_CTRL_ID)"

# b_ctrl_t2: output-only streamed T2 claims its bank; receipt line shows the
# frontier and a reminder-free tail.
t2_body "$OUT/b_ctrl_t2.req" "$B_CTRL_ID" 1
res0=$(m ds4_continuation_resolved_total)
bank0=$(m 'ds4_route_decisions_total{reason="continuous_bank_continuation"}')
c=$(sse b_ctrl_t2 /v1/responses "$OUT/b_ctrl_t2.req")
code_is b_ctrl_t2 "$c" 200
res1=$(m ds4_continuation_resolved_total)
bank1=$(m 'ds4_route_decisions_total{reason="continuous_bank_continuation"}')
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "b_ctrl_t2: registry never resolved"
[ "${bank1:-0}" -gt "${bank0:-0}" ] || fail "b_ctrl_t2: bank-continuation decision never recorded"
line=$(srv_last 'cont bank continuation admit')
parse_receipt b_ctrl_t2 "$line"
[ "$F" -gt 0 ] || fail "b_ctrl_t2: frontier_bytes=$F, want > 0 (bank record published no frontier)"
[ "$F" -lt "$GATE_BYTES" ] || fail "b_ctrl_t2: control frontier_bytes=$F not below the gate"
[ "$REM" = "0" ] || fail "b_ctrl_t2: reminder=$REM below the gate"
[ "$B" -eq "$S_CTRL_B" ] || fail "b_ctrl_t2: tail_bytes=$B differs from the serial control tail $S_CTRL_B (same output text, same render)"
B_CTRL_F=$F; B_CTRL_B=$B
note "b_ctrl_t2 PASS (bank claim; frontier_bytes=$F tail_bytes=$B reminder=0)"

# b_deep_t1 / b_deep_t2: the deep pair streamed.
n=$(t1_body "$OUT/b_deep_t1.req" "$DEEP_BYTES" 1) || fail "b_deep_t1: body build failed"
c=$(sse b_deep_t1 /v1/responses "$OUT/b_deep_t1.req")
code_is b_deep_t1 "$c" 200
grep -q 'function_call_arguments' "$OUT/b_deep_t1.sse" || \
  fail "b_deep_t1: deep streamed T1 answered without a tool call (PRECONDITION slip at ~$n user bytes, not a frontier failure)"
B_DEEP_ID=$(call_id_of "$OUT/b_deep_t1.sse"); [ -n "$B_DEEP_ID" ] || fail "b_deep_t1: no call_id"
note "b_deep_t1 PASS (user_bytes=$n call_id=$B_DEEP_ID)"

t2_body "$OUT/b_deep_t2.req" "$B_DEEP_ID" 1
res0=$(m ds4_continuation_resolved_total)
c=$(sse b_deep_t2 /v1/responses "$OUT/b_deep_t2.req")
code_is b_deep_t2 "$c" 200
res1=$(m ds4_continuation_resolved_total)
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "b_deep_t2: registry never resolved"
line=$(srv_last 'cont bank continuation admit')
parse_receipt b_deep_t2 "$line"
[ "$F" -ge "$GATE_BYTES" ] || fail "b_deep_t2: frontier_bytes=$F below the gate"
[ "$REM" = "1" ] || fail "b_deep_t2: reminder=$REM past the gate (frontier_bytes=$F)"
[ "$B" -eq $((B_CTRL_B + REM_BYTES)) ] || fail "b_deep_t2: tail_bytes=$B, want control $B_CTRL_B + REM_BYTES $REM_BYTES"
B_DEEP_F=$F; B_DEEP_B=$B
note "b_deep_t2 PASS (bank claim past the gate; frontier_bytes=$F tail_bytes=$B reminder=1)"

# b_deep_t3: the bank chain's second hop advances the frontier too.
grep -q 'function_call_arguments' "$OUT/b_deep_t2.sse" || \
  fail "b_deep_t3: the reminder-bearing streamed T2 answered without a tool call (slip ON the reminder; finding)"
B_T3_ID=$(call_id_of "$OUT/b_deep_t2.sse"); [ -n "$B_T3_ID" ] || fail "b_deep_t3: no call_id on T2"
t2_body "$OUT/b_deep_t3.req" "$B_T3_ID" 1
res0=$(m ds4_continuation_resolved_total)
c=$(sse b_deep_t3 /v1/responses "$OUT/b_deep_t3.req")
code_is b_deep_t3 "$c" 200
res1=$(m ds4_continuation_resolved_total)
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "b_deep_t3: registry never resolved"
line=$(srv_last 'cont bank continuation admit')
parse_receipt b_deep_t3 "$line"
[ "$F" -gt $((B_DEEP_F + B_DEEP_B - EOS_BYTES)) ] || fail "b_deep_t3: frontier_bytes=$F did not advance past $B_DEEP_F + $B_DEEP_B - $EOS_BYTES"
[ "$REM" = "1" ] || fail "b_deep_t3: reminder=$REM on the second hop"
note "b_deep_t3 PASS (bank frontier advanced $B_DEEP_F -> $F; reminder=1 on hop 2)"

# ==================== receipts + teardown ===================================
scp -q "$R:$RWORK/srv.log" "$OUT/srv.log" || fail "srv.log scp failed"
scp -q "$R:$RWORK/trace.log" "$OUT/trace.log" || fail "trace.log scp failed"
ssh "$R" "pkill -x ds4-server; exit 0"
RC=0
note "ALL LEGS PASS — receipts in $OUT"
