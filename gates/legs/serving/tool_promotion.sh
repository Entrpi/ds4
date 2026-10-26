#!/bin/bash
# serving/tool_promotion.sh — v0.5.6 API Inc 6 live gate: tool-capable promotion
# (plan §5 Inc 6 gate line: complete/multiple/incomplete/budget-cut tool
# calls, deterministic repair, tool ID continuity, thinking signatures,
# mixed rows; §4.6 consumer + retention).
#
# What Inc 6 changed (all behind DS4_SERVER_CONT_TOOLS_ANTHROPIC /
# DS4_SERVER_CONT_TOOLS_RESPONSES, default ON, composed with the Inc 3
# per-surface switches):
#   6a  STREAMING anth/resp tool turns ride the continuous lane; the Inc 5c
#       BATCH_BANK publication at the cont finalize is their record.
#       Buffered tool turns STAY SERIAL (the corrective-retry contract has
#       no row-local cont equivalent; plan §5 allows keeping them serial).
#   6b  OUTPUT-ONLY T2s whose pinned record is BANK-owned route continuous
#       (DS4_NEED_BANK_FRONTIER) and admit via cont_registry_bank_claim +
#       generation/frontier equality IN PLACE on their own bank; any
#       failure answers the protocol-native 409 (registry miss).
#   6c  victim placement never destroys a bank whose LIVE record is inside
#       grace or hard-pinned; with EVERY candidate protected the admission
#       sheds 503 continuation_hold (the serial hold idiom, bank side).
#
# Boot 1 legs (anthropic mechanics; DEFAULT retention 60/300/60):
#   t1_stream       STREAMING /v1/messages tool turn rides CONT: SSE
#                   input_json_delta + stop_reason tool_use; published +1;
#                   lane{anthropic_messages,continuous} +1; serial unmoved
#                   (LANE-ENTRY TRAP)
#   t2_stream       zero-delay OUTPUT-ONLY streamed T2 (the cont pbt twin):
#                   200; decision{continuous_bank_continuation} +1; srv.log
#                   "cont bank continuation admit"; resolved +1; SSE usage
#                   cache_read_input_tokens > 0 (the bank KV was kept)
#   t2_repeat409    the SAME T2 again: bank records have NO eager demote --
#                   the record is LIVE but the frontier moved (T2 extended
#                   the bank) -> claim refuses, native 409, missed +1,
#                   srv.log "cont bank continuation refused" (the live
#                   consumed/ABA surface)
#   budget_cut      STREAMING T1 with a tiny budget cuts mid-toolcall on the
#                   CONT lane: either the deterministic repair completes it
#                   (srv.log "cont chat repaired unterminated tool call",
#                   record published) or it finishes honest max_tokens with
#                   NO record -- either way no strand, and the srv.log
#                   marker proves which branch ran
#   think_sig       thinking-enabled STREAMING tool turn: thinking deltas +
#                   signature_delta + tool_use in one stream; the T2
#                   resolves 200 (thinking signatures survive promotion)
#   buffered_serial engagement-NEGATIVE: a buffered anth tool turn still
#                   runs SERIAL (decision need_continuation_publish +1,
#                   cont bank admits unmoved, serial entries +1)
#
# Boot 2 legs (responses + cross-surface; DEFAULT retention):
#   rt1_stream      STREAMING /v1/responses tool turn rides CONT:
#                   function_call_arguments deltas; published +1;
#                   lane{openai_responses,continuous} +1
#   rt2_stream      output-only streamed function_call_output T2: 200,
#                   bank admit +1, resolved +1
#   rt2_repeat409   same T2 again -> native 409 + missed +1 (frontier moved)
#   multi_calls     a T1 asked for TWO list_files calls; T2 returns a
#                   tool_result for EVERY id the stream produced -- the
#                   claim's exact id-SET equality resolves (>=1 id; the
#                   multi case is logged when the model complied)
#   mixed_rows      an anth STREAMING tool T1 and a plain OpenAI cont chat
#                   run CONCURRENTLY in one continuous group; both 200
#   crossproto_bank a toolu_ bank id presented as a Responses
#                   function_call_output -> parse-side 400 (protocol-scoped
#                   registry, the bank bleed twin)
#   resp_buffered_serial engagement-NEGATIVE buffered responses tool turn
#                   stays serial
#
# Boot 3 legs (6c retention, DS4_SERVER_COALESCE_MAX=2 DS4_CONT_GRACE_S=20;
# 2 banks make protection pressure deterministic):
#   prot_t1         streamed tool T1 -> bank A holds a LIVE in-grace record
#   prot_shed       an OCCUPIER stream holds bank B; a cold chat arrives:
#                   bank A protected + bank B occupied -> NO victim -> 503
#                   + Retry-After + shed{continuation_hold} +1 + srv.log
#                   "cont admission hold shed" (the bank-side hold, live)
#   prot_t2         the output-only T2 still claims bank A -> 200 (the
#                   protection existed FOR this)
#   postgrace_ok    after the occupier ends and grace lapses the SAME cold
#                   probe shape admits 200 (protection is grace-bounded;
#                   victim eviction works again)
#
# NOT live-gated here (documented):
#   - kill switches: serving/cont_registry.sh now pins BOTH tool switches OFF
#     on every boot -- that whole gate is the switch-off oracle (Inc 5
#     behavior byte-for-byte).
#   - buffered T2 with tools on a bank record (mixed stream->buffered
#     client): keeps the buffered corrective hold -> serial resolve cannot
#     match a bank record -> honest 409 -> full replay; the routing rows
#     are unit-gated (test_route_decide_reason_table).
#   - bank pin protection for QUEUED T2s: unit-gated
#     (test_cont_registry_bank_protection); the serial twin runs live as
#     cont_registry_gate pin_wins.
#   - restart/full-replay: registry is in-memory; inc5_close_gate
#     restart_t1/restart_replay cover the identical parse-side surface.
#
# PER-BOOT FUNDED WINDOW: boots 1/2 spend 5-6 cont admissions, boot 3
# spends 4 -- all inside the ~8-10 per 16k boot budget.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box left free.
# Env overrides: R (${DS4_BOX33:?}) BINDIR (${DS4_BOX_TREE:?})
#                PORT (8000) TUNNEL_PORT (18000) CTX (16384)
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
set -uo pipefail

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/inc6_tool_gate
OUT=${OUT:-/tmp/inc6_tool_gate_$$}
mkdir -p "$OUT"
BASE="http://127.0.0.1:$TUNNEL_PORT"

log(){ echo "[$(date +%H:%M:%S)] $*"; }
fail(){ log "FAIL: $*"; exit 1; }

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
  log "boot: killing old ds4-server on $R"
  ssh "$R" "pkill -x ds4-server; sleep 2; pkill -9 -x ds4-server; mkdir -p $RWORK; rm -f /tmp/ds4.lock; exit 0"
  wait_mem 100
  ssh "$R" ": > $SRV; cd $BINDIR; env ${BOOT_ENV:-} setsid nohup ./ds4-server -c $CTX --port $PORT \
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
  log "boot: up"
}

post(){
  curl -s -m 240 -o "$OUT/$1.json" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
sse(){ # $1=name $2=path $3=body -> writes $OUT/$1.sse, echoes http code
  curl -s -m 240 --no-buffer -o "$OUT/$1.sse" -w '%{http_code}' "$BASE$2" \
       -H 'Content-Type: application/json' -d "$3"
}
has(){ grep -q "$2" "$OUT/$1.json" || fail "$1: missing [$2] in $(head -c 300 "$OUT/$1.json")"; }
shas(){ grep -q "$2" "$OUT/$1.sse" || fail "$1: missing [$2] in the stream"; }
code_is(){ [ "$2" = "$3" ] || fail "$1: HTTP $2, want $3 ($(head -c 300 "$OUT/$1".* 2>/dev/null))"; }
# METRICS-HELPER TRAP: extract the number BEFORE head -1.
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
srv_count(){ ssh "$R" "grep -cF \"$1\" $RWORK/srv.log" 2>/dev/null; }

TOOLS_ANTH='[{"name":"list_files","description":"List the files in a directory","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'
TOOLS_RESP='[{"type":"function","name":"list_files","description":"List the files in a directory","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}]'

# ==================== boot 1: anthropic mechanics (default retention) =======
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

# t1_stream: the promotion itself.  LANE-ENTRY TRAP: assert the cont entry
# AND that serial never moved.
pub0=$(m ds4_continuation_records_published_total)
lane0=$(m 'ds4_route_requests_total{surface="anthropic_messages",lane="continuous"}')
ser0=$(m ds4_requests_serial_total)
c=$(sse t1_stream /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is t1_stream "$c" 200
shas t1_stream 'event: message_stop'
shas t1_stream '"stop_reason":"tool_use"'
shas t1_stream 'input_json_delta'
T1_ID=$(grep -o '"id":"toolu_[^"]*"' "$OUT/t1_stream.sse" | head -1 | cut -d'"' -f4)
[ -n "$T1_ID" ] || fail "t1_stream: no toolu_ id in the stream"
pub1=$(m ds4_continuation_records_published_total)
lane1=$(m 'ds4_route_requests_total{surface="anthropic_messages",lane="continuous"}')
ser1=$(m ds4_requests_serial_total)
[ "${pub1:-0}" -gt "${pub0:-0}" ] || fail "t1_stream: no bank record published (${pub0:-?} -> ${pub1:-?})"
[ "${lane1:-0}" -gt "${lane0:-0}" ] || fail "t1_stream: never entered the cont lane (${lane0:-?} -> ${lane1:-?})"
[ "${ser1:-0}" -eq "${ser0:-0}" ] || fail "t1_stream: serial entries moved (${ser0:-?} -> ${ser1:-?})"
log "t1_stream PASS (streamed tool turn rode cont and published; id=$T1_ID)"

# t2_stream: zero-delay output-only streamed T2 -- the cont
# publication-before-terminal twin AND the 6b consumer in one leg.
T2_BODY='{"model":"m","max_tokens":1200,"temperature":0,"stream":true,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$T1_ID"'","content":"file1.txt\nfile2.txt"}]}],"tools":'"$TOOLS_ANTH"'}'
res0=$(m ds4_continuation_resolved_total)
bank0=$(m 'ds4_route_decisions_total{reason="continuous_bank_continuation"}')
c=$(sse t2_stream /v1/messages "$T2_BODY")
code_is t2_stream "$c" 200
shas t2_stream 'event: message_stop'
res1=$(m ds4_continuation_resolved_total)
bank1=$(m 'ds4_route_decisions_total{reason="continuous_bank_continuation"}')
[ "${res1:-0}" -gt "${res0:-0}" ] || fail "t2_stream: registry never resolved (${res0:-?} -> ${res1:-?})"
[ "${bank1:-0}" -gt "${bank0:-0}" ] || fail "t2_stream: bank-continuation decision never recorded (${bank0:-?} -> ${bank1:-?})"
[ "$(srv_count 'cont bank continuation admit')" -ge 1 ] || fail "t2_stream: bank admit log line missing"
grep -qE '"cache_read_input_tokens":[1-9]' "$OUT/t2_stream.sse" || fail "t2_stream: cache_read_input_tokens not > 0 (bank KV was not kept)"
log "t2_stream PASS (output-only T2 claimed its bank in place; resolved ${res0:-0} -> $res1)"

# t2_repeat409: bank records have NO eager demote sites -- the record is
# still LIVE but the T2 extended the bank, so the frontier equality refuses.
miss0=$(m ds4_continuation_missed_total)
c=$(sse t2_repeat409 /v1/messages "$T2_BODY")
code_is t2_repeat409 "$c" 409
shas t2_repeat409 'continuation state is not available'
miss1=$(m ds4_continuation_missed_total)
[ "${miss1:-0}" -gt "${miss0:-0}" ] || fail "t2_repeat409: missed_total never moved (${miss0:-?} -> ${miss1:-?})"
[ "$(srv_count 'cont bank continuation refused')" -ge 1 ] || fail "t2_repeat409: refusal log line missing"
log "t2_repeat409 PASS (consumed frontier refused with the native 409; missed ${miss0:-0} -> $miss1)"

# budget_cut: the cut lands on the CONT lane now; the ladder must either
# deterministically repair (record published) or finish honest max_tokens
# with NO record -- and the srv.log marker names the branch.
rep0=$(srv_count 'cont chat repaired unterminated tool call'); rep0=${rep0:-0}
cut0=$(srv_count 'cont chat tool call cut by token budget'); cut0=${cut0:-0}
pubb0=$(m ds4_continuation_records_published_total)
c=$(sse budget_cut /v1/messages '{"model":"m","max_tokens":24,"temperature":0,"stream":true,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is budget_cut "$c" 200
rep1=$(srv_count 'cont chat repaired unterminated tool call'); rep1=${rep1:-0}
cut1=$(srv_count 'cont chat tool call cut by token budget'); cut1=${cut1:-0}
pubb1=$(m ds4_continuation_records_published_total)
if [ "$rep1" -gt "$rep0" ]; then
  [ "${pubb1:-0}" -gt "${pubb0:-0}" ] || fail "budget_cut: repaired but no record published"
  shas budget_cut '"stop_reason":"tool_use"'
  log "budget_cut PASS (deterministic repair completed the cut call; record published)"
elif [ "$cut1" -gt "$cut0" ]; then
  [ "${pubb1:-0}" -eq "${pubb0:-0}" ] || fail "budget_cut: length-cut turn published (${pubb0:-?} -> ${pubb1:-?})"
  shas budget_cut '"stop_reason":"max_tokens"'
  log "budget_cut PASS (honest max_tokens cut; no strand, no record)"
else
  # The tiny budget may cut before the tool block even starts: honest
  # max_tokens with no marker is a plain length row, still no record.
  shas budget_cut '"stop_reason":"max_tokens"'
  [ "${pubb1:-0}" -eq "${pubb0:-0}" ] || fail "budget_cut: cut-before-toolstart turn published"
  log "budget_cut PASS (cut before the tool block; no record)"
fi

# think_sig: thinking + tools + streaming in one cont row; the signature
# ships in the stream and the T2 still resolves.
c=$(sse think_t1 /v1/messages '{"model":"m","max_tokens":1600,"temperature":0,"stream":true,"thinking":{"type":"enabled"},"messages":[{"role":"user","content":"Use the list_files tool to list the files in /var. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is think_t1 "$c" 200
shas think_t1 '"stop_reason":"tool_use"'
shas think_t1 'thinking_delta'
shas think_t1 'signature_delta'
TH_ID=$(grep -o '"id":"toolu_[^"]*"' "$OUT/think_t1.sse" | head -1 | cut -d'"' -f4)
[ -n "$TH_ID" ] || fail "think_t1: no toolu_ id"
resT0=$(m ds4_continuation_resolved_total)
c=$(sse think_t2 /v1/messages '{"model":"m","max_tokens":1600,"temperature":0,"stream":true,"thinking":{"type":"enabled"},"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$TH_ID"'","content":"log.txt\ncache"}]}],"tools":'"$TOOLS_ANTH"'}')
code_is think_t2 "$c" 200
resT1=$(m ds4_continuation_resolved_total)
[ "${resT1:-0}" -gt "${resT0:-0}" ] || fail "think_t2: thinking T2 did not resolve (${resT0:-?} -> ${resT1:-?})"
log "think_sig PASS (thinking + signature streamed on cont; T2 resolved)"

# buffered_serial: engagement-NEGATIVE.  The corrective contract holds
# buffered tool turns serial even with every switch at its default.
dec0=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
serb0=$(m ds4_requests_serial_total)
admit0=$(srv_count 'cont bank continuation admit'); admit0=${admit0:-0}
c=$(post buffered_serial /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /etc. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is buffered_serial "$c" 200
has buffered_serial '"stop_reason":"tool_use"'
dec1=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
serb1=$(m ds4_requests_serial_total)
admit1=$(srv_count 'cont bank continuation admit'); admit1=${admit1:-0}
[ "${dec1:-0}" -gt "${dec0:-0}" ] || fail "buffered_serial: publish-hold decision never recorded (${dec0:-?} -> ${dec1:-?})"
[ "${serb1:-0}" -gt "${serb0:-0}" ] || fail "buffered_serial: serial entries unmoved (${serb0:-?} -> ${serb1:-?})"
[ "$admit1" -eq "$admit0" ] || fail "buffered_serial: a bank admit happened for a buffered turn"
log "buffered_serial PASS (buffered tool turn kept the serial corrective contract)"

# ==================== boot 2: responses + cross-surface =====================
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

pubr0=$(m ds4_continuation_records_published_total)
laner0=$(m 'ds4_route_requests_total{surface="openai_responses",lane="continuous"}')
c=$(sse rt1_stream /v1/responses '{"model":"m","max_output_tokens":1200,"temperature":0,"stream":true,"input":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_RESP"'}')
code_is rt1_stream "$c" 200
shas rt1_stream 'function_call_arguments'
RT1_ID=$(grep -o '"call_id":"[^"]*"' "$OUT/rt1_stream.sse" | head -1 | cut -d'"' -f4)
[ -n "$RT1_ID" ] || fail "rt1_stream: no call_id in the stream"
pubr1=$(m ds4_continuation_records_published_total)
laner1=$(m 'ds4_route_requests_total{surface="openai_responses",lane="continuous"}')
[ "${pubr1:-0}" -gt "${pubr0:-0}" ] || fail "rt1_stream: no bank record published (${pubr0:-?} -> ${pubr1:-?})"
[ "${laner1:-0}" -gt "${laner0:-0}" ] || fail "rt1_stream: never entered the cont lane (${laner0:-?} -> ${laner1:-?})"
log "rt1_stream PASS (streamed function-call turn rode cont; call_id=$RT1_ID)"

RT2_BODY='{"model":"m","max_output_tokens":1200,"temperature":0,"stream":true,"input":[{"type":"function_call_output","call_id":"'"$RT1_ID"'","output":"file1.txt\nfile2.txt"}],"tools":'"$TOOLS_RESP"'}'
resr0=$(m ds4_continuation_resolved_total)
c=$(sse rt2_stream /v1/responses "$RT2_BODY")
code_is rt2_stream "$c" 200
resr1=$(m ds4_continuation_resolved_total)
[ "${resr1:-0}" -gt "${resr0:-0}" ] || fail "rt2_stream: registry never resolved (${resr0:-?} -> ${resr1:-?})"
[ "$(srv_count 'cont bank continuation admit')" -ge 1 ] || fail "rt2_stream: bank admit log line missing"
log "rt2_stream PASS (output-only Responses T2 claimed its bank)"

missr0=$(m ds4_continuation_missed_total)
c=$(sse rt2_repeat409 /v1/responses "$RT2_BODY")
code_is rt2_repeat409 "$c" 409
missr1=$(m ds4_continuation_missed_total)
[ "${missr1:-0}" -gt "${missr0:-0}" ] || fail "rt2_repeat409: missed_total never moved (${missr0:-?} -> ${missr1:-?})"
log "rt2_repeat409 PASS (consumed Responses frontier refused natively)"

# multi_calls: exact id-SET equality on however many calls the turn made.
c=$(sse multi_t1 /v1/messages '{"model":"m","max_tokens":1600,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Call the list_files tool twice in this single turn: once for /tmp and once for /home. Emit both tool calls together."}],"tools":'"$TOOLS_ANTH"'}')
code_is multi_t1 "$c" 200
shas multi_t1 '"stop_reason":"tool_use"'
MULTI_IDS=$(grep -o '"id":"toolu_[^"]*"' "$OUT/multi_t1.sse" | cut -d'"' -f4 | sort -u)
NIDS=$(echo "$MULTI_IDS" | grep -c toolu_)
[ "$NIDS" -ge 1 ] || fail "multi_t1: no toolu_ ids"
python3 - "$NIDS" $MULTI_IDS > "$OUT/multi_t2_req.json" <<'PY' || fail "multi_t2: request build failed"
import json, sys
ids = sys.argv[2:]
content = [{"type": "tool_result", "tool_use_id": i, "content": "ok: file%d.txt" % n}
           for n, i in enumerate(ids)]
print(json.dumps({"model": "m", "max_tokens": 1200, "temperature": 0, "stream": True,
                  "messages": [{"role": "user", "content": content}],
                  "tools": [{"name": "list_files", "description": "List the files in a directory",
                             "input_schema": {"type": "object", "properties": {"path": {"type": "string"}},
                                              "required": ["path"]}}]}))
PY
resm0=$(m ds4_continuation_resolved_total)
c=$(curl -s -m 240 --no-buffer -o "$OUT/multi_t2.sse" -w '%{http_code}' "$BASE/v1/messages" \
     -H 'Content-Type: application/json' -d @"$OUT/multi_t2_req.json")
code_is multi_t2 "$c" 200
resm1=$(m ds4_continuation_resolved_total)
[ "${resm1:-0}" -gt "${resm0:-0}" ] || fail "multi_t2: id-set claim did not resolve (${resm0:-?} -> ${resm1:-?})"
log "multi_calls PASS ($NIDS call(s) in the turn; T2 with the full id set resolved)"

# mixed_rows: a tool row and a plain row share one continuous group.
c1f="$OUT/mixed_tool.sse"; c2f="$OUT/mixed_plain.json"
curl -s -m 240 --no-buffer -o "$c1f" -w '%{http_code}' "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":1200,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /opt. Call the tool."}],"tools":'"$TOOLS_ANTH"'}' > "$OUT/mixed_tool.code" &
MIX_PID=$!
sleep 0.3
c=$(curl -s -m 240 -o "$c2f" -w '%{http_code}' "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":48,"temperature":0,"messages":[{"role":"user","content":"Name three colors, comma separated."}]}')
wait "$MIX_PID" 2>/dev/null
[ "$c" = "200" ] || fail "mixed_rows: plain row HTTP $c"
[ "$(cat "$OUT/mixed_tool.code")" = "200" ] || fail "mixed_rows: tool row HTTP $(cat "$OUT/mixed_tool.code")"
grep -q '"stop_reason":"tool_use"' "$c1f" || fail "mixed_rows: tool row did not finish tool_use"
grep -q '"content"' "$c2f" || fail "mixed_rows: plain row empty"
log "mixed_rows PASS (tool + plain rows served concurrently on cont)"

# crossproto_bank: the anth bank id must never validate as a Responses
# continuation (protocol-scoped registry index).
MIX_ID=$(grep -o '"id":"toolu_[^"]*"' "$c1f" | head -1 | cut -d'"' -f4)
[ -n "$MIX_ID" ] || fail "crossproto_bank: no toolu_ id from the mixed tool row"
c=$(post crossproto_bank /v1/responses '{"model":"m","max_output_tokens":600,"temperature":0,"input":[{"type":"function_call_output","call_id":"'"$MIX_ID"'","output":"nope"}],"tools":'"$TOOLS_RESP"'}')
code_is crossproto_bank "$c" 400
has crossproto_bank 'retry by replaying the full input history'
log "crossproto_bank PASS (bank id refused across protocols at parse)"

# resp_buffered_serial: engagement-NEGATIVE twin on Responses.
decr0=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
c=$(post resp_buffered_serial /v1/responses '{"model":"m","max_output_tokens":1200,"temperature":0,"input":[{"role":"user","content":"Use the list_files tool to list the files in /etc. Call the tool."}],"tools":'"$TOOLS_RESP"'}')
code_is resp_buffered_serial "$c" 200
has resp_buffered_serial '"type":"function_call"'
decr1=$(m 'ds4_route_decisions_total{reason="need_continuation_publish"}')
[ "${decr1:-0}" -gt "${decr0:-0}" ] || fail "resp_buffered_serial: publish-hold decision never recorded"
log "resp_buffered_serial PASS (buffered Responses tool turn stayed serial)"

# ==================== boot 3: 6c retention (2 banks, grace 20) ==============
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_SERVER_COALESCE_MAX=2 DS4_CONT_GRACE_S=20" boot
ssh "$R" "grep -q 'persistent batch ctx ready (max_seq=2' $RWORK/srv.log" || \
  fail "boot3: max_seq=2 not honored ($(ssh "$R" "grep 'persistent batch ctx ready' $RWORK/srv.log" 2>/dev/null))"

# prot_t1: bank A takes the tool turn; its record is LIVE and in grace.
c=$(sse prot_t1 /v1/messages '{"model":"m","max_tokens":1200,"temperature":0,"stream":true,"messages":[{"role":"user","content":"Use the list_files tool to list the files in /tmp. Call the tool."}],"tools":'"$TOOLS_ANTH"'}')
code_is prot_t1 "$c" 200
shas prot_t1 '"stop_reason":"tool_use"'
PROT_ID=$(grep -o '"id":"toolu_[^"]*"' "$OUT/prot_t1.sse" | head -1 | cut -d'"' -f4)
[ -n "$PROT_ID" ] || fail "prot_t1: no toolu_ id"
log "prot_t1 PASS (bank record live, grace running)"

# prot_shed: the occupier holds bank B; a cold probe finds bank A protected
# and bank B occupied -> no victim -> 503 continuation_hold.
curl -s -m 240 --no-buffer -o "$OUT/occupier.sse" "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":700,"temperature":0,"stream":true,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Count from 1 to 200, comma separated."}]}' &
OCC_PID=$!
sleep 3
shed0=$(m 'ds4_requests_shed_total{reason="continuation_hold"}')
c=$(curl -s -m 60 -o "$OUT/prot_shed.json" -w '%{http_code}' -D "$OUT/prot_shed.hdr" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}')
[ "$c" = "503" ] || fail "prot_shed: HTTP $c, want 503 ($(head -c 200 "$OUT/prot_shed.json"))"
grep -qi 'retry-after' "$OUT/prot_shed.hdr" || fail "prot_shed: no Retry-After header"
shed1=$(m 'ds4_requests_shed_total{reason="continuation_hold"}')
[ "${shed1:-0}" -gt "${shed0:-0}" ] || fail "prot_shed: continuation_hold shed never counted (${shed0:-?} -> ${shed1:-?})"
[ "$(srv_count 'cont admission hold shed')" -ge 1 ] || fail "prot_shed: hold-shed log line missing"
log "prot_shed PASS (all victims protected/occupied -> 503 continuation_hold)"

# prot_t2: the protection existed FOR this -- the output-only T2 claims
# bank A in place while the occupier still runs.
resp0=$(m ds4_continuation_resolved_total)
c=$(sse prot_t2 /v1/messages '{"model":"m","max_tokens":600,"temperature":0,"stream":true,"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"'"$PROT_ID"'","content":"file1.txt\nfile2.txt"}]}],"tools":'"$TOOLS_ANTH"'}')
code_is prot_t2 "$c" 200
resp1=$(m ds4_continuation_resolved_total)
[ "${resp1:-0}" -gt "${resp0:-0}" ] || fail "prot_t2: protected frontier did not resolve (${resp0:-?} -> ${resp1:-?})"
log "prot_t2 PASS (T2 claimed the protected bank while the occupier ran)"

wait "$OCC_PID" 2>/dev/null
grep -q 'event: message_stop' "$OUT/occupier.sse" || fail "occupier stream never finished cleanly"

# postgrace_ok: protection is grace-bounded -- wait out the window, then the
# SAME cold probe shape admits (a victim exists again).
sleep 22
c=$(post postgrace_ok /v1/chat/completions '{"model":"m","max_tokens":32,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}')
code_is postgrace_ok "$c" 200
has postgrace_ok '"content"'
log "postgrace_ok PASS (grace lapsed; cold admission works again)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT"
