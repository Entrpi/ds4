#!/bin/bash
# serving/cont_streaming.sh — v0.5.6 API Inc 4: streaming on the continuous lane.
#
# Inc 4a legs (boot A, zero-config + DS4_MEM_FLOOR_GB=2): transport starts at
# successful ADMISSION (plan §4.7), not at first token, and the admission
# prefill carries a heartbeat between chunks.
#
#   stream_small     tiny-prompt chat stream: valid end-to-end after the
#                    transport-timing change (also absorbs first-request
#                    warmup so the timing legs measure steady state).
#   early_transport  ~12k-token chat stream on a 16k boot: headers must
#                    arrive within seconds of admission (TTFB << prefill),
#                    with >=1 ": prefill" keepalive comment during prefill,
#                    then a valid stream.  Old behavior = first byte after
#                    the full prefill (~12s+), so TTFB<5s is discriminating.
#   completion_shape completion-kind stream through the early-transport
#                    path keeps the serial oracle's plain text_completion
#                    shape (no chat objects, no role preamble).
#   abort_prefill    client killed mid-prefill (curl -m): the alive() poll
#                    abandons the pending admission, settles CANCELED, the
#                    server stays healthy and the next request serves.
#
# Engagement proof per leg: the surface's cont lane cell moves EXACTLY as
# expected and ds4_requests_serial_total is unmoved (LANE-ENTRY trap law).
# Runs FROM the Mac over the ssh tunnel.  End state: server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
PORT=${PORT:-8000}
TUNNEL_PORT=${TUNNEL_PORT:-18000}
CTX=${CTX:-16384}
RWORK=/tmp/cont_streaming_gate
OUT=${OUT:-/tmp/cont_streaming_gate_$$}
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

# number extraction BEFORE head (METRICS-HELPER TRAP: the fixed-string grep
# also matches the "# TYPE" comment line for unlabeled metrics).
m(){ curl -s -m 10 "$BASE/metrics" | grep -F "$1" | grep -oE '[0-9]+$' | head -1; }
lane(){ m "surface=\"$1\",lane=\"$2\""; }

# Timestamped SSE reader: raw stream -> $1, per-chunk timing + TTFB -> $2.
sse_timed(){  # $1=out.sse $2=out.t $3=path $4=body $5=overall-timeout
  python3 - "$TUNNEL_PORT" "$3" "$4" "$5" > "$OUT/$1" 2> "$OUT/$2" <<'PYEOF'
import socket, sys, time
port, path, body, tmo = int(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4])
s = socket.create_connection(("127.0.0.1", port), timeout=tmo)
req = (f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
       f"Content-Length: {len(body.encode())}\r\nConnection: close\r\n\r\n{body}")
t0 = time.time()
s.sendall(req.encode())
s.settimeout(tmo)
first = -1.0
out = []
while True:
    try:
        chunk = s.recv(65536)
    except socket.timeout:
        sys.stderr.write("TIMEOUT\n"); break
    if not chunk:
        break
    if first < 0:
        first = time.time() - t0
    sys.stderr.write(f"CHUNK {time.time()-t0:.3f} {len(chunk)}\n")
    out.append(chunk)
sys.stderr.write(f"TTFB {first:.3f}\n")
sys.stdout.buffer.write(b"".join(out))
PYEOF
}

sse_has(){ grep -q "$2" "$OUT/$1" || fail "$1: missing [$2]"; }
sse_lacks(){ grep -q "$2" "$OUT/$1" && fail "$1: found forbidden [$2]"; return 0; }
delta_is(){ # $1=name $2=before $3=after $4=want
  [ $(( $3 - $2 )) -eq "$4" ] || fail "$1: delta $(( $3 - $2 )), want $4"
}

# Deep prompt: this sentence tokenizes at ~19 tok/repeat (measured 17110 for
# 900 on the 0731 tokenizer) -- 800 repeats ≈ 15.2k tokens: inside the 16k
# bank with template margin, prefill comfortably >> the 5s heartbeat cadence.
DEEP_PROMPT=$(python3 -c "print(('the quick brown fox jumps over the lazy dog near the riverbank while autumn leaves drift slowly downward ' * 800).strip())")

# ==================== boot A: Inc 4a transport lifecycle ====================
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

# ---- stream_small: correctness after the transport-timing change ----------
oc0=$(lane openai_chat continuous); sl0=$(m ds4_requests_serial_total)
sse_timed small.sse small.t /v1/chat/completions \
  '{"stream":true,"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one planet."}]}' 180
sse_has small.sse 'data: \[DONE\]'
sse_has small.sse '"delta":{"role":"assistant"}'
sse_has small.sse '"finish_reason"'
oc1=$(lane openai_chat continuous); sl1=$(m ds4_requests_serial_total)
delta_is "stream_small cont lane" "$oc0" "$oc1" 1
delta_is "stream_small serial"    "$sl0" "$sl1" 0
log "stream_small PASS"

# ---- early_transport: TTFB << prefill + >=1 keepalive ----------------------
oc0=$oc1; sl0=$sl1
sse_timed deep.sse deep.t /v1/chat/completions \
  "{\"stream\":true,\"max_tokens\":48,\"temperature\":0,\"thinking\":false,\"stream_options\":{\"include_usage\":true},\"messages\":[{\"role\":\"user\",\"content\":\"Summarize in one sentence: $DEEP_PROMPT\"}]}" 300
ttfb=$(grep '^TTFB' "$OUT/deep.t" | awk '{print $2}')
[ -n "$ttfb" ] || fail "early_transport: no TTFB recorded"
awk -v t="$ttfb" 'BEGIN{exit !(t >= 0 && t < 5.0)}' || \
  fail "early_transport: TTFB ${ttfb}s, want <5s (old behavior = full prefill)"
sse_has deep.sse ': prefill'
sse_has deep.sse 'data: \[DONE\]'
sse_has deep.sse '"finish_reason"'
# prefill actually deep, two ways: the usage chunk's prompt_tokens proves the
# prompt size (stream_options.include_usage), and the stream SPAN proves the
# transport was open long before the first token could exist (48-token budget
# finishes in well under a second once decode starts).
pt=$(grep -o '"prompt_tokens":[0-9]*' "$OUT/deep.sse" | head -1 | grep -oE '[0-9]+$')
[ -n "$pt" ] && [ "$pt" -ge 8000 ] || fail "early_transport: prompt_tokens ${pt:-none}, want >=8000"
span=$(grep '^CHUNK' "$OUT/deep.t" | tail -1 | awk '{print $2}')
awk -v s="${span:-0}" -v t="$ttfb" 'BEGIN{exit !(s - t > 5.0)}' || \
  fail "early_transport: stream span ${span:-0}s - TTFB ${ttfb}s <= 5s (prefill not deep?)"
oc1=$(lane openai_chat continuous); sl1=$(m ds4_requests_serial_total)
delta_is "early_transport cont lane" "$oc0" "$oc1" 1
delta_is "early_transport serial"    "$sl0" "$sl1" 0
log "early_transport PASS (TTFB=${ttfb}s prompt_tokens=$pt)"

# ---- completion_shape: plain branch untouched by early transport ----------
cc0=$(lane openai_completion continuous); sl0=$sl1
sse_timed comp.sse comp.t /v1/completions \
  '{"stream":true,"prompt":"The capital of France is","max_tokens":32,"temperature":0}' 180
sse_has comp.sse '"object":"text_completion"'
sse_has comp.sse 'data: \[DONE\]'
sse_lacks comp.sse 'chat.completion.chunk'
sse_lacks comp.sse '"delta"'
cc1=$(lane openai_completion continuous); sl1=$(m ds4_requests_serial_total)
delta_is "completion_shape cont lane" "$cc0" "$cc1" 1
delta_is "completion_shape serial"    "$sl0" "$sl1" 0
log "completion_shape PASS"

# ---- abort_prefill: kill mid-prefill -> CANCELED, healthy, bank unwound ----
ca0=$(m 'ds4_requests_total{outcome="canceled"}')
curl -s -N -m 3 -o "$OUT/abort.sse" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d "{\"stream\":true,\"max_tokens\":48,\"temperature\":0,\"thinking\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Summarize: $DEEP_PROMPT\"}]}" \
     >/dev/null 2>&1
grep -q 'HTTP/1.1 200\|text/event-stream' "$OUT/abort.sse" 2>/dev/null || true  # raw capture optional
n=0
while :; do
  ca1=$(m 'ds4_requests_total{outcome="canceled"}')
  [ -n "$ca1" ] && [ "$ca1" -gt "$ca0" ] && break
  n=$((n+1)); [ $n -ge 30 ] && fail "abort_prefill: canceled never moved (${ca0}->${ca1:-?})"
  sleep 2
done
hc=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$hc" = "200" ] || fail "abort_prefill: health probe HTTP $hc"
curl -s -m 180 -o "$OUT/recover.json" "$BASE/v1/chat/completions" \
     -H 'Content-Type: application/json' \
     -d '{"max_tokens":400,"temperature":0,"messages":[{"role":"user","content":"Name one color."}]}'
grep -q '"finish_reason"' "$OUT/recover.json" || fail "abort_prefill: recovery request failed"
log "abort_prefill PASS (canceled ${ca0}->${ca1})"

# ---- boot A ledger ----------------------------------------------------------
st=$(m ds4_requests_started_total); cp=$(m 'ds4_requests_total{outcome="completed"}')
ca=$(m 'ds4_requests_total{outcome="canceled"}')
[ "$st" -eq $(( cp + ca )) ] || fail "ledger: started=$st != completed=$cp + canceled=$ca"
log "ledger PASS (started=$st completed=$cp canceled=$ca)"

# =================== boot B: Inc 4b Anthropic streaming =====================
# The promoted /v1/messages stream rides cont: native event automata, the
# thinking/text block split, stop-sequence honesty, admission-time transport
# with the native ping heartbeat, and a mid-decode abort settling CANCELED.
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

first(){ grep -n "$2" "$OUT/$1" | head -1 | cut -d: -f1; }
order(){ # $1=file $2..=patterns in required order
  local f=$1 prev=0 prevpat="(start)" ln; shift
  for pat in "$@"; do
    ln=$(first "$f" "$pat")
    [ -n "$ln" ] || fail "$f: missing event [$pat]"
    [ "$ln" -gt "$prev" ] || fail "$f: [$pat] at line $ln not after [$prevpat] at $prev"
    prev=$ln; prevpat=$pat
  done
}
anth_text(){ # reassemble text deltas
  python3 - "$OUT/$1" <<'PYEOF'
import sys, json
text = ""
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("data: "):
        try: e = json.loads(line[6:])
        except Exception: continue
        if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
            text += e["delta"]["text"]
sys.stdout.write(text)
PYEOF
}

# ---- anth_stream_basic: think-off text automata ----------------------------
ac0=$(lane anthropic_messages continuous); sl0=$(m ds4_requests_serial_total)
sse_timed anthb.sse anthb.t /v1/messages \
  '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Name one planet."}]}' 180
order anthb.sse '"type":"message_start"' '"type":"content_block_start"' \
      '"type":"content_block_delta"' '"type":"content_block_stop"' \
      '"type":"message_delta"' '"type":"message_stop"'
sse_has anthb.sse 'event: message_start'
sse_has anthb.sse '"stop_reason":"end_turn"'
sse_lacks anthb.sse 'chat.completion.chunk'
[ -n "$(anth_text anthb.sse)" ] || fail "anth_stream_basic: empty reassembled text"
ac1=$(lane anthropic_messages continuous); sl1=$(m ds4_requests_serial_total)
delta_is "anth_stream_basic cont lane" "$ac0" "$ac1" 1
delta_is "anth_stream_basic serial"    "$sl0" "$sl1" 0
log "anth_stream_basic PASS"

# ---- anth_stream_thinking: thinking block precedes text --------------------
# Reasoning is the model's choice at temp 0 (standing law): one retry on an
# unreasoned answer, then fail.  Lane deltas count the attempts actually made.
ac0=$ac1; sl0=$sl1
attempts=0
for try in 1 2; do
  attempts=$try
  sse_timed antht.sse antht.t /v1/messages \
    '{"model":"m","stream":true,"max_tokens":600,"temperature":0,"messages":[{"role":"user","content":"If a train travels 60 miles in 1.5 hours, what is its average speed in mph?"}]}' 240
  grep -q '"type":"thinking_delta"' "$OUT/antht.sse" && break
  [ $try -eq 2 ] && fail "anth_stream_thinking: unreasoned on both attempts"
  log "anth_stream_thinking: attempt $try unreasoned, retrying"
done
order antht.sse '"type":"thinking_delta"' '"type":"signature_delta"' '"type":"text_delta"'
order antht.sse '"type":"message_start"' '"type":"content_block_start"' \
      '"type":"message_delta"' '"type":"message_stop"'
ac1=$(lane anthropic_messages continuous); sl1=$(m ds4_requests_serial_total)
delta_is "anth_stream_thinking cont lane" "$ac0" "$ac1" "$attempts"
delta_is "anth_stream_thinking serial"    "$sl0" "$sl1" 0
log "anth_stream_thinking PASS (attempts=$attempts)"

# ---- anth_stream_stops: stop-sequence honesty on the stream ----------------
ac0=$ac1; sl0=$sl1
sse_timed anths.sse anths.t /v1/messages \
  '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"thinking":{"type":"disabled"},"stop_sequences":[","],"messages":[{"role":"user","content":"Count upward from one, separating numbers with commas: one, two, three..."}]}' 180
sse_has anths.sse '"stop_reason":"stop_sequence"'
sse_has anths.sse '"stop_sequence":","'
txt=$(anth_text anths.sse)
case "$txt" in *,*) fail "anth_stream_stops: stop string leaked into deltas [$txt]";; esac
ac1=$(lane anthropic_messages continuous); sl1=$(m ds4_requests_serial_total)
delta_is "anth_stream_stops cont lane" "$ac0" "$ac1" 1
delta_is "anth_stream_stops serial"    "$sl0" "$sl1" 0
log "anth_stream_stops PASS"

# ---- anth_stream_deep: admission transport + native ping heartbeat ---------
ac0=$ac1; sl0=$sl1
sse_timed anthd.sse anthd.t /v1/messages \
  "{\"model\":\"m\",\"stream\":true,\"max_tokens\":48,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"},\"messages\":[{\"role\":\"user\",\"content\":\"Summarize in one sentence: $DEEP_PROMPT\"}]}" 300
ttfb=$(grep '^TTFB' "$OUT/anthd.t" | awk '{print $2}')
awk -v t="${ttfb:--1}" 'BEGIN{exit !(t >= 0 && t < 5.0)}' || \
  fail "anth_stream_deep: TTFB ${ttfb:-none}s, want <5s"
sse_has anthd.sse 'event: ping'
sse_has anthd.sse 'data: {"type": "ping"}'
sse_lacks anthd.sse ': prefill'
sse_has anthd.sse '"type":"message_stop"'
# Anthropic frame: prompt = input + cache_read + cache_creation (a cold
# admit charges the WHOLE prompt to cache_creation and input_tokens is 0).
pt=$(python3 - "$OUT/anthd.sse" <<'PYEOF'
import sys, json
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("data: ") and '"message_start"' in line:
        u = json.loads(line[6:])["message"]["usage"]
        print(u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
              + u.get("cache_creation_input_tokens", 0))
        break
PYEOF
)
[ -n "$pt" ] && [ "$pt" -ge 8000 ] || fail "anth_stream_deep: prompt-frame sum ${pt:-none}, want >=8000"
ac1=$(lane anthropic_messages continuous); sl1=$(m ds4_requests_serial_total)
delta_is "anth_stream_deep cont lane" "$ac0" "$ac1" 1
delta_is "anth_stream_deep serial"    "$sl0" "$sl1" 0
log "anth_stream_deep PASS (TTFB=${ttfb}s input_tokens=$pt)"

# ---- anth_stream_abort: mid-decode disconnect settles CANCELED -------------
ca0=$(m 'ds4_requests_total{outcome="canceled"}')
curl -s -N -m 4 -o /dev/null "$BASE/v1/messages" \
     -H 'Content-Type: application/json' \
     -d '{"model":"m","stream":true,"max_tokens":1500,"temperature":0,"messages":[{"role":"user","content":"Write a long essay about rivers."}]}' \
     2>/dev/null
n=0
while :; do
  ca1=$(m 'ds4_requests_total{outcome="canceled"}')
  [ -n "$ca1" ] && [ "$ca1" -gt "$ca0" ] && break
  n=$((n+1)); [ $n -ge 30 ] && fail "anth_stream_abort: canceled never moved"
  sleep 2
done
hc=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$hc" = "200" ] || fail "anth_stream_abort: health probe HTTP $hc"
log "anth_stream_abort PASS (canceled ${ca0}->${ca1})"

# ============ boot C: Inc 4b kill switch restores legacy serial =============
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_SERVER_CONT_ANTHROPIC=0" boot
ac0=$(lane anthropic_messages continuous); sl0=$(m ds4_requests_serial_total)
sse_timed anthk.sse anthk.t /v1/messages \
  '{"model":"m","stream":true,"max_tokens":400,"temperature":0,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"Name one metal."}]}' 180
order anthk.sse '"type":"message_start"' '"type":"content_block_delta"' \
      '"type":"message_stop"'
ac1=$(lane anthropic_messages continuous); sl1=$(m ds4_requests_serial_total)
delta_is "anth_killswitch cont lane" "$ac0" "$ac1" 0
delta_is "anth_killswitch serial"    "$sl0" "$sl1" 1
log "anth_killswitch PASS (legacy serial, native bytes)"

# =================== boot D: Inc 4c Responses streaming =====================
BOOT_ENV="DS4_MEM_FLOOR_GB=2" boot

resp_seq_check(){ # sequence_number strictly 0,1,2,... across every data: event
  python3 - "$OUT/$1" <<'PYEOF'
import sys, json
want = 0
for line in open(sys.argv[1], errors="replace"):
    if not line.startswith("data: "): continue
    try: e = json.loads(line[6:])
    except Exception: continue
    sn = e.get("sequence_number")
    if sn is None: sys.exit(f"event {e.get('type')} missing sequence_number")
    if sn != want: sys.exit(f"sequence_number {sn}, want {want}")
    want += 1
PYEOF
}
resp_text(){ # reassemble output_text deltas
  python3 - "$OUT/$1" <<'PYEOF'
import sys, json
text = ""
for line in open(sys.argv[1], errors="replace"):
    if line.startswith("data: "):
        try: e = json.loads(line[6:])
        except Exception: continue
        if e.get("type") == "response.output_text.delta":
            text += e.get("delta", "")
sys.stdout.write(text)
PYEOF
}

# ---- resp_stream_basic: default effort -> reasoning then text automata -----
rc0=$(lane openai_responses continuous); sl0=$(m ds4_requests_serial_total)
sse_timed respb.sse respb.t /v1/responses \
  '{"stream":true,"max_output_tokens":600,"temperature":0,"input":"If a train travels 60 miles in 1.5 hours, what is its average speed in mph?"}' 240
order respb.sse '"type":"response.created"' '"type":"response.output_item.added"' \
      '"type":"response.output_text.delta"' '"type":"response.output_text.done"' \
      '"type":"response.output_item.done"' '"type":"response.completed"'
resp_seq_check respb.sse || fail "resp_stream_basic: $(resp_seq_check respb.sse 2>&1)"
[ -n "$(resp_text respb.sse)" ] || fail "resp_stream_basic: empty reassembled text"
rc1=$(lane openai_responses continuous); sl1=$(m ds4_requests_serial_total)
delta_is "resp_stream_basic cont lane" "$rc0" "$rc1" 1
delta_is "resp_stream_basic serial"    "$sl0" "$sl1" 0
log "resp_stream_basic PASS"

# ---- resp_stream_reasoning: thinking deltas precede text --------------------
# Reasoning deltas require the PROTOCOL opt-in (reasoning.summary) -- without
# it the machine walks past <think>...</think> and suppresses emission on
# BOTH lanes (measured: a no-opt-in stream carries the answer only, no leak).
rc0=$rc1; sl0=$sl1
attempts=0
for try in 1 2; do
  attempts=$try
  sse_timed respr.sse respr.t /v1/responses \
    '{"stream":true,"max_output_tokens":600,"temperature":0,"reasoning":{"summary":"auto"},"input":"A rectangle is twice as long as it is wide and its perimeter is 36; what is its area?"}' 240
  grep -q '"type":"response.reasoning_summary_text.delta"' "$OUT/respr.sse" && break
  [ $try -eq 2 ] && fail "resp_stream_reasoning: unreasoned on both attempts"
  log "resp_stream_reasoning: attempt $try unreasoned, retrying"
done
# The ds4 oracle's shape (serial machine = protocol oracle, Inc 0a law):
# reasoning deltas stream first, text deltas follow at </think>, and the
# done-events cluster fires at FINALIZE (responses_sse_finish_live) --
# summary_text.done comes AFTER the text deltas, not at the think boundary.
order respr.sse '"type":"response.reasoning_summary_part.added"' \
      '"type":"response.reasoning_summary_text.delta"' \
      '"type":"response.output_text.delta"' \
      '"type":"response.reasoning_summary_text.done"' \
      '"type":"response.reasoning_summary_part.done"' \
      '"type":"response.output_text.done"' '"type":"response.completed"'
resp_seq_check respr.sse || fail "resp_stream_reasoning: $(resp_seq_check respr.sse 2>&1)"
rc1=$(lane openai_responses continuous); sl1=$(m ds4_requests_serial_total)
delta_is "resp_stream_reasoning cont lane" "$rc0" "$rc1" "$attempts"
delta_is "resp_stream_reasoning serial"    "$sl0" "$sl1" 0
log "resp_stream_reasoning PASS (attempts=$attempts)"

# ---- resp_stream_zero: 3c wire honesty on the STREAM ------------------------
rc0=$rc1; sl0=$sl1
sse_timed respz.sse respz.t /v1/responses \
  '{"stream":true,"max_output_tokens":0,"temperature":0,"input":"Say hi."}' 120
sse_has respz.sse '"type":"response.created"'
sse_has respz.sse '"status":"incomplete"'
sse_has respz.sse '"reason":"max_output_tokens"'
sse_has respz.sse '"output_tokens":0'
sse_lacks respz.sse '"type":"response.output_text.delta"'
sse_lacks respz.sse '"type":"response.reasoning_summary_text.delta"'
resp_seq_check respz.sse || fail "resp_stream_zero: $(resp_seq_check respz.sse 2>&1)"
rc1=$(lane openai_responses continuous); sl1=$(m ds4_requests_serial_total)
delta_is "resp_stream_zero cont lane" "$rc0" "$rc1" 1
delta_is "resp_stream_zero serial"    "$sl0" "$sl1" 0
log "resp_stream_zero PASS"

# ---- resp_stream_deep: admission transport + comment keepalive --------------
rc0=$rc1; sl0=$sl1
sse_timed respd.sse respd.t /v1/responses \
  "{\"stream\":true,\"max_output_tokens\":48,\"temperature\":0,\"reasoning\":{\"effort\":\"none\"},\"input\":\"Summarize in one sentence: $DEEP_PROMPT\"}" 300
ttfb=$(grep '^TTFB' "$OUT/respd.t" | awk '{print $2}')
awk -v t="${ttfb:--1}" 'BEGIN{exit !(t >= 0 && t < 5.0)}' || \
  fail "resp_stream_deep: TTFB ${ttfb:-none}s, want <5s"
sse_has respd.sse ': prefill'
sse_lacks respd.sse 'event: ping'
sse_has respd.sse '"type":"response.created"'
grep -qE '"type":"response.(completed|incomplete)"' "$OUT/respd.sse" || \
  fail "resp_stream_deep: no terminal response event"
rc1=$(lane openai_responses continuous); sl1=$(m ds4_requests_serial_total)
delta_is "resp_stream_deep cont lane" "$rc0" "$rc1" 1
delta_is "resp_stream_deep serial"    "$sl0" "$sl1" 0
log "resp_stream_deep PASS (TTFB=${ttfb}s)"

# ---- resp_stream_abort: mid-decode disconnect settles CANCELED --------------
ca0=$(m 'ds4_requests_total{outcome="canceled"}')
curl -s -N -m 4 -o /dev/null "$BASE/v1/responses" \
     -H 'Content-Type: application/json' \
     -d '{"stream":true,"max_output_tokens":1500,"temperature":0,"input":"Write a long essay about rivers."}' \
     2>/dev/null
n=0
while :; do
  ca1=$(m 'ds4_requests_total{outcome="canceled"}')
  [ -n "$ca1" ] && [ "$ca1" -gt "$ca0" ] && break
  n=$((n+1)); [ $n -ge 30 ] && fail "resp_stream_abort: canceled never moved"
  sleep 2
done
hc=$(curl -s -m 10 -o /dev/null -w '%{http_code}' "$BASE/v1/models")
[ "$hc" = "200" ] || fail "resp_stream_abort: health probe HTTP $hc"
log "resp_stream_abort PASS (canceled ${ca0}->${ca1})"

# ============ boot E: Inc 4c kill switch restores legacy serial =============
BOOT_ENV="DS4_MEM_FLOOR_GB=2 DS4_SERVER_CONT_RESPONSES=0" boot
rc0=$(lane openai_responses continuous); sl0=$(m ds4_requests_serial_total)
sse_timed respk.sse respk.t /v1/responses \
  '{"stream":true,"max_output_tokens":400,"temperature":0,"input":"Name one metal."}' 180
order respk.sse '"type":"response.created"' '"type":"response.completed"'
resp_seq_check respk.sse || fail "resp_killswitch: $(resp_seq_check respk.sse 2>&1)"
rc1=$(lane openai_responses continuous); sl1=$(m ds4_requests_serial_total)
delta_is "resp_killswitch cont lane" "$rc0" "$rc1" 0
delta_is "resp_killswitch serial"    "$sl0" "$sl1" 1
log "resp_killswitch PASS (legacy serial, native bytes)"

ssh "$R" "pkill -x ds4-server; exit 0"
log "ALL LEGS PASS — artifacts in $OUT"
