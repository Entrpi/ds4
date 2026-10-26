#!/bin/bash
# memory/admission_burst.sh — v0.5.5 governance Inc0 (task #15): admission
# double-booking closed.
#
# Bug class (field: NVIDIA forum 376884 post 129, OllieOllie BUGFIXES.md,
# preserved local/docs/field-reports/): the comp-cache budget check charged
# resident pages + own projection only, so N banks admitted while earlier
# admissions' growth was still landing jointly overran the budget (same-pass
# AND cross-pass variants) and the compressor store wrote past the mapped
# extent. Fix: every other occupied bank's still-outstanding admission-time
# projection (pfbase+pflen) is charged into the check (ds4.c, Inc0).
#
# Drive (port of their multi_chat_stress shape to our conventions):
#   N_STREAMS concurrent conversations, each warms ~SEED sentences over two
#   small turns, then ALL fire a ~BIG-sentence suffix simultaneously
#   (barrier) = the synchronized multi-bank admission burst.
#
# Legs:
#   A (stock):   burst must serve clean. PASS = every stream completes,
#       zero illegal/cuBLAS/'continuous batch failed', server alive.
#   B (pinned):  DS4_BATCH_VMM_BUDGET_MB pin forces the reject path
#       deterministically (documented gate trick). PASS = at least one
#       'cont admit rejected on comp-cache budget' line CONTAINING the new
#       'outstanding' field, server alive, no crash lines. This is the
#       ENGAGEMENT receipt: the outstanding term is visible in the verdict.
#
# Runs FROM the Mac over SSH. End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
# Single-instance guard: two concurrent copies share the port, the server
# log, and kill_all -- they corrupt each other's legs (bit us 08-04: a
# duplicate spawn made instance 1's wait_mem race instance 2's boots).
LOCKDIR=/tmp/adm_burst_gate.lockdir
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "another admission_burst_gate instance is running ($LOCKDIR exists; pid $(cat "$LOCKDIR/pid" 2>/dev/null || echo '?')) -- refusing to double-drive"; exit 3
fi
echo $$ > "$LOCKDIR/pid"
trap 'rmdir_rc=$?; rm -rf "$LOCKDIR"; exit $rmdir_rc' EXIT
R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-131072}
N_STREAMS=${N_STREAMS:-4}
SEED_SENT=${SEED_SENT:-1100}     # ~17k tokens
BIG_SENT=${BIG_SENT:-1250}       # ~19k-token synchronized suffix
SRV=/tmp/admission_burst_gate.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -8 $SRV" 2>/dev/null; kill_all; exit 1; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

boot(){ # $1 = extra env
  kill_all
  wait_mem
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup env DS4_SERVER_COALESCE_MAX=$N_STREAMS $1 ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done }

drive(){ # $1 = per-leg run nonce; runs the burst driver on the box.
  # The nonce is stamped by the DRIVER into every line (08-04 collision
  # postmortem: a verdict grep over a shared tee file matched another
  # instance's output block; nonce + strict per-stream check close that).
  ssh "$R" "cat > /tmp/adm_burst_driver_$1.py <<'EOF'
import http.client, json, sys, threading, time

PORT, NS, SEED, BIG = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
NONCE = sys.argv[5]

def sent(seed, tag, n):
    return ' '.join('the tidal basin at station %d-%s-%d filled ahead of the model forecast'
                    % (seed, tag, i) for i in range(n))

def post(messages, mt=32, timeout=2400):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request('POST', '/v1/chat/completions',
              json.dumps({'model': 'ds4', 'max_tokens': mt, 'temperature': 0,
                          'reasoning_effort': 'off', 'messages': messages}),
              {'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    if r.status != 200:
        return r.status, None, data[:160]
    j = json.loads(data)
    return r.status, j['choices'][0]['message'].get('content') or '', j['choices'][0]['finish_reason']

barrier = threading.Barrier(NS, timeout=1800)
results = {}
lock = threading.Lock()

def worker(sid):
    msgs = [{'role': 'user', 'content': sent(sid + 3, 'seed', SEED) + ' Reply with exactly: OK'}]
    for turn in range(3):
        if turn == 2:
            msgs.append({'role': 'user', 'content': sent(sid + 3, 'big', BIG) + ' Reply with exactly: DONE'})
            try:
                barrier.wait()
            except threading.BrokenBarrierError:
                pass
        elif turn == 1:
            msgs.append({'role': 'user', 'content': sent(sid + 3, 't1', 12) + ' Reply with exactly: OK'})
        st, content, fin = post(msgs)
        with lock:
            print('[%s] stream=%d turn=%d st=%s fin=%s' % (NONCE, sid, turn, st, fin)); sys.stdout.flush()
        if st != 200:
            results[sid] = 'failed_turn_%d_%s' % (turn, fin)
            try: barrier.abort()
            except Exception: pass
            return
        msgs.append({'role': 'assistant', 'content': content})
    results[sid] = 'completed'

threads = [threading.Thread(target=worker, args=(s,)) for s in range(NS)]
for t in threads: t.start()
for t in threads: t.join()
for s in range(NS):
    print('[%s] RESULT stream=%d %s' % (NONCE, s, results.get(s, 'unknown')))
EOF
python3 /tmp/adm_burst_driver_$1.py $PORT $N_STREAMS $SEED_SENT $BIG_SENT $1"
}

check_streams(){ # $1 = nonce, $2 = tee file: every stream completed, none else
  local s
  for s in $(seq 0 $((N_STREAMS - 1))); do
    grep -q "^\[$1\] RESULT stream=$s completed\$" "$2" || return 1
  done
  [ "$(grep "^\[$1\] RESULT " "$2" | grep -vc "completed\$")" -eq 0 ]
}

# ---- Leg A: stock burst must serve clean --------------------------------
log "== LEG A: stock burst (streams=$N_STREAMS seed=$SEED_SENT big=$BIG_SENT) =="
boot "DS4_NOOP=0"
NONCE_A="A$(date +%s)$$"
drive "$NONCE_A" | tee /tmp/adm_burst_A.txt
check_streams "$NONCE_A" /tmp/adm_burst_A.txt || fail "leg A: not every stream completed under nonce $NONCE_A"
ssh "$R" "grep -aqE 'illegal|cuBLAS|continuous batch failed' $SRV" && fail "leg A: crash lines in server log"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "leg A: server died"
log "leg A PASS ($n_ok/$N_STREAMS clean)"

# ---- Leg B: pinned budget forces the reject path; 'outstanding' visible --
log "== LEG B: pinned-budget reject engagement =="
boot "DS4_BATCH_VMM_BUDGET_MB=${PIN_MB:-256}"
NONCE_B="B$(date +%s)$$"
drive "$NONCE_B" | tee /tmp/adm_burst_B.txt
check_streams "$NONCE_B" /tmp/adm_burst_B.txt || fail "leg B: not every stream completed under nonce $NONCE_B"
rej=$(ssh "$R" "grep -a 'cont admit rejected on comp-cache budget' $SRV" | head -3)
[ -n "$rej" ] || fail "leg B: pinned budget produced NO budget rejects (pin too big? shape drifted?)"
echo "$rej" | grep -q 'outstanding' || fail "leg B: reject line lacks the outstanding field (Inc0 not engaged): $rej"
ssh "$R" "grep -aqE 'illegal|cuBLAS|continuous batch failed' $SRV" && fail "leg B: crash lines under pinned budget"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "leg B: server died"
log "leg B PASS (reject fired with outstanding field: ${rej%%$'\n'*})"

kill_all
log "ADMISSION-BURST-GATE-PASS"
