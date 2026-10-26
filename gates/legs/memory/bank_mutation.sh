#!/bin/bash
# memory/bank_mutation.sh — v0.5.4+: interrupted bank-mutation integrity
# (forum 376884 posts 126-127, scope_cr33p).  Field shape: v0.5.2's
# dead-client abort fires while an admission is REPLAYING onto a bank
# that carries prior state (warm cached prefix / truncate replay) and a
# later kernel on that bank dies with "cuBLAS f16 matmul failed: status
# 14" + illegal memory access.  This gate drives exactly those shapes:
#
#   leg warm  (crash-2 shape): EOS-finished turn 1 builds a ~24k warm
#       record; turn 2 warm-matches it and streams a ~12k suffix; the
#       client socket CLOSES mid-suffix (abort between chunks); then the
#       same turn 2 retries patiently and must serve clean.  xN cycles.
#   leg trunc (crash-1 shape): turn 2 shares only a ~16k prefix with the
#       record and diverges (partial admission: cut + long replay); the
#       client dies mid-replay; retry must serve clean.  xN cycles.
#   leg dthink (378855/41 shape): a THINKING row's client dies mid-DECODE;
#       the retry must resume off the prompt-only record instead of
#       re-paying the whole prefill.  x2 cycles.
#
# PASS = every abort prints 'pending admission aborted', ZERO illegal
# access / cuBLAS failures / 'continuous batch failed' anywhere, every
# retry serves HTTP 200 with content, server alive end to end.
# On the pre-fix tree this gate is the REPRO ATTEMPT for the field
# crash; on the fixed tree it pins the mutation-epilogue behavior.
#
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-131072}
CYCLES=${CYCLES:-3}
SRV=/tmp/bank_mutation_gate.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -6 $SRV" 2>/dev/null; kill_all; exit 1; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

log "== boot (stock -c $CTX, spec on: field shape) =="
kill_all
wait_mem
ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup env ${EXTRA_ENV:-DS4_NOOP=0} ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
done
log "boot ok"

ssh "$R" "cat > /tmp/bank_mutation_driver.py <<'EOF'
import http.client, json, os, re, sys, threading, time

PORT, LOG = int(sys.argv[1]), sys.argv[2]
CYCLES = int(sys.argv[3])
BAD = ('illegal', 'cuBLAS', 'continuous batch failed', 'CUDA')

def sent(seed, n):   # ~14 tok per sentence, deterministic per seed
    return ' '.join('the tidal basin at station %d-%d filled ahead of the model forecast'
                    % (seed, i) for i in range(n))

def post(messages, mt=24, stream=False, timeout=900, effort='off'):
    body = json.dumps({'model': 'ds4', 'max_tokens': mt, 'temperature': 0,
                       'reasoning_effort': effort, 'stream': stream,
                       'messages': messages})
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request('POST', '/v1/chat/completions', body,
              {'Content-Type': 'application/json'})
    return c

def read_json(c):
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if r.status == 200 else data[:200])

def log_size():
    return os.path.getsize(LOG)

def log_slice(off):
    with open(LOG, 'rb') as f:
        f.seek(off)
        return f.read().decode(errors='replace')

def wait_line(off, needle, deadline):
    t0 = time.time()
    while time.time() - t0 < deadline:
        if needle in log_slice(off): return True
        time.sleep(2)
    return False

def assert_clean(off, tag):
    tail = log_slice(off)
    for b in BAD:
        for ln in tail.splitlines():
            if b in ln:
                print('GATE-FAIL %s: crash-class line: %r' % (tag, ln[:160]))
                sys.exit(1)

def run_leg(name, base_msgs, follow_fn, kill_s):
    for cyc in range(1, CYCLES + 1):
        off = log_size()
        # turn 1: establish the record (EOS finish; reasoning off).
        st, j = read_json(post(base_msgs))
        if st != 200 or j['choices'][0]['finish_reason'] != 'stop':
            print('GATE-FAIL %s c%d: turn1 st=%s finish=%s' %
                  (name, cyc, st, j if st != 200 else j['choices'][0]['finish_reason']))
            sys.exit(1)
        reply = j['choices'][0]['message'].get('content') or ''
        # FRESH suffix per cycle: post-fix, an aborted replay is a checkpoint,
        # so retrying the SAME suffix leaves later cycles almost nothing to
        # replay and the kill outruns nothing (measured: cycle 2 replayed
        # 4.7k in <6s).  A fresh suffix keeps every cycle's abort mid-mutation.
        msgs2 = base_msgs + [{'role': 'assistant', 'content': reply},
                             {'role': 'user', 'content': follow_fn(cyc)}]
        # turn 2: stream, then DISCONNECT mid-replay.
        off2 = log_size()
        c = post(msgs2, mt=24, stream=True, timeout=30)
        time.sleep(kill_s)
        c.close()          # dead client: the abort must fire between chunks
        aborted = wait_line(off2, 'pending admission aborted', 90)
        time.sleep(3)      # let the retire (record install) land in the log
        n_keep = 0
        if not aborted:
            # the admission may have finished before the kill on a fast box:
            # count it, keep the cycle (the retry still probes the bank).
            print('%s c%d: NOTE no abort line (prefill outran the kill)' % (name, cyc))
        else:
            m = re.findall(r'bank retains (\d+) committed tokens', log_slice(off2))
            if m: n_keep = int(m[-1])
            print('%s c%d: aborted mid-mutation (retains %d)' % (name, cyc, n_keep))
        sys.stdout.flush()
        assert_clean(off, '%s c%d post-abort' % (name, cyc))
        kept = ('committed tokens as warm record' in log_slice(off2)) if n_keep else False
        # retry the SAME turn 2 patiently: the bank must serve clean.
        off3 = log_size()
        st, j = read_json(post(msgs2))
        if st != 200 or not (j['choices'][0]['message'].get('content') or '').strip():
            print('GATE-FAIL %s c%d: retry st=%s' % (name, cyc, st)); sys.exit(1)
        assert_clean(off, '%s c%d post-retry' % (name, cyc))
        cached = (j.get('usage', {}).get('prompt_tokens_details', {}) or {}).get('cached_tokens', 0)
        admits = re.findall(r'(?:cached|cut)=(\d+)', log_slice(off3))
        c_admit = int(admits[-1]) if admits else -1
        print('%s c%d: retry served clean (cached=%d admit=%d retains=%d)' %
              (name, cyc, cached, c_admit, n_keep)); sys.stdout.flush()
        # EXPECT_REUSE=1 (post-fix) receipts, per cycle:
        #   1. floor: the retry reuses a large prefix (any route).
        #   2. watermark: the abort-retained committed prefix was re-announced
        #      as a warm record AND the retry's admit reused AT LEAST that
        #      many tokens -- an aborted replay is a checkpoint, not waste.
        #   3. honesty: usage.cached_tokens equals the admit line's number
        #      (the cont path reported 0 for every admit shape before v0.5.4).
        if os.environ.get('EXPECT_REUSE') == '1':
            if cached < 12000:
                print('GATE-FAIL %s c%d: retry cached=%d < 12000 -- prefix not reused' %
                      (name, cyc, cached)); sys.exit(1)
            if n_keep:
                if not kept:
                    print('GATE-FAIL %s c%d: no watermark record install line after abort' %
                          (name, cyc)); sys.exit(1)
                if c_admit < n_keep:
                    print('GATE-FAIL %s c%d: retry admit=%d < retained %d -- watermark not routed' %
                          (name, cyc, c_admit, n_keep)); sys.exit(1)
            if c_admit >= 0 and cached != c_admit:
                print('GATE-FAIL %s c%d: usage cached=%d != admit line %d -- usage dishonest' %
                      (name, cyc, cached, c_admit)); sys.exit(1)

# DTHINK_ONLY=1: crash-repro mode -- skip the admission legs and drive only
# the decode-abort cycles (the 08-03 illegal-access repro shape: long DSpark
# decodes then big fork-replays).  DTHINK_CYCLES overrides the cycle count.
dthink_only = os.environ.get('DTHINK_ONLY') == '1'

# leg warm: turn 2 extends the record with a ~12k-token suffix.
base = [{'role': 'user', 'content':
         sent(1, 1700) + ' Reply with exactly: OK'}]
if not dthink_only:
    run_leg('warm', base,
            lambda cyc: sent(200 + cyc, 860) + ' Reply with exactly: DONE',
            kill_s=float(sys.argv[4]))

# leg trunc: turn 2 shares only the first ~16k tokens, then diverges hard
# (edited history -> partial admission: cut + long replay).
if not dthink_only:
    base_t = [{'role': 'user', 'content':
               sent(3, 1700) + ' Reply with exactly: OK'}]
    st, j = read_json(post(base_t))
    if st != 200: print('GATE-FAIL trunc seed st=%s' % st); sys.exit(1)
    edited = [{'role': 'user', 'content':
               sent(3, 1150) + ' ' + sent(4, 900) + ' Reply with exactly: OK'}]
    run_leg('trunc', edited,
            lambda cyc: sent(500 + cyc, 860) + ' Reply with exactly: DONE',
            kill_s=float(sys.argv[4]))

# leg dthink (378855/41 shape): a THINKING row's client dies mid-DECODE
# (post-prefill).  Pre-v0.5.4 the aborted row retired with NO record (the
# reasoning-strip retire path required an EOS finish), so an identical retry
# re-paid the whole prefill -- the field report's three consecutive 300s
# client timeouts each re-prefilled ~96k tokens.  Post-fix the prompt text
# alone is installed as the record and the retry resumes off it.
def run_decode_leg(name, base_msgs, follow_content, cycles=2):
    for cyc in range(1, cycles + 1):
        off = log_size()
        st, j = read_json(post(base_msgs))
        if st != 200 or j['choices'][0]['finish_reason'] != 'stop':
            print('GATE-FAIL %s c%d: turn1 st=%s' % (name, cyc, st)); sys.exit(1)
        reply = j['choices'][0]['message'].get('content') or ''
        msgs2 = base_msgs + [{'role': 'assistant', 'content': reply},
                             {'role': 'user', 'content': follow_content}]
        off2 = log_size()
        c = post(msgs2, mt=512, stream=True, timeout=900, effort='low')
        r = c.getresponse()
        r.read(64)         # first stream bytes => prefill done, decode running
        c.close()          # dead client mid-DECODE (RST: unread bytes pending)
        print('%s c%d: decode-abort landed' % (name, cyc)); sys.stdout.flush()
        # wait for the retire receipt, not a fixed sleep: the abort lands at
        # the next per-token probe, and the record install is part of the
        # retire.  Racing the retry past a still-live row forks the trunk
        # instead (measured) and asserts the wrong thing.
        kept = wait_line(off2, 'prompt as warm record', 90)
        time.sleep(2)
        assert_clean(off, '%s c%d post-abort' % (name, cyc))
        off3 = log_size()
        st, j = read_json(post(msgs2, mt=256, effort='low'))
        if st != 200:
            print('GATE-FAIL %s c%d: retry st=%s' % (name, cyc, st)); sys.exit(1)
        assert_clean(off, '%s c%d post-retry' % (name, cyc))
        cached = (j.get('usage', {}).get('prompt_tokens_details', {}) or {}).get('cached_tokens', 0)
        admits = re.findall(r'(?:cached|cut)=(\d+)', log_slice(off3))
        c_admit = int(admits[-1]) if admits else -1
        print('%s c%d: retry served clean (cached=%d admit=%d kept=%d)' %
              (name, cyc, cached, c_admit, int(kept))); sys.stdout.flush()
        if os.environ.get('EXPECT_REUSE') == '1':
            if not kept:
                print('GATE-FAIL %s c%d: no decode-abort record line' % (name, cyc)); sys.exit(1)
            if cached < 12000:
                print('GATE-FAIL %s c%d: retry cached=%d < 12000 -- prompt not reused' %
                      (name, cyc, cached)); sys.exit(1)
            if c_admit >= 0 and cached != c_admit:
                print('GATE-FAIL %s c%d: usage cached=%d != admit line %d -- usage dishonest' %
                      (name, cyc, cached, c_admit)); sys.exit(1)

base_d = [{'role': 'user', 'content':
           sent(6, 1700) + ' Reply with exactly: OK'}]
# The follow must demand a LONG generation: a short ask EOS'd at 84 tokens
# ~3s after first byte, before the close was even noticed -- the row FINISHED
# and retired via the EOS path, so there was no abort to test (measured).
# 100 stations: the natural answer (~800+ tok) always exceeds the 512
# budget, so every cycle length-finishes (finish=0) and hits the aborted-
# thinking retire branch.  At 60 stations the row sometimes EOS'd just
# under budget (stochastic thinking sampling) and retired down the silent
# EOS path instead -- a flaky assert, not a fix regression.
run_decode_leg('dthink', base_d, sent(7, 860) +
               ' Now write one short line for each of stations 1 through 100,'
               ' describing its tidal pattern in detail.',
               cycles=int(os.environ.get('DTHINK_CYCLES', '2')))
print('DRIVER-DONE')
EOF
EXPECT_REUSE=${EXPECT_REUSE:-0} DTHINK_ONLY=${DTHINK_ONLY:-0} DTHINK_CYCLES=${DTHINK_CYCLES:-2} python3 /tmp/bank_mutation_driver.py $PORT $SRV $CYCLES ${KILL_S:-6}" | tee /tmp/bank_mutation_out.txt
grep -q "GATE-FAIL" /tmp/bank_mutation_out.txt && fail "driver reported a failure"
grep -q "DRIVER-DONE" /tmp/bank_mutation_out.txt || fail "driver died"
naborts=0
if [ "${DTHINK_ONLY:-0}" != "1" ]; then
  naborts=$(grep -c "aborted mid-mutation" /tmp/bank_mutation_out.txt || true)
  [ "$naborts" -ge 3 ] || fail "only $naborts aborts landed mid-mutation (engagement too thin; tune KILL_S)"
fi
ndec=$(grep -c "decode-abort landed" /tmp/bank_mutation_out.txt || true)
[ "$ndec" -ge 1 ] || fail "no decode-abort cycles landed (dthink leg did not engage)"
ssh "$R" "grep -aq 'illegal' $SRV" && fail "illegal access in server log"
ssh "$R" "grep -aq 'continuous batch failed' $SRV" && fail "batch death in server log"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "server died"
kill_all
log "bank_mutation_gate PASS ($naborts mid-mutation aborts, all clean)"
