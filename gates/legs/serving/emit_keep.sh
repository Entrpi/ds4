#!/bin/bash
# serving/emit_keep.sh — v0.5.5 task #14: ms_emit_keep lifecycle correctness.
#
# Bug class (found 08-04 while root-causing the illegal-access crash, ledger
# local/docs/sweep_v05/shallow2k/shallow2k_attribution_notes.md):
# bank_fork_copy never reset ms_emit_keep[dst], so a bank whose PRIOR life
# was a partial-fork (rewind) dst carried stale keep = cut/4+1 into its next
# life; a later FULL fork admit with cached_rows < stale_keep made
# ms_emit_keep_restore overwrite freshly-replayed rows with the PRIOR
# tenant's stash (silent corruption), and the never-cleared keep blocked
# cont-graph capture for the bank's lifetime (eager-decode tax).
#
# Drive (COALESCE_MAX=4 => banks 0-3, from the deterministic 08-04 harness):
#   1. T (32.6k trunk)                        -> bank0        [record T]
#   2. V = T[:~31k]+diverge -> partial fork src=0 dst=1 cut~31008
#        => the suffix replay's first emits MUST fire the boundary restore
#        (log 'partial-fork boundary restore bank=1', legit engagement)
#   3. B (19.2k independent trunk) -> free bank                [record B]
#   4. V-RESEND (identical prompt): equal-length trap -> partial fork off
#        bank1 elsewhere -> identical record SUPERSEDES bank1's
#   5. TRIGGER: B+reply+12.9k follow -> full-match B, superseded bank1 is
#        the fork target: fork dst=1 cached~19.2k (rows 4752 < stale-era
#        keep 7753).  POST-FIX: dst inherits src's keep (0) -> the restore
#        must NOT fire in the trigger window; zero crash lines; response OK.
#
# PASS = restore line present in the V window, absent in the trigger window,
# fork admit landed on the shaped bank below the stale-era threshold, zero
# illegal/cuBLAS/batch-failure lines, server alive.
# Runs FROM the Mac over SSH.  End state: ds4-server killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example
R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
CTX=${CTX:-131072}
SRV=/tmp/emit_keep_gate.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_all(){ ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; rm -f /tmp/ds4.lock; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -8 $SRV" 2>/dev/null; kill_all; exit 1; }

wait_mem(){ local n=0 got=0
  while :; do
    got=$(ssh "$R" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
    [ -n "$got" ] && [ "$got" -ge 100 ] && return 0
    n=$((n+1)); [ $n -ge 36 ] && fail "MemAvailable never reached 100G"; sleep 5
  done }

log "== boot (-c $CTX, COALESCE_MAX=4, spec on) =="
kill_all
wait_mem
ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup env DS4_SERVER_COALESCE_MAX=4 ${EXTRA_ENV:-DS4_NOOP=0} ./$BIN -c $CTX --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
done
log "boot ok"

ssh "$R" "cat > /tmp/emit_keep_driver.py <<'EOF'
import http.client, json, os, re, sys, time

PORT, LOG = int(sys.argv[1]), sys.argv[2]
RESTORE = 'partial-fork boundary restore bank='
BAD = ('illegal', 'cuBLAS', 'continuous batch failed', 'CUDA error',
       'boundary-row restore failed', 'fp8 refresh failed', 'fp4 restore failed')

def sent(seed, n):
    return ' '.join('the tidal basin at station %d-%d filled ahead of the model forecast'
                    % (seed, i) for i in range(n))

def post(messages, mt=24, timeout=1800):
    c = http.client.HTTPConnection('127.0.0.1', PORT, timeout=timeout)
    c.request('POST', '/v1/chat/completions',
              json.dumps({'model': 'ds4', 'max_tokens': mt, 'temperature': 0,
                          'reasoning_effort': 'off', 'messages': messages}),
              {'Content-Type': 'application/json'})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, (json.loads(data) if r.status == 200 else data[:300])

def log_size():
    return os.path.getsize(LOG)

def log_slice(off):
    with open(LOG, 'rb') as f:
        f.seek(off)
        return f.read().decode(errors='replace')

def assert_clean(off, tag):
    for ln in log_slice(off).splitlines():
        if any(b in ln for b in BAD):
            print('GATE-FAIL %s: %r' % (tag, ln[:160])); sys.exit(1)

def must_stop(tag, st, j):
    if st != 200 or j['choices'][0]['finish_reason'] != 'stop':
        print('GATE-FAIL %s: st=%s %s' % (tag, st,
              j if st != 200 else j['choices'][0]['finish_reason']))
        sys.exit(1)
    return j['choices'][0]['message'].get('content') or ''

T = [{'role': 'user', 'content': sent(3, 2100) + ' Reply with exactly: OK'}]
V = [{'role': 'user', 'content': sent(3, 2000) + ' ' + sent(9, 60) + ' Reply with exactly: OK'}]
B = [{'role': 'user', 'content': sent(21, 1250) + ' Reply with exactly: OK'}]

off0 = log_size()
must_stop('T', *post(T))
assert_clean(off0, 'T')

offV = log_size()
must_stop('V', *post(V))
mV = re.findall(r'partial fork admit src=(\d+) dst=(\d+) cut=(\d+)', log_slice(offV))
if not mV:
    print('GATE-FAIL V: no partial fork admit'); sys.exit(1)
vbank, vcut = int(mV[-1][1]), int(mV[-1][2])
stale_keep = vcut // 4 + 1
vrestores = [ln for ln in log_slice(offV).splitlines() if RESTORE in ln]
if not vrestores:
    print('GATE-FAIL V: legit partial-fork replay fired NO boundary restore '
          '(engagement lost -- did the restore or its log line change?)')
    sys.exit(1)
mvr = re.search(r'bank=(\d+)', vrestores[0])
if not mvr or int(mvr.group(1)) != vbank:
    print('GATE-FAIL V: restore fired on bank %s, expected %d'
          % (mvr.group(1) if mvr else '?', vbank)); sys.exit(1)
assert_clean(offV, 'V')
print('V shaped: bank=%d cut=%d (stale-era keep %d); legit restore fired: %s'
      % (vbank, vcut, stale_keep, vrestores[0].strip()[-60:])); sys.stdout.flush()

offB = log_size()
replyB = must_stop('B', *post(B))
assert_clean(offB, 'B')

offR = log_size()
must_stop('Vresend', *post(V))
assert_clean(offR, 'Vresend')

offX = log_size()
trig = B + [{'role': 'assistant', 'content': replyB},
            {'role': 'user', 'content': sent(99, 860) + ' Reply with exactly: DONE'}]
st, j = post(trig)
time.sleep(3)
if st != 200 or not (j['choices'][0]['message'].get('content') or '').strip():
    print('GATE-FAIL trigger: st=%s' % st); sys.exit(1)
mX = re.findall(r'fork admit src=(\d+) dst=(\d+) cached=(\d+) suffix=(\d+)', log_slice(offX))
if not mX:
    print('GATE-FAIL trigger: no fork admit line (placement rules changed?): %r'
          % [l for l in log_slice(offX).splitlines() if 'admit' in l]); sys.exit(1)
src, dst, cached, suf = (int(x) for x in mX[-1])
if dst != vbank:
    print('GATE-FAIL trigger: fork dst=%d, wanted shaped bank %d' % (dst, vbank)); sys.exit(1)
if cached // 4 >= stale_keep:
    print('GATE-FAIL trigger: fork rows %d not below stale-era keep %d (shape drifted)'
          % (cached // 4, stale_keep)); sys.exit(1)
xrestores = [ln for ln in log_slice(offX).splitlines() if RESTORE in ln]
if xrestores:
    print('GATE-FAIL trigger: boundary restore fired on a FULL fork into a '
          'reused bank (stale ms_emit_keep leaked): %r' % xrestores[0][:160])
    sys.exit(1)
assert_clean(offX, 'trigger')
print('trigger: fork src=%d dst=%d cached=%d suffix=%d rows %d < stale-era keep %d, '
      'NO restore fired, clean' % (src, dst, cached, suf, cached // 4, stale_keep))
print('EMIT-KEEP-GATE-PASS')
EOF
python3 /tmp/emit_keep_driver.py $PORT $SRV" | tee /tmp/emit_keep_gate_out.txt
grep -q "GATE-FAIL" /tmp/emit_keep_gate_out.txt && fail "driver reported a failure"
grep -q "EMIT-KEEP-GATE-PASS" /tmp/emit_keep_gate_out.txt || fail "driver died before verdict"
ssh "$R" "grep -aq 'illegal' $SRV" && fail "illegal access in server log"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "server died"
kill_all
log "emit_keep_gate PASS"
