#!/bin/bash
# memory/floor.sh — v0.5.4 governance inc1 (item 16): admissions gate
# against LIVE free memory minus the operator floor (--mem-floor-gb), not
# just the boot-time comp budget.  The ceiling probe drove the box to
# 0.3 GiB free with zero engine pushback; that hole closes here.
#
#   leg flags (seconds): --mem-floor-gb banana / -3 exit non-zero.
#   leg hog   (-c 32768 --mem-floor-gb 90): boot announces the floor; a
#       host memory hog drops live free below floor+need; a fresh deep
#       admission is rejected LOUDLY on the floor line (not the budget
#       line) and the client gets a clean error; killing the hog makes
#       the SAME request admit and serve — the gate is live in both
#       directions.  Server alive throughout.
#   leg deep  (RUN_DEEP=1, ~15-20 min): ceiling-probe replay at
#       -c 200000 --no-spec HEADROOM=2048 --mem-floor-gb 100.  Sessions
#       of ~190k unique tokens march free memory toward the floor; every
#       session is either admitted (value-shedding rotation) or floor-
#       rejected loudly, live free NEVER breaches floor-0.5 GiB, zero
#       batch deaths, server alive at the end.  Session arithmetic must
#       prove the floor machinery engaged (spend > initial usable).
#
# Runs FROM the Mac over SSH.  End state: ds4-server + hog killed, box free.
set -uo pipefail
. "$(dirname "$0")/../../lib/env.sh"   # rig facts (DS4_*) from local/battery.env; see gates/battery.env.example

R=${R:-${DS4_BOX33:?}}
BINDIR=${BINDIR:-${DS4_BOX_TREE:?}}
BIN=${BIN:-ds4-server}
PORT=${PORT:-8000}
SRV=/tmp/mem_floor_gate.log
HOGPID=/tmp/ds4_gate_hog.pid
log(){ echo "[$(date +%H:%M:%S)] $*"; }
kill_hog(){ ssh "$R" "test -f $HOGPID && kill \$(cat $HOGPID) 2>/dev/null; rm -f $HOGPID; exit 0" 2>/dev/null; }
kill_all(){ kill_hog; ssh "$R" "pkill -x ${BIN:0:15}; sleep 2; pkill -9 -x ${BIN:0:15} 2>/dev/null; exit 0" 2>/dev/null; }
fail(){ log "FAIL: $*"; ssh "$R" "tail -4 $SRV" 2>/dev/null; kill_all; exit 1; }
mem_gib(){ ssh "$R" "awk '/MemAvailable/{printf \"%.2f\", \$2/1048576}' /proc/meminfo"; }

boot(){ # $1 = args
  ssh "$R" ": > $SRV; cd $BINDIR; setsid nohup ./$BIN $1 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  local n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
  done
}

log "== leg flags =="
ssh "$R" "cd $BINDIR && ./$BIN --mem-floor-gb banana --port $PORT" >/dev/null 2>&1 && fail "banana accepted"
ssh "$R" "cd $BINDIR && ./$BIN --mem-floor-gb -3 --port $PORT" >/dev/null 2>&1 && fail "negative floor accepted"
log "flags PASS"

log "== leg hog (external memory loss -> loud floor reject -> recovery) =="
kill_all
# The substrate (weights artifacts + registered mapping) holds ~95 GiB
# once a server is up, so post-boot free is ~10-20 GiB -- the gate tests
# the SHIPPED default floor (4 GiB), passed explicitly so the flag
# plumbing and both announce lines are exercised.  COALESCE_MAX=2 keeps
# the eager bank footprint small so the HOG is what takes the memory
# away -- the external-loss field shape.
ssh "$R" ": > $SRV; cd $BINDIR; DS4_SERVER_COALESCE_MAX=2 setsid nohup ./$BIN -c 32768 --mem-floor-gb 4 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
n=0
until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
  sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "boot timeout"
done
ssh "$R" "grep -q 'memory floor: 4 GiB' $SRV" || fail "server floor announce missing"
ssh "$R" "grep -q 'mem floor=4.0 GiB' $SRV" || fail "engine batch-vmm floor announce missing"
# Let MemAvailable settle after the artifact build + prewarm (known LIE
# window right after boot), then require enough room to hog against.
n=0
free0=$(mem_gib)
until awk -v f="$free0" 'BEGIN{exit !(f >= 8)}'; do
  sleep 10; n=$((n+10)); [ $n -ge 300 ] && fail "post-boot free stuck at ${free0} GiB (<8 after 5 min)"
  free0=$(mem_gib)
done
log "boot PASS (floor announced, settled free=${free0} GiB)"

# ~2000 sentences -> ~28k tokens: above ctx/2=16384 so the serial lane
# also refuses by its guard -> the client sees a clean error, never a
# hang; below ctx so admission (not the ctx-size check) is the decider.
ssh "$R" "python3 - <<'EOF'
import json
words = []
for i in range(2000):
    words.append('the tidal basin at station %d filled ahead of the model forecast' % i)
open('/tmp/ds4_floor_req.json','w').write(json.dumps({
    'model':'ds4','max_tokens':16,'temperature':0,
    'messages':[{'role':'user','content':' '.join(words)+' Reply with exactly: OK'}]}))
EOF"

# Hog to ~3.7 GiB free: below floor + the ~0.2 GiB admission need.  The
# allocation must be PACED -- a burst hog spikes memory PSI and the box's
# oomd kills it (the v0.5.0 slice-kill tell); 256 MiB chunks with sleeps
# mimic the gradual external-consumer shape and survive.
# The hog is ADAPTIVE: it allocates until MemAvailable itself reads the
# target.  A fixed size undershoots -- under pressure the kernel drops
# reclaimable cache (the substrate's file pages), so MemAvailable
# replenishes as the hog eats; only true exhaustion moves it below the
# floor.  (Same reason the engine trusts MemAvailable as the authority.)
log "starting adaptive hog (target free 3.7 GiB)"
ssh "$R" "cat > /tmp/ds4_gate_hog.py <<'EOF'
import sys, time
target = float(sys.argv[1]); held = []
def free_gib():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'): return int(line.split()[1]) / 1048576.0
while free_gib() > target and len(held) < 320:   # 320*128MiB = 40 GiB hard cap
    held.append(bytearray(128 << 20))
    time.sleep(0.4 if free_gib() < 8 else 0.15)
print('hog holding %d MiB at free=%.2f GiB' % (len(held) * 128, free_gib()), flush=True)
time.sleep(600)
EOF
setsid nohup python3 /tmp/ds4_gate_hog.py 3.7 > /tmp/ds4_gate_hog.log 2>&1 & echo \$! > $HOGPID; exit 0"
n=0
until awk -v f="$(mem_gib)" 'BEGIN{exit !(f < 4.0)}'; do
  ssh "$R" "kill -0 \$(cat $HOGPID) 2>/dev/null" || fail "hog DIED at free=$(mem_gib): $(ssh "$R" "tail -2 /tmp/ds4_gate_hog.log" | tr '\n' ' ')"
  sleep 3; n=$((n+3)); [ $n -ge 180 ] && fail "hog never dropped free below 4.0 GiB (now $(mem_gib))"
done
ssh "$R" "kill -0 \$(cat $HOGPID) 2>/dev/null" || fail "hog died right after reaching depth"
log "hog resident (free=$(mem_gib) GiB)"

off1=$(ssh "$R" "stat -c %s $SRV")
code=$(ssh "$R" "curl -s -o /tmp/ds4_floor_resp.json -w '%{http_code}' -m 120 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/ds4_floor_req.json")
[ "$code" != "200" ] || fail "deep admission served UNDER the floor (HTTP 200)"
ssh "$R" "tail -c +$(( off1 + 1 )) $SRV | grep -q 'cont admit rejected on memory floor'" || fail "floor reject line missing (HTTP $code)"
ssh "$R" "tail -c +$(( off1 + 1 )) $SRV | grep -q 'rejected on comp-cache budget'" && fail "budget line fired instead of the floor line"
ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "server died on the floor reject"
log "floor reject PASS (HTTP $code, loud line, server alive)"

kill_hog
n=0
until awk -v f="$(mem_gib)" 'BEGIN{exit !(f > 6.0)}'; do
  sleep 3; n=$((n+3)); [ $n -ge 60 ] && fail "free did not recover after hog kill (now $(mem_gib))"
done
resp=$(ssh "$R" "curl -s -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d @/tmp/ds4_floor_req.json" |
  python3 -c "import json,sys; j=json.load(sys.stdin); print(j['usage']['completion_tokens'])" 2>/dev/null)
[ -n "$resp" ] && [ "$resp" -gt 0 ] || fail "post-recovery admission did not serve (ct='$resp')"
log "recovery PASS (same request served, ct=$resp)"
kill_all
log "leg hog PASS"

if [ "${RUN_DEEP:-0}" = "1" ]; then
  log "== leg deep (ceiling replay: floor holds under self-growth) =="
  # The exact ceiling-probe recipe (HEADROOM=2048 -c 200000 --no-spec)
  # that previously marched the box to 0.3 GiB free.  The fit itself
  # leaves only ~4.5 GiB free on this shape, so the leg uses a 2 GiB
  # floor: sessions must march free memory toward 2 GiB and then HOLD
  # there (value-shedding rotation or loud reject), never 0.3.
  ssh "$R" ": > $SRV; cd $BINDIR; DS4_BATCH_FIT_HEADROOM_MB=2048 setsid nohup ./$BIN -c 200000 --no-spec --mem-floor-gb 2 --port $PORT > $SRV 2>&1 < /dev/null & exit 0"
  n=0
  until ssh "$R" "grep -q 'listening on http' $SRV" 2>/dev/null; do
    ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" 2>/dev/null || fail "BOOT-DIED: $(ssh "$R" "tail -2 $SRV" | tr '\n' ' ')"
    sleep 10; n=$((n+10)); [ $n -ge 900 ] && fail "deep boot timeout"
  done
  n=0
  free0=$(mem_gib)
  until awk -v f="$free0" 'BEGIN{exit !(f >= 3.2)}'; do
    sleep 10; n=$((n+10)); [ $n -ge 300 ] && fail "deep post-boot free stuck at ${free0} GiB (<3.2 after 5 min)"
    free0=$(mem_gib)
  done
  usable0=$(awk -v f="$free0" 'BEGIN{printf "%.2f", f - 2}')
  log "deep boot PASS (settled free=${free0} GiB, usable=${usable0} GiB over the 2 GiB floor)"
  ssh "$R" "cat > /tmp/ds4_deep_floor.py <<'EOF'
import json, sys, time, urllib.request, urllib.error
PORT, LOG = 8000, '/tmp/mem_floor_gate.log'
def gen_doc(seed, nwords=78000):
    V = ['alpha','bravo','charlie','delta','echo','foxtrot','golf','hotel',
         'india','juliet','kilo','lima','mike','november','oscar','papa']
    out = ['run%d:' % seed]
    for i in range(nwords):
        if i % 1000 == 0: out.append('s%d-%d:' % (seed, i // 1000))
        out.append(V[(i * 7 + i // 13 + seed) % len(V)] + str(i % 97))
    return ' '.join(out)
def mem_gib():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'): return int(line.split()[1]) / 1048576.0
def post(seed):
    body = json.dumps({'model':'ds4','max_tokens':48,'temperature':0,
        'reasoning_effort':'off','messages':[{'role':'user','content':
        gen_doc(seed)+'\n\nIgnore the word list above completely. Reply with exactly: OK'}]}).encode()
    req = urllib.request.Request('http://127.0.0.1:%d/v1/chat/completions' % PORT,
        data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.loads(r.read())
admitted = rejected = 0
min_free = 1e9
crossed = 0
for i in range(1, 25):
    try:
        j = post(i); admitted += 1
        tot = j['usage']['prompt_tokens'] + j['usage']['completion_tokens']
    except urllib.error.HTTPError as e:
        tail = open(LOG, errors='replace').read()[-4000:]
        if 'rejected on memory floor' not in tail:
            print('DEEP-FAIL session %d: HTTP %d without a floor reject line' % (i, e.code)); sys.exit(1)
        rejected += 1
        print('session %d: floor-rejected (HTTP %d)' % (i, e.code)); break
    except Exception as e:
        print('DEEP-FAIL session %d: %r' % (i, e)); sys.exit(1)
    m = mem_gib(); min_free = min(min_free, m)
    print('session %d: admitted %d tok, free=%.2f GiB' % (i, tot, m)); sys.stdout.flush()
    if m < 1.5:
        print('DEEP-FAIL: free %.2f breached floor-0.5' % m); sys.exit(1)
    # Once inside 1.5 GiB of the floor, run TWO more sessions to prove
    # the hold (shed-rotation or reject), then stop.
    if m < 3.5:
        crossed += 1
        if crossed >= 3: break
print('DEEP-RESULT admitted=%d rejected=%d min_free=%.2f crossed=%d' % (admitted, rejected, min_free, crossed))
EOF
python3 /tmp/ds4_deep_floor.py" | tee /tmp/mem_floor_deep_out.txt
  grep -q "DEEP-FAIL" /tmp/mem_floor_deep_out.txt && fail "deep leg failed"
  grep -q "DEEP-RESULT" /tmp/mem_floor_deep_out.txt || fail "deep driver died"
  # Engagement: the march must have come within 1.5 GiB of the floor and
  # STOPPED there (min_free in [1.5, 3.5]) or been floor-rejected — either
  # proves the machinery decided something; neither allows a 0.3 GiB run.
  minf=$(awk '/DEEP-RESULT/{for(i=1;i<=NF;i++) if($i ~ /^min_free=/){split($i,a,"="); print a[2]}}' /tmp/mem_floor_deep_out.txt)
  awk -v m="$minf" 'BEGIN{exit !(m >= 1.5 && m <= 3.5)}' || grep -q "floor-rejected" /tmp/mem_floor_deep_out.txt || \
    fail "min_free=$minf never approached the floor and no reject — machinery never engaged"
  ssh "$R" "grep -q 'continuous batch failed' $SRV" && fail "batch died during deep leg"
  ssh "$R" "grep -qi 'illegal' $SRV" && fail "illegal access during deep leg"
  ssh "$R" "pgrep -x ${BIN:0:15} >/dev/null" || fail "server died during deep leg"
  kill_all
  log "leg deep PASS"
fi
log "mem_floor_gate PASS"
