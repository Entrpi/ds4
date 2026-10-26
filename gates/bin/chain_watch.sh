#!/usr/bin/env bash
# gates/bin/chain_watch.sh — per-event watchdog for a detached chain (release-process §3.0, 2026-09-04;
# refinements 16:4x).  Runs on the Mac (in a Monitor task or a terminal) and polls the chain's output
# directory on its DRIVER box over ssh (or locally with W_ALIAS=local).  It prints ONE line per noteworthy
# event and nothing otherwise, so a Monitor turns every line into a notification:
#   PROGRESS <new leg.log line>        a new "== stage" / "rc=" / verdict line landed (trimmed to the verdict fragment)
#   STALL idle=<min> budget=<min> stage=<last "==" line>   no file under the watched dirs changed for longer than
#                                      the stall budget: HALF the current leg's own budget when the stage line
#                                      carries "budget N min" (orchestrator chains), else W_STALL_MIN
#   DEAD ...                           heartbeat older than 3 min on two polls (orchestrator chains, W_HB), or the
#                                      driver process gone (hand chains), with no sentinel
#   SERVER-DOWN <n> ds4-server on <server> mid-stage      (only with W_SERVER and W_EXPECT_SERVER=1)
#   UNREACHABLE x<n> / RECOVERED       consecutive probe failures (announced at 5, then every 10) / back
#   DONE <final verdict line(s)>       the sentinel exists; the watchdog exits 0
# Exit-status law: every remote probe ends with `; true` so a normal poll never reads as a failure
# (the first 09-04 monitors were silent because a trailing `test -e sentinel` made every normal poll "fail").
#
# Env:  W_ALIAS   ssh alias of the DRIVER box (e.g. $DS4_BOX33) or "local" (no ssh)            [required]
#       W_RD      the chain's output dir on the driver                                        [required]
#       W_PGREP   pgrep -f pattern of the chain driver, bracketed ("[v]8_3l_chain33")          [required unless W_HB]
#       W_HB      heartbeat file (glob) under W_RD written by orchestrator chains (default: heartbeat_*; unset with W_HB=none)
#       W_STALL_MIN  fallback stall minutes when the stage line has no budget (default 40); W_STALL_REPEAT (default 30)
#       W_LOG / W_SENT  leg log and sentinel names inside W_RD (default leg.log / leg.DONE; W_SENT may be a glob)
#       W_AUX_ALIAS + W_AUX_DIR  a second activity source (e.g. the server box's soak dir) folded into idle
#       W_SERVER  ssh alias of the server box; with W_EXPECT_SERVER=1 a poll with 0 ds4-server mid-stage is SERVER-DOWN
#       W_POLL    seconds between polls (default 60);  W_ONCE=1 prints the probe once and exits (self-test)
set -uo pipefail
: "${W_ALIAS:?}" "${W_RD:?}"
W_PGREP=${W_PGREP:-}; W_HB=${W_HB:-heartbeat_*}
W_STALL_MIN=${W_STALL_MIN:-40}; W_STALL_REPEAT=${W_STALL_REPEAT:-30}; W_POLL=${W_POLL:-60}
W_LOG=${W_LOG:-leg.log}; W_SENT=${W_SENT:-leg.DONE}; W_AUX_ALIAS=${W_AUX_ALIAS:-}; W_AUX_DIR=${W_AUX_DIR:-}
W_SERVER=${W_SERVER:-}; W_EXPECT_SERVER=${W_EXPECT_SERVER:-0}; W_ONCE=${W_ONCE:-0}
[ -n "$W_PGREP" ] || [ "$W_HB" != none ] || { echo "chain_watch: need W_PGREP or a heartbeat (W_HB)"; exit 2; }
run_on(){ local where=$1; shift; if [ "$where" = local ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=12 "$where" "$@"; fi; }
seen=0; unreach=0; stall_announced=0; dead_polls=0
probe(){  # DONE / ALIVE / HB (heartbeat age s, or -1) / IDLE (s since the newest file) / then the leg log
  run_on "$W_ALIAS" "cd '$W_RD' 2>/dev/null || { echo NODIR; exit 0; }
    if ls $W_SENT >/dev/null 2>&1; then echo DONE=1; else echo DONE=0; fi
    if [ -n '$W_PGREP' ]; then echo ALIVE=\$(pgrep -f '$W_PGREP' | wc -l | tr -d ' '); else echo ALIVE=-1; fi
    hbf=\$(ls -t $W_HB 2>/dev/null | head -1); if [ -n \"\$hbf\" ] && [ '$W_HB' != none ]; then echo HB=\$(( \$(date +%s) - \$(cat \"\$hbf\" 2>/dev/null || echo 0) )); else echo HB=-1; fi
    if stat -f %m . >/dev/null 2>&1; then n=\$(find . -type f -not -name 'heartbeat_*' -exec stat -f %m {} + 2>/dev/null | sort -n | tail -1); else n=\$(find . -type f -not -name 'heartbeat_*' -exec stat -c %Y {} + 2>/dev/null | sort -n | tail -1); fi; [ -n \"\$n\" ] && echo IDLE=\$(( \$(date +%s) - n )) || echo IDLE=0
    echo ---LOG---; cat '$W_LOG' 2>/dev/null; true" 2>/dev/null
}
aux_idle(){ [ -n "$W_AUX_ALIAS" ] || { echo 999999; return; }
  run_on "$W_AUX_ALIAS" "if stat -f %m . >/dev/null 2>&1; then n=\$(find '$W_AUX_DIR' -type f -exec stat -f %m {} + 2>/dev/null | sort -n | tail -1); else n=\$(find '$W_AUX_DIR' -type f -exec stat -c %Y {} + 2>/dev/null | sort -n | tail -1); fi; [ -n \"\$n\" ] && echo \$(( \$(date +%s) - n )) || echo 999999; true" 2>/dev/null || echo 999999; }
server_count(){ [ -n "$W_SERVER" ] || { echo -; return; }; run_on "$W_SERVER" "pgrep -cx ds4-server; true" 2>/dev/null || echo -; }
trim(){ sed -E 's/^(\[[0-9:]+\])? *//' | sed -E 's/\|.*//' | cut -c1-160; }   # one fragment per event: drop the joined tails
while :; do
  out=$(probe); prc=$?
  if [ $prc -ne 0 ] || [ -z "$out" ]; then
    unreach=$((unreach+1)); { [ $unreach = 5 ] || [ $((unreach % 10)) = 0 ]; } && echo "UNREACHABLE x$unreach ($W_ALIAS)"
    [ "$W_ONCE" = 1 ] && exit 1; sleep "$W_POLL"; continue
  fi
  [ $unreach -ge 5 ] && echo "RECOVERED after $unreach failed polls"; unreach=0
  case "$out" in NODIR*) echo "NODIR $W_RD on $W_ALIAS"; [ "$W_ONCE" = 1 ] && exit 1; sleep "$W_POLL"; continue;; esac
  done_flag=$(printf '%s\n' "$out" | sed -n 's/^DONE=//p' | head -1); alive=$(printf '%s\n' "$out" | sed -n 's/^ALIVE=//p' | head -1)
  hb=$(printf '%s\n' "$out" | sed -n 's/^HB=//p' | head -1); idle=$(printf '%s\n' "$out" | sed -n 's/^IDLE=//p' | head -1)
  logtxt=$(printf '%s\n' "$out" | sed -n '/^---LOG---$/,$p' | sed '1d')
  total=$(printf '%s\n' "$logtxt" | grep -c ''); stageline=$(printf '%s\n' "$logtxt" | grep -E '^\[.*\] == ' | tail -1)
  stage=$(printf '%s\n' "$stageline" | cut -c1-120)
  budget=$(printf '%s\n' "$stageline" | grep -oE 'budget [0-9]+ min' | grep -oE '[0-9]+')
  if [ -n "$budget" ]; then stall_min=$(( budget / 2 )); [ $stall_min -lt 5 ] && stall_min=5; else stall_min=$W_STALL_MIN; fi
  if [ "$total" -gt "$seen" ]; then
    printf '%s\n' "$logtxt" | tail -n +$((seen+1)) | grep -E 'rc=|PASS|FAIL|== |VERDICT|MISS|STALL|DEAD|CAP' | grep -vE '^[0-9,.]+(,[0-9,.]+)*$' | while IFS= read -r ln; do echo "PROGRESS $(printf '%s' "$ln" | trim)"; done
    seen=$total; stall_announced=0
  fi
  if [ "$W_ONCE" = 1 ]; then echo "PROBE done=$done_flag alive=$alive hb=${hb}s idle=${idle}s stall_at=${stall_min}min stage='$stage' server=$(server_count)"; exit 0; fi
  if [ "$done_flag" = 1 ]; then printf '%s\n' "$logtxt" | grep -E 'PASS|FAIL' | tail -2 | while IFS= read -r ln; do echo "DONE $(printf '%s' "$ln" | trim)"; done; exit 0; fi
  dead_now=0
  if [ "${hb:--1}" -ge 0 ]; then [ "$hb" -gt 180 ] && dead_now=1; elif [ "${alive:--1}" = 0 ]; then dead_now=1; fi
  if [ $dead_now = 1 ]; then dead_polls=$((dead_polls+1)); [ $dead_polls -ge 2 ] && { echo "DEAD $( [ "${hb:--1}" -ge 0 ] && echo "heartbeat ${hb}s old" || echo "driver gone" ), no sentinel (stage='$stage', idle=$((idle/60))min)"; exit 2; }; else dead_polls=0; fi
  a=$(aux_idle); [ "$a" -lt "$idle" ] 2>/dev/null && idle=$a
  if [ $((idle/60)) -ge "$stall_min" ]; then
    if [ $stall_announced = 0 ] || [ $(( (idle/60 - stall_min) % W_STALL_REPEAT )) = 0 ]; then
      [ $stall_announced = 0 ] || [ $((idle/60)) -gt $stall_min ] && echo "STALL idle=$((idle/60))min budget=${stall_min}min stage='$stage' server=$(server_count)"; stall_announced=1
    fi
  fi
  if [ "$W_EXPECT_SERVER" = 1 ] && [ -n "$stage" ]; then sc=$(server_count); [ "$sc" = 0 ] && [ $((idle/60)) -ge 3 ] && echo "SERVER-DOWN 0 ds4-server on $W_SERVER mid-stage '$stage' (idle $((idle/60))min)"; fi
  sleep "$W_POLL"
done
