#!/usr/bin/env bash
# gates/bin/release_battery.sh — the release battery orchestrator (release-process §3.0, 2026-09-04).
# Runs the manifest's legs tier by tier on BOTH GB10s at once: within a tier the legs are grouped by
# SERVER box, each group becomes one detached box-side chain on its DRIVER box (the BOX-SIDE DRIVER LAW:
# a driver that ssh-es its server host runs on the OTHER box), every leg leaves `leg.out` + `rc` in its
# own directory, and the chain appends exactly one
#   VERDICT leg=<name> PASS|FAIL rc=<n> <matched line>
# line per leg (the VERDICT-LINE LAW: the verdict is the member driver's own line, matched by the
# manifest's regex, never re-derived).  A FAIL in a tier marked stop=1 ends the run before the next
# tier starts.  Results are pulled back into $OUT on the driving workstation with an index.md.
#
# Manifest: gates/battery.manifest, RENDERED from the legs' headers by gates/bin/manifest.sh (edit the
# leg, re-render).  One leg per line, fields separated by ' @@ ' (regexes and commands may contain '|'),
# full-line '#' comments only:
#   tier @@ stop @@ leg @@ server @@ driver @@ trigger @@ est_min @@ verdict_regex @@ command
#     server/driver: 33 | 15 | mac      trigger: always | <free text: the decision goes in index.md>
#     command: run on the DRIVER in its ds4 tree with env RB_LEG, RB_OUT (leg dir), RB_SERVER (name of the
#              server box as seen FROM THE DRIVER), RB_TREE (the server's tree), RB_GGUF, RB_VBASE, RB_ENC,
#              RB_RUN, RB_NONCE, RB_LOCK_DIR
#   Legs whose trigger is not "always" run only when listed in RUN_TRIGGERED="leg1 leg2" (or ALL=1);
#   skipped triggered legs are written into index.md as "SKIPPED (trigger: ...)" — never implicit.
#
# Rig facts (hosts, trees, models) come from local/battery.env through gates/lib/env.sh; nothing about a
# machine is written here.  Robustness: every leg runs under a time budget of BUDGET_MULT x its estimate
# (floor BUDGET_FLOOR_MIN) through gates/bin/tmo.sh (rc 124/137/142 => LEG_TIMEOUT), every chain keeps a
# 30 s heartbeat file, the wait loop reports STALL (no file change > 40 min), DEAD (heartbeat > 3 min,
# twice) and CAP (sum of budgets + 15 min); legs without a verdict are FAIL rows, never blanks.  Each
# chain registers itself in $DS4_CHAIN_REG for gates/bin/chain_watch.sh + rearm_watchers.sh.
# Usage:   OUT=local/docs/<arc>/battery-<date> bash gates/bin/release_battery.sh [TIERS="0 1 2 2b 3"]
#          DRY=1 prints the generated chains without launching.   RESUME=1 skips legs with a VERDICT.
#          LOCK_OWNER=<nonce> accepts a box lock this session already holds (asserted, never removed).
set -uo pipefail
[ "${BASH_VERSINFO[0]}" -ge 4 ] || { echo "release_battery.sh needs bash >= 4 (macOS: /opt/homebrew/bin/bash)"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd); HERE=${RB_HERE:-$HERE}   # RB_HERE: the real gates/bin when running from the self-copy below
. "$HERE/../lib/env.sh"
GATES=$DS4_GATES; REPO=$DS4_REPO
MANIFEST=${MANIFEST:-$GATES/battery.manifest}
OUT=${OUT:-$REPO/local/docs/battery_$(date +%Y%m%d_%H%M)}; mkdir -p "$OUT"
# Run from a private copy: bash reads a script incrementally, so an in-place edit of this file while a
# battery runs shifts the live instance's read offset (the 09-04 re-gate died with a syntax error on its
# final line after such an edit).  The copy is immune; HERE keeps pointing at the real gates/bin dir.
if [ -z "${RB_SELF_COPY:-}" ]; then cp -f "$0" "$OUT/orchestrator.sh"; RB_SELF_COPY=1 RB_HERE="$HERE" exec bash "$OUT/orchestrator.sh" "$@"; fi
TIERS=${TIERS:-"0 1 2 2b 3"}   # a tier is a label; "2b" runs after "2" (one driver box per server box per tier); "3m" = manual pointers, never run
RUN_TRIGGERED=${RUN_TRIGGERED:-}; ALL=${ALL:-0}; DRY=${DRY:-0}; RESUME=${RESUME:-0}
BUDGET_MULT=${BUDGET_MULT:-3}; BUDGET_FLOOR_MIN=${BUDGET_FLOOR_MIN:-10}   # per-leg budget = max(floor, est x mult) minutes
rig DS4_BOX33 DS4_BOX15 DS4_BOX33_LAN DS4_BOX15_LAN DS4_BOX_TREE DS4_BOX_SCRATCH
GGUF=${GGUF:-${DS4_GGUF_VISION_DIR:-${DS4_GGUF_DIR:?}}}   # the catalog dir the deep gates boot from (the vision catalog when the release ships vision)
VBASE=${VBASE:-${DS4_VISION_BASE:-}}; ENC=${ENC:-${DS4_VISION_ENCODER:-}}
LOCK=$DS4_BOX_LOCK_DIR; SCRATCH=$DS4_BOX_SCRATCH
RUNTAG=$(basename "$OUT")
LOCK_OWNER=${LOCK_OWNER:-}                 # a box-lock nonce this session already holds: accepted, asserted, never removed
RB_NONCE=${LOCK_OWNER:-release_battery $RUNTAG}
LOG=$OUT/battery.log; : > "$LOG"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
mac_alias(){ case $1 in 33) echo "$DS4_BOX33";; 15) echo "$DS4_BOX15";; mac) echo mac;; esac; }   # the box as the workstation reaches it
lan(){ case $1 in 33) echo "$DS4_BOX33_LAN";; 15) echo "$DS4_BOX15_LAN";; mac) echo mac;; esac; }  # the box as the OTHER box reaches it
tree(){ case $1 in 33) echo "$DS4_BOX_TREE";; 15) echo "${DS4_BOX15_TREE:-$DS4_BOX_TREE}";; mac) echo "$REPO";; esac; }
box_run(){ local box=$1; shift; if [ "$box" = mac ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=15 "$(mac_alias $box)" "$@"; fi; }

# ---- parse the manifest ----
LEGS=(); while IFS= read -r line; do   # full-line comments only: commands may contain '#' (e.g. ${g#*:})
  case "$line" in \#*) continue;; esac; [ -z "${line// /}" ] && continue; LEGS+=("$line"); done < "$MANIFEST"
field(){ echo "$1" | awk -F' @@ ' -v n=$2 '{gsub(/^ +| +$/,"",$n); print $n}'; }
should_run(){ local trig=$1 leg=$2; [ "$trig" = always ] && return 0; [ "$ALL" = 1 ] && return 0
  case " $RUN_TRIGGERED " in *" $leg "*) return 0;; esac; return 1; }

# ---- box preconditions (locks, idle): only the GB10s the selected legs actually use ----
BOXES=""; for l in "${LEGS[@]}"; do case " $TIERS " in *" $(field "$l" 1) "*) ;; *) continue;; esac
  should_run "$(field "$l" 6)" "$(field "$l" 3)" || continue
  for b in $(field "$l" 4) $(field "$l" 5); do [ "$b" = mac ] && continue; case " $BOXES " in *" $b "*) ;; *) BOXES="$BOXES $b";; esac; done; done
# The lock trap is armed BEFORE any nonce is written and acts only on TAKEN (the boxes whose nonce this run
# wrote): an exit 3 on the second box's precondition used to leave the first box locked (review 09-05 F10).
TAKEN=""
if [ -n "$LOCK_OWNER" ]; then trap 'for b in $TAKEN; do [ "$DRY" = 1 ] || box_run $b "mkdir -p $LOCK && echo '"'"'$RB_NONCE'"'"' > $LOCK/nonce" >/dev/null 2>&1; done' EXIT
else trap 'for b in $TAKEN; do [ "$DRY" = 1 ] || box_run $b "rm -rf $LOCK" >/dev/null 2>&1; done' EXIT; fi
for b in $BOXES; do
  n=$(box_run $b "cat $LOCK/nonce 2>/dev/null || echo __FREE__"); [ -n "$n" ] || { log "FAIL: box .$b unreachable at the precondition (no answer from ssh)"; exit 3; }   # an empty answer is not "free" (review 09-05, K3 4 / GLM 7)
  srv=$(box_run $b "pgrep -x ds4-server | wc -l | tr -d ' '; true" | head -1)   # pgrep -c exits 1 on zero matches
  [ -n "$srv" ] || { log "FAIL: box .$b unreachable at the precondition (no server count)"; exit 3; }
  log "box .$b: lock=$n ds4-server=$srv"
  if [ "$DRY" != 1 ]; then
    case "$n" in __FREE__|"") ;; "$RB_NONCE") ;; "$LOCK_OWNER") ;; *) log "FAIL: box .$b lock held by another session ($n)"; exit 3;; esac   # only THIS run's nonce (a RESUME) or the held one; another battery's nonce is another session (review 09-05 F2)
    [ "${srv:-0}" = 0 ] || { log "FAIL: a ds4-server is up on .$b"; exit 3; }
  fi
  [ "$DRY" = 1 ] || { box_run $b "mkdir -p $LOCK && echo '$RB_NONCE' > $LOCK/nonce"; TAKEN="$TAKEN $b"; }
done

index_init(){ { echo "# Release battery $RUNTAG"; echo; echo "| Tier | Leg | Server | Driver | Trigger | Verdict | rc | Minutes | Receipt |"; echo "|---|---|---|---|---|---|---|---|---|"; } > "$OUT/index.md"; }
index_row(){ echo "| $1 | $2 | .$3 | .$4 | $5 | $6 | $7 | $8 | \`$9\` |" >> "$OUT/index.md"; }
[ "$RESUME" = 1 ] && [ -f "$OUT/index.md" ] || index_init

overall=PASS
for tier in $TIERS; do
  log "== tier $tier"
  declare -A GROUP=() GDRIVER=() STOP=()
  for l in "${LEGS[@]}"; do
    [ "$(field "$l" 1)" = "$tier" ] || continue
    leg=$(field "$l" 3); srv=$(field "$l" 4); drv=$(field "$l" 5); trig=$(field "$l" 6); stop=$(field "$l" 2)
    if ! should_run "$trig" "$leg"; then log "   $leg: SKIPPED (trigger: $trig)"; index_row $tier $leg $srv $drv "$trig" "SKIPPED (trigger)" - - -; continue; fi
    if [ "$RESUME" = 1 ] && grep -q "VERDICT leg=$leg " "$OUT/$leg/leg.out" 2>/dev/null; then
      pv=$(grep -E "^VERDICT leg=$leg " "$OUT/$leg/leg.out" | tail -1 | awk '{print $3}')
      log "   $leg: RESUME skip (recorded $pv)"
      [ "$pv" = PASS ] || { overall=FAIL; [ "$stop" = 1 ] && stop_now=1; }   # a recorded FAIL still rules the run (review 09-05 F1)
      continue; fi
    key=$srv; [ -n "${GDRIVER[$key]:-}" ] && [ "${GDRIVER[$key]}" != "$drv" ] && { log "FAIL: tier $tier server .$srv has two driver boxes (${GDRIVER[$key]} vs $drv): split the tier"; exit 4; }
    GDRIVER[$key]=$drv; GROUP[$key]+="$l"$'\n'; [ "${STOP[$key]:-0}" = 1 ] || STOP[$key]=$stop   # OR over the group's legs (review 09-05 F12)
  done
  [ ${#GROUP[@]} = 0 ] && { log "   (no legs)"; continue; }
  declare -A cap_est=()
  for srv in "${!GROUP[@]}"; do c=15; while IFS= read -r l; do [ -z "$l" ] && continue; e=$(field "$l" 7); b=$((e*BUDGET_MULT)); [ $b -lt $BUDGET_FLOOR_MIN ] && b=$BUDGET_FLOOR_MIN; c=$((c+b)); done <<< "${GROUP[$srv]}"; cap_est[$srv]=$c; done
  # ---- generate + launch one chain per server-box group ----
  for srv in "${!GROUP[@]}"; do
    drv=${GDRIVER[$srv]}; chain=$OUT/chain_t${tier}_s${srv}.sh; sent="rb_${RUNTAG}_t${tier}_s${srv}.DONE"
    dtree=$(tree $drv); rbserver=$( [ "$drv" = mac ] && mac_alias $srv || lan $srv )
    remote_out=$( [ "$drv" = mac ] && echo "$OUT" || echo "$SCRATCH/rb_${RUNTAG}" )
    {
      echo '#!/bin/bash'; echo "set -uo pipefail; cd '$dtree'"
      echo "export RB_SERVER='$rbserver' RB_GGUF='$GGUF' RB_TREE='$(tree $srv)' RB_VBASE='$VBASE' RB_ENC='$ENC' RB_RUN='$RUNTAG' RB_NONCE='$RB_NONCE' RB_LOCK_DIR='$LOCK' RB_SELF='$( [ "$srv" = "$drv" ] && echo 1 || echo 0 )'"   # RB_SELF=1: the driver IS the server box (box-side legs); server-side actions run locally
      echo "OUTD='$remote_out'; mkdir -p \"\$OUTD\"; SENT=\"\$OUTD/$sent\"; rm -f \"\$SENT\"; CH=\"\$OUTD/chain_t${tier}_s${srv}.log\"; : > \"\$CH\""
      echo 'log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$CH"; }'
      echo 'kill_srv(){ if [ "$RB_SERVER" = mac ] || [ -z "$RB_SERVER" ]; then :; elif [ "$RB_SELF" = 1 ]; then for p in $(pgrep -x ds4-server); do kill $p; done; sleep 2; for p in $(pgrep -x ds4-server); do kill -9 $p; done; else ssh -o BatchMode=yes "$RB_SERVER" "for p in \$(pgrep -x ds4-server); do kill \$p; done; sleep 2; for p in \$(pgrep -x ds4-server); do kill -9 \$p; done; exit 0" 2>/dev/null; fi; }'
      # after every leg the battery nonce is re-asserted on the server box (only when the lock is free or already ours)
      echo 'reassert_nonce(){ if [ "$RB_SERVER" = mac ] || [ -z "$RB_SERVER" ]; then :; elif [ "$RB_SELF" = 1 ]; then n=$(cat $RB_LOCK_DIR/nonce 2>/dev/null); case "$n" in ""|release_battery*|"$RB_NONCE") mkdir -p $RB_LOCK_DIR && echo "$RB_NONCE" > $RB_LOCK_DIR/nonce;; *) echo "   NONCE-FOREIGN $n";; esac; else ssh -o BatchMode=yes "$RB_SERVER" "n=\$(cat $RB_LOCK_DIR/nonce 2>/dev/null); case \"\$n\" in \"\"|release_battery*|\"$RB_NONCE\") mkdir -p $RB_LOCK_DIR && echo \"$RB_NONCE\" > $RB_LOCK_DIR/nonce;; *) echo \"NONCE-FOREIGN \$n\";; esac; exit 0" 2>/dev/null | sed "s/^/   /"; fi; }'
      # per-leg time budget: GNU timeout where present, a perl process-group alarm elsewhere (see tmo.sh)
      echo 'tmo(){ bash "'"$dtree"'/gates/bin/tmo.sh" "$@"; }'
      # heartbeat: a dead chain is distinguishable from a slow leg (chain_watch.sh / the orchestrator read its age)
      echo "HB=\"\$OUTD/heartbeat_t${tier}_s${srv}\"; ( while :; do date +%s > \"\$HB\"; sleep 30; done ) & HBPID=\$!; trap 'kill \$HBPID 2>/dev/null' EXIT"
      while IFS= read -r l; do [ -z "$l" ] && continue
        leg=$(field "$l" 3); est=$(field "$l" 7); rx=$(field "$l" 8); cmd=$(field "$l" 9)
        budget=$(( est * BUDGET_MULT )); [ $budget -lt $BUDGET_FLOOR_MIN ] && budget=$BUDGET_FLOOR_MIN
        # PROVENANCE (09-05): a box driver's RD is scratch shared across runs -- wipe the leg dir so a leg that never
        # starts cannot leave another run's leg.out (and its VERDICT) behind for the pull; the Mac's OUT is the receipt
        # dir itself and keeps today's behavior (RESUME already decided this leg has no recorded verdict)
        echo "export RB_LEG='$leg' RB_OUT=\"\$OUTD/$leg\"; $( [ "$drv" = mac ] || echo 'rm -rf "$RB_OUT"; ' )mkdir -p \"\$RB_OUT\"; t0=\$(date +%s)"
        echo "log \"== $leg (est ${est} min, budget ${budget} min)\""
        printf '%s\n' "cat > \"\$RB_OUT/cmd.sh\" <<'RB_CMD_EOF'" "$cmd" "RB_CMD_EOF"
        echo "tmo $((budget*60)) bash \"\$RB_OUT/cmd.sh\" > \"\$RB_OUT/leg.out\" 2>&1; rc=\$?; echo \$rc > \"\$RB_OUT/rc\"; [ \$rc = 124 ] || [ \$rc = 137 ] || [ \$rc = 142 ] && echo \"LEG_TIMEOUT budget=${budget}min rc=\$rc\" >> \"\$RB_OUT/leg.out\""
        echo "mfull=\$(grep -E '$rx' \"\$RB_OUT/leg.out\" | tail -1); m=\$(printf '%s' \"\$mfull\" | cut -c1-200); v=FAIL; [ \$rc = 0 ] && [ -n \"\$mfull\" ] && case \"\$mfull\" in *FAIL*) v=FAIL;; *) v=PASS;; esac"   # the FAIL test reads the whole matched line; only the display is truncated (review 09-05, GLM 7)
        echo "[ -n \"\$m\" ] || m=\"(no line matched the verdict regex: $rx)\""
        echo "echo \"VERDICT leg=$leg \$v rc=\$rc \$(( (\$(date +%s)-t0)/60 ))min \$m\" | tee -a \"\$RB_OUT/leg.out\" >> \"\$CH\"; log \"   \$v rc=\$rc \$m\""
        echo "kill_srv; reassert_nonce"
      done <<< "${GROUP[$srv]}"
      echo 'log "== CHAIN_DONE"; kill $HBPID 2>/dev/null; touch "$SENT"'
    } > "$chain"; chmod +x "$chain"
    if [ "$DRY" = 1 ]; then log "   [dry] chain for server .$srv on driver .$drv: $chain"; sed 's/^/      /' "$chain" | head -40; continue; fi
    # a RESUME re-launching a chain of the same name: its previous sentinel must be gone BEFORE the launch,
    # or the wait loop's first poll reads the stale file as "done" (the 09-05 03:00 real_budgets re-run:
    # "FAIL (no verdict)" two seconds after launch while the chain ran on for 35 min)
    if [ "$drv" = mac ]; then rm -f "$remote_out/$sent"; else box_run $drv "rm -f '$remote_out/$sent'" >/dev/null 2>&1; fi
    REG=$DS4_CHAIN_REG; mkdir -p "$REG"; regf="$REG/rb_${RUNTAG}_t${tier}_s${srv}.env"
    { echo "ALIAS=$( [ "$drv" = mac ] && echo local || mac_alias $drv )"; echo "RD=$remote_out"; echo "PGREP='[c]hain_t${tier}_s${srv}.sh'"; echo "SENT=$sent"; echo "LOG=chain_t${tier}_s${srv}.log"; echo "HB=heartbeat_t${tier}_s${srv}"
      [ "$srv" = mac ] || echo "SERVER=$(mac_alias $srv)"; echo "EXPECT_SERVER=0"; echo "CAP_MIN=$(( ${cap_est[$srv]:-120} + 30 ))"; echo "NOTE='release battery $RUNTAG tier $tier server .$srv'"; } > "$regf"
    if [ "$drv" = mac ]; then ( nohup perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' bash "$chain" > "$OUT/chain_t${tier}_s${srv}.out" 2>&1 < /dev/null & ); log "   launched (mac) chain for server .$srv"
    else rsync -q -t "$chain" "$(mac_alias $drv):$SCRATCH/rb_${RUNTAG}_chain_t${tier}_s${srv}.sh" && box_run $drv "setsid --fork bash -c 'bash $SCRATCH/rb_${RUNTAG}_chain_t${tier}_s${srv}.sh > $SCRATCH/rb_${RUNTAG}_chain_t${tier}_s${srv}.out 2>&1 < /dev/null' > /dev/null 2>&1 < /dev/null; echo launched" | sed 's/^/   /'; log "   launched (.$drv) chain for server .$srv"; fi
  done
  [ "$DRY" = 1 ] && continue
  # ---- wait for every chain of the tier: sentinel, heartbeat (DEAD), idle (STALL), and a cap of sum(budgets)+15 ----
  for srv in "${!GROUP[@]}"; do
    drv=${GDRIVER[$srv]}; sent="rb_${RUNTAG}_t${tier}_s${srv}.DONE"; remote_out=$( [ "$drv" = mac ] && echo "$OUT" || echo "$SCRATCH/rb_${RUNTAG}" )
    cap=15; while IFS= read -r l; do [ -z "$l" ] && continue; e=$(field "$l" 7); b=$((e*BUDGET_MULT)); [ $b -lt $BUDGET_FLOOR_MIN ] && b=$BUDGET_FLOOR_MIN; cap=$((cap+b)); done <<< "${GROUP[$srv]}"
    n=0; dead=0; stall_said=0; tstart=$(date +%s)
    until box_run $drv "test -e '$remote_out/$sent'" 2>/dev/null; do
      n=$((n+1)); sleep "${WAIT_POLL_S:-60}"; elapsed=$(( ($(date +%s) - tstart) / 60 ))
      st=$(box_run $drv "hb=\$(cat '$remote_out/heartbeat_t${tier}_s${srv}' 2>/dev/null || echo 0); echo HB=\$(( \$(date +%s) - hb )); if stat -f %m . >/dev/null 2>&1; then nf=\$(find '$remote_out' -type f -not -name 'heartbeat_*' -exec stat -f %m {} + 2>/dev/null | sort -n | tail -1); else nf=\$(find '$remote_out' -type f -not -name 'heartbeat_*' -exec stat -c %Y {} + 2>/dev/null | sort -n | tail -1); fi; echo IDLE=\$(( \$(date +%s) - \${nf:-\$(date +%s)} )); tail -1 '$remote_out/chain_t${tier}_s${srv}.log' 2>/dev/null; true" 2>/dev/null)
      hb=$(printf '%s\n' "$st" | sed -n 's/^HB=//p'); idle=$(printf '%s\n' "$st" | sed -n 's/^IDLE=//p'); last=$(printf '%s\n' "$st" | tail -1 | cut -c1-120)
      if [ "${hb:-0}" -gt 180 ] && [ $n -gt 3 ]; then dead=$((dead+1)); [ $dead -ge 2 ] && { log "   DEAD: server .$srv chain heartbeat ${hb}s old, no sentinel (last: $last)"; break; }; else dead=0; fi
      if [ "${idle:-0}" -gt 2400 ] && [ $stall_said = 0 ]; then log "   STALL: server .$srv chain idle $((idle/60)) min (last: $last); the leg's own budget will end it"; stall_said=1; fi
      [ $((n % 15)) = 0 ] && log "   waiting on server .$srv chain (${elapsed}/${cap} min): $last"
      [ $elapsed -ge $cap ] && { log "   CAP: server .$srv chain exceeded its budget sum (${cap} min): $last"; break; }
    done
    rm -f "$DS4_CHAIN_REG/rb_${RUNTAG}_t${tier}_s${srv}.env"
    # a chain that died/capped: kill its whole process group (the chain is a session leader), never just the script
    box_run $drv "test -e '$remote_out/$sent'" 2>/dev/null || box_run $drv "for p in \$(pgrep -f 'chain_t${tier}_s${srv}.sh'); do kill -TERM -- -\$p 2>/dev/null || kill \$p 2>/dev/null; done; sleep 2; for p in \$(pgrep -f 'chain_t${tier}_s${srv}.sh'); do kill -KILL -- -\$p 2>/dev/null || kill -9 \$p 2>/dev/null; done; true" >/dev/null 2>&1
    # PROVENANCE (09-05): pull ONLY this chain's legs and chain files.  The driver's RD ($SCRATCH/rb_<run>) is shared
    # across runs, and the whole-dir pull re-imported another run's leg dirs into OUT, which RESUME then skipped as
    # "recorded PASS" (regate3: five tier-2 receipts were regate2's, twice).
    if [ "$drv" != mac ]; then
      while IFS= read -r l; do [ -z "$l" ] && continue; leg=$(field "$l" 3); mkdir -p "$OUT/$leg"; rsync -rt "$(mac_alias $drv):$remote_out/$leg/" "$OUT/$leg/" 2>/dev/null; done <<< "${GROUP[$srv]}"
      for f in "chain_t${tier}_s${srv}.log" "heartbeat_t${tier}_s${srv}" "$sent"; do rsync -t "$(mac_alias $drv):$remote_out/$f" "$OUT/" 2>/dev/null; done
    fi
    while IFS= read -r l; do [ -z "$l" ] && continue; leg=$(field "$l" 3)
      vl=$(grep -E "^VERDICT leg=$leg " "$OUT/$leg/leg.out" 2>/dev/null | tail -1); v=$(echo "$vl" | awk '{print $3}'); rc=$(echo "$vl" | awk '{print $4}' | sed 's/rc=//'); mins=$(echo "$vl" | awk '{print $5}' | sed 's/min//')
      [ -z "$v" ] && v="FAIL (no verdict: chain died/capped/timed out)"
      log "   $leg: ${v} rc=${rc:-?} ${mins:-?} min"; index_row $tier $leg $srv $drv "$(field "$l" 6)" "${v}" "${rc:-?}" "${mins:-?}" "$leg/leg.out"
      [ "${v:-FAIL}" = PASS ] || { overall=FAIL; [ "${STOP[$srv]}" = 1 ] && stop_now=1; }
    done <<< "${GROUP[$srv]}"
  done
  [ "${stop_now:-0}" = 1 ] && { log "STOP: a stop-tier leg failed in tier $tier"; break; }
done
log "RELEASE BATTERY: $overall (index $OUT/index.md)"
[ "$overall" = PASS ]
