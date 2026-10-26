# gates/lib/boot.sh — the shared boot / wait / kill helpers for gates that drive a ds4-server on a GB10
# over ssh (the trio every older gate carries as its own copy; new legs use these, old ones migrate
# when touched).  Sources env.sh + verdict.sh.
#
#   SRV_HOST   ssh name of the server box as seen from where the gate runs (a battery leg gets it
#              as $RB_SERVER; a hand run from the Mac defaults to $DS4_BOX33)
#   SRV_TREE   the ds4 tree on the server box ($RB_TREE / $DS4_BOX_TREE)
#   SRV_LOG    the server's log on the box (default /tmp/ds4_gate_srv.log)
#   SRV_PORT   (default 8000)
#
#   srv_boot "ARGS" ["ENV=.. ENV2=.."]   detached boot on the box (setsid, all three fds off the ssh);
#                                        returns after the ready line or fails after SRV_BOOT_TIMEOUT s
#   srv_wait_ready [SECONDS]             poll the log for the READY line (batch ctx ready / listening)
#   srv_kill                             kill every ds4-server on the box (TERM, then KILL), by pid
#   srv_count                            number of ds4-server processes on the box
#   srv_grep PATTERN / srv_count_of PATTERN   first match / count in the server log
#   srv_metric NAME                      one /metrics gauge from the box (curl on the box side)
#   box_lock_assert NONCE                take or re-assert the box lock (fails when held by someone else)
#   box_lock_release                     remove the lock dir
. "$(dirname "${BASH_SOURCE[0]}")/verdict.sh"
SRV_HOST=${SRV_HOST:-${RB_SERVER:-${DS4_BOX33:-}}}
SRV_TREE=${SRV_TREE:-${RB_TREE:-${DS4_BOX_TREE:-}}}
SRV_LOG=${SRV_LOG:-/tmp/ds4_gate_srv.log}
SRV_PORT=${SRV_PORT:-8000}
SRV_BOOT_TIMEOUT=${SRV_BOOT_TIMEOUT:-900}
SRV_READY_RE=${SRV_READY_RE:-'persistent batch ctx ready|listening on|server ready'}
_ssh(){ ssh -o BatchMode=yes -o ConnectTimeout=15 "$SRV_HOST" "$@"; }
srv_count(){ _ssh "pgrep -x ds4-server | wc -l | tr -d ' '; true" 2>/dev/null | tail -1; }
srv_kill(){ _ssh "for p in \$(pgrep -x ds4-server); do kill \$p; done; sleep 2; for p in \$(pgrep -x ds4-server); do kill -9 \$p; done; exit 0" 2>/dev/null; }
srv_grep(){ _ssh "grep -m1 -E \"$1\" $SRV_LOG 2>/dev/null; true" 2>/dev/null; }
srv_count_of(){ local c; c=$(_ssh "grep -cE \"$1\" $SRV_LOG 2>/dev/null; true" 2>/dev/null | tail -1); echo "${c:-0}"; }
srv_wait_ready(){ local n=0 max=${1:-$SRV_BOOT_TIMEOUT}
  while [ $n -lt "$max" ]; do
    if _ssh "grep -qE '$SRV_READY_RE' $SRV_LOG 2>/dev/null" 2>/dev/null; then return 0; fi
    [ "$(srv_count)" = 0 ] && [ $n -gt 5 ] && { log "server died during boot: $(_ssh "tail -3 $SRV_LOG" 2>/dev/null)"; return 1; }
    sleep 5; n=$((n+5)); done
  log "server not ready after ${max}s"; return 1; }
srv_boot(){ local args=$1 envs=${2:-}
  [ -n "$SRV_HOST" ] && [ -n "$SRV_TREE" ] || { echo "srv_boot: SRV_HOST/SRV_TREE unset (rig keys DS4_BOX33 / DS4_BOX_TREE)" >&2; return 2; }
  srv_kill
  _ssh "cd $SRV_TREE && : > $SRV_LOG && setsid --fork bash -c 'env $envs ./ds4-server $args > $SRV_LOG 2>&1 < /dev/null' > /dev/null 2>&1 < /dev/null; echo booted" >/dev/null 2>&1
  srv_wait_ready; }
srv_metric(){ _ssh "curl -s http://127.0.0.1:$SRV_PORT/metrics | grep -E '^$1( |\{)' | tail -1 | awk '{print \$NF}'; true" 2>/dev/null | tail -1; }
box_lock_assert(){ local nonce=$1
  _ssh "n=\$(cat $DS4_BOX_LOCK_DIR/nonce 2>/dev/null); case \"\$n\" in \"\"|\"$nonce\") mkdir -p $DS4_BOX_LOCK_DIR && echo \"$nonce\" > $DS4_BOX_LOCK_DIR/nonce; exit 0;; *) echo \"LOCK-HELD \$n\"; exit 3;; esac" 2>/dev/null; }
box_lock_release(){ _ssh "rm -rf $DS4_BOX_LOCK_DIR; exit 0" 2>/dev/null; }
