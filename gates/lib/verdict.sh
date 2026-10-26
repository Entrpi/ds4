# gates/lib/verdict.sh — the one-line verdict contract of a battery leg (release-process §3.0, VERDICT-LINE law).
#
# A leg prints exactly one final line the orchestrator matches with the leg's `# verdict:` regex:
#     [HH:MM:SS] <TAG>_PASS <detail>      exit 0
#     [HH:MM:SS] <TAG>_FAIL <detail>      exit 1
# Nothing looser: a bare "FAIL" alternative once matched an EXPECTED mid-leg "TRANSPORT FAIL 503".
# The regex for a leg using these helpers is  ^\[[0-9:]+\] <TAG>_(PASS|FAIL)
#
#   VERDICT_TAG=GOLDENS_XBOX      set before use (default: the leg file's stem, upper-cased)
#   log MSG                       timestamped line to stdout (and $VERDICT_LOG when set)
#   pass DETAIL / fail DETAIL     print the verdict line and exit; fail runs $VERDICT_CLEANUP first when set
#   count_fail                    increments a failure counter; finish DETAIL turns it into pass/fail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
VERDICT_TAG=${VERDICT_TAG:-$(basename "${0%.sh}" | tr '[:lower:]' '[:upper:]')}
VERDICT_LOG=${VERDICT_LOG:-}
_vfails=0
log(){ local l; l="[$(date +%H:%M:%S)] $*"; echo "$l"; [ -n "$VERDICT_LOG" ] && echo "$l" >> "$VERDICT_LOG"; return 0; }
pass(){ log "${VERDICT_TAG}_PASS $*"; exit 0; }
fail(){ [ -n "${VERDICT_CLEANUP:-}" ] && eval "$VERDICT_CLEANUP"; log "${VERDICT_TAG}_FAIL $*"; exit 1; }
count_fail(){ _vfails=$((_vfails+1)); log "FAIL: $*"; }
finish(){ if [ "$_vfails" = 0 ]; then pass "$*"; else fail "$_vfails failure(s); $*"; fi; }
