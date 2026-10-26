#!/usr/bin/env bash
# gates/bin/leg.sh — run ONE ad-hoc leg through the release-battery infra (budget, heartbeat, sentinel,
# verdict row, registry entry, watchdog-able), without writing a manifest by hand.
#   bash gates/bin/leg.sh --server 33 --driver 15 [--name NAME] [--est MIN] [--regex 'PASS|FAIL'] \
#        [--lock-owner NONCE] [--out DIR] -- <command run on the driver in its ds4 tree>
# The command sees RB_LEG RB_OUT RB_SERVER RB_TREE RB_GGUF RB_VBASE RB_ENC RB_NONCE like any manifest leg.
# --lock-owner: a box-lock nonce THIS session already holds (accepted, asserted, never removed).
# The verdict regex names the driver's exact final line (VERDICT REGEX LAW); default: a line starting with
# PASS/FAIL or ending in _PASS/_FAIL.  Exit status = the battery's (0 = PASS).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); . "$HERE/../lib/env.sh"; REPO=$DS4_REPO
server=""; driver=""; name=""; est=10; regex='(^|[ _])(PASS|FAIL)([ :_]|$)'; owner=""; out=""
while [ $# -gt 0 ]; do case $1 in
  --server) server=$2; shift 2;; --driver) driver=$2; shift 2;; --name) name=$2; shift 2;; --est) est=$2; shift 2;;
  --regex) regex=$2; shift 2;; --lock-owner) owner=$2; shift 2;; --out) out=$2; shift 2;; --) shift; break;;
  *) echo "leg.sh: unknown option $1"; exit 2;; esac; done
[ -n "$server" ] && [ -n "$driver" ] && [ $# -gt 0 ] || { sed -n '2,9p' "$0"; exit 2; }
name=${name:-adhoc_$(date +%H%M%S)}; out=${out:-$REPO/local/docs/adhoc/${name}_$(date +%Y%m%d_%H%M)}
mkdir -p "$out"; mf="$out/leg.manifest"
printf '0 @@ 0 @@ %s @@ %s @@ %s @@ always @@ %s @@ %s @@ %s\n' "$name" "$server" "$driver" "$est" "$regex" "$*" > "$mf"
echo "leg.sh: $name  server .$server  driver .$driver  est ${est} min  out $out"
MANIFEST="$mf" TIERS=0 OUT="$out" LOCK_OWNER="$owner" bash "$HERE/release_battery.sh"
