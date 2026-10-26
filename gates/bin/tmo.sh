#!/usr/bin/env bash
# gates/bin/tmo.sh SECONDS CMD... — run CMD under a time budget and kill its WHOLE process tree on expiry
# (exit 124).  GNU timeout where present (it signals the child's process group), gtimeout on a Homebrew Mac,
# else a perl fallback that puts the child in its own process group and signals that group only (never the
# caller's: the 2026-09-04 functional test showed a plain `perl alarm` taking the calling chain down with it).
set -u
secs=$1; shift
if command -v timeout >/dev/null 2>&1; then exec timeout -k 60 "$secs" "$@"; fi
if command -v gtimeout >/dev/null 2>&1; then exec gtimeout -k 60 "$secs" "$@"; fi
exec perl -MPOSIX -e '
  my $t = shift @ARGV; my $p = fork();
  if (!$p) { POSIX::setpgid(0, 0); exec @ARGV or exit 127; }
  $SIG{ALRM} = sub { kill "TERM", -$p; sleep 2; kill "KILL", -$p; waitpid($p, 0); exit 124; };
  alarm $t; waitpid($p, 0); my $rc = $? >> 8; $rc = 128 + ($? & 127) if ($? & 127); exit $rc;
' "$secs" "$@"
