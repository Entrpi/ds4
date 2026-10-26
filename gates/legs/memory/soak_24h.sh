#!/usr/bin/env bash
# battery: soak_24h tier=3 stop=0 server=33 driver=15 trigger="memgov/census/ledger code changed or a major" est=1440 seq=20
# verdict: MEMGOV SOAK: (ALL ASSERTS PASS|FAILED)
# catches: slow leaks, governor faults and torn fallbacks that only appear with age (48 x 30-min cycles, pulses every 8th, an observe-mode rollback slice); the aged-server drift signature
# cmd: bash gates/legs/memory/soak_24h.sh
HOURS=24 CYCLE_MIN=30 PULSE_EVERY=8 POLL_CAP_MIN=1500 VERDICT_TAG=SOAK_24H exec bash "$(dirname "$0")/soak_drive.sh"
