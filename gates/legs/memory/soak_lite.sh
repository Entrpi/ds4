#!/usr/bin/env bash
# battery: soak_lite tier=3 stop=0 server=33 driver=15 trigger=always est=60 seq=10
# verdict: MEMGOV SOAK: (ALL ASSERTS PASS|FAILED)
# catches: the four-cycle shape in ~1 h (4 x 8 min; pulse cycles run long): the first saturation pulse against the GRANTED pool (the boot bank fit as it really funded), the warm replay, the observe-mode rollback slice, epoch advance and rejects flat on quiet cycles; the leg that found the vision-reserve double count and the promise-blind hog target
# cmd: bash gates/legs/memory/soak_lite.sh
# 09-05: CYCLES=4 CYCLE_MIN=8 (was HOURS=2 CYCLE_MIN=30): the longest timer the pulse depends on is the 300 s continuation TTL;
# 22 idle minutes per cycle measured nothing at this age (the 24 h soak keeps 30-min cycles as the drift instrument).
CYCLES=4 CYCLE_MIN=8 PULSE_EVERY=2 VERDICT_TAG=SOAK_LITE exec bash "$(dirname "$0")/soak_drive.sh"
