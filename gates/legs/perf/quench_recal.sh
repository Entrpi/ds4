#!/usr/bin/env bash
# battery: quench_recal tier=3m stop=0 server=33 driver=33 trigger="model or drafter identity changed" est=60 seq=20
# verdict: QUENCH_RECAL_(PASS|FAIL)
# catches: a mis-calibrated DSpark quench guard (net-negative speculative decode) after a model or drafter change; accept parity does not exempt it (attended: release-process §3g)
# cmd: bash gates/legs/perf/quench_recal.sh
echo "SEE release-process §3g: dspark_trace_replay.py calibration per model identity (corpus drivers not in the tree); not automated here"; exit 1
