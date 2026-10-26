#!/usr/bin/env bash
# battery: evals tier=3m stop=0 server=15 driver=15 trigger="checkpoint or quantization changed" est=420 seq=10
# verdict: RUN_DONE_evals errors=0
# catches: quality regressions from numerics or quantization changes: 2,248 frozen items (MMLU, GSM8K, HumanEval, MBPP, IFEval, needles) on the synced binary; the only leg with a quality oracle (attended: release-process §3f)
# cmd: bash gates/legs/quality/evals.sh
echo "SEE release-process §3f: the attended eval rig procedure (the frozen suite lives outside the tree); not automated here"; exit 1
