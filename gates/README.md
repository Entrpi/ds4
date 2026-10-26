# gates/ — the ds4 gate battery

Everything here proves a property of a built ds4 against a live rig: the unit suite is in
`tests/`, the speed benchmarks and their history in `speed-bench/` (upstream's layout), and
this directory holds the **gates**: scripts that boot a server or a CLI on a GPU box, drive it,
and print one verdict line. The release battery is a curated subset of them, run in tiers on
two boxes by the orchestrator.

```
gates/
  README.md              this file
  battery.manifest       the release queue, RENDERED from the legs' headers (bin/manifest.sh); checked in
  battery.env.example    the rig file template; the real one is local/battery.env (gitignored)
  bin/                   tooling: release_battery.sh (orchestrator), leg.sh (one ad-hoc leg through the
                         same infra), chain_watch.sh (per-event watchdog), rearm_watchers.sh (after a
                         restart), tmo.sh (process-tree time budget), manifest.sh (render / --check)
  lib/                   env.sh (loads the rig file), verdict.sh (the verdict-line contract),
                         boot.sh (shared boot/wait/kill over ssh), needle_prompt.py,
                         sse_probe_client.py, needle_toolcall_harness.py (shared drivers)
  legs/
    build/               cpu, metal, cuda quiet builds; cross-box goldens
    memory/              the governor, banks, admission, ledger, KV capacity, soaks
    serving/             API surfaces, lanes, protocol, tool calls
    vision/              the Vision-Exp members (identity, needles, cli, agent, api, seed) + helpers
    agent/               tool-eval-bench legs
    perf/                tripwire, ABBA, stamps, profiles
    quality/             the eval-rig pointer
```

## Running a gate by hand

Every leg sources `lib/env.sh`, which loads `local/battery.env` (copy `battery.env.example`,
fill in your hosts, trees and model paths). A gate that needs a key you left empty fails by
name at that line. Knobs keep their historical names (`R`, `BINDIR`, `GGUF`, `OUT`, `CTX`...)
and default from the rig keys, so a hand run is:

```
R=$DS4_BOX33 OUT=/tmp/x bash gates/legs/memory/defaults_derivation.sh
```

Most gates are driven **from the workstation over ssh** (they boot the server on the box).
Box-side legs (the vision members, the build legs) run in the box's mirror of this tree; the
mirror needs its own `local/battery.env`. Neither GB10 can ssh itself and workstation aliases
do not resolve on the boxes, so a leg that ssh-es its server runs on the OTHER box using the
`_LAN` names (the BOX-SIDE DRIVER law).

## The release battery

`make battery` (or `OUT=local/docs/<arc>/battery-<date> bash gates/bin/release_battery.sh`)
runs `battery.manifest` tier by tier on both boxes at once. Tiers are ordered by catch rate per
minute and a failed `stop=1` leg ends the run before the next tier spends a GPU minute:

| Tier | Members | Stop on FAIL |
|---|---|---|
| 0 | quiet builds, three shapes (`make cpu` FIRST: it is the leg that catches backend-neutral symbol traps) | yes |
| 1 | units + goldens (.33), the vision members (.33); cross-box goldens, 12k twins, deep 240k (.15) | yes |
| 2 | tool-slip depth (one shipped-defaults run), teb, perf tripwire; memory-truth fast set and deep admits by trigger | no |
| 2b | real-budgets gate, driven from the workstation (it syncs and rebuilds the box) | no |
| 3 | the ~1 h soak-lite (4 x 8 min cycles; pulse cycles run long); the 24 h soak and the reminder-off depth gate by trigger | no |
| 3m | attended pointers (evals, quench recal), never launched | |

Laws the tooling enforces (release-process §3.0 has the reasoning):

- **VERDICT-LINE**: a leg prints exactly one final line that its `# verdict:` regex names
  exactly; the chain records `VERDICT leg=<name> PASS|FAIL rc=<n> <line>` and never re-derives
  a verdict. A leg with no matching line is a FAIL row, never a blank.
- **TRIGGER**: legs whose trigger is not `always` run only when named in `RUN_TRIGGERED` (or
  `ALL=1`); a skipped leg is written into `index.md` as `SKIPPED (trigger: ...)`.
- **BOX-SIDE DRIVER**: within a tier, one driver box per server box; the driver is the other
  box for legs that ssh their server.
- **WATCHDOG**: every leg runs under `tmo.sh` with a budget of 3x its estimate (floor 10 min);
  every chain keeps a 30 s heartbeat; the orchestrator reports STALL / DEAD / CAP and kills a
  dead chain's whole process group. Chains register in `$DS4_CHAIN_REG`; `bin/chain_watch.sh`
  turns a chain into per-event lines for a monitor, and `bin/rearm_watchers.sh` re-arms them
  after a restart.
- **LOCK**: the orchestrator takes the box-ownership nonce (`$DS4_BOX_LOCK_DIR/nonce`) on the
  boxes it uses, accepts a nonce the session already holds (`LOCK_OWNER=`), re-asserts it after
  every leg, and never removes one it did not take.
- **SELF-COPY**: the orchestrator runs from a copy under `$OUT`; never edit a running bash
  script in place.

`bin/leg.sh --server 33 --driver 15 -- <command>` runs ONE ad-hoc command through the same
budget / heartbeat / sentinel / verdict / registry machinery.

## Adding a leg

1. Put the script under `legs/<subsystem>/<behavior>.sh`; source `lib/verdict.sh` (or
   `lib/boot.sh` when it boots a server) and end with `pass`/`fail`/`finish`, or print your own
   final line.
2. To make it a battery member, add the header block in its first 40 lines:
   ```
   # battery: <leg-name> tier=<0|1|2|2b|3|3m> stop=<0|1> server=<33|15|mac> driver=<33|15|mac> trigger=<always|"why"> est=<min> [seq=<order>]
   # verdict: <regex naming the leg's OWN final line>
   # catches: <the failure class only this leg sees>
   # cmd: <optional; default "bash gates/legs/<sub>/<file>"; the leg sees RB_LEG RB_OUT RB_SERVER RB_TREE RB_GGUF RB_VBASE RB_ENC RB_RUN RB_NONCE RB_LOCK_DIR>
   ```
3. `bash gates/bin/manifest.sh` re-renders `battery.manifest`; `make check-manifest` fails when
   the rendered file is stale. `bash gates/bin/manifest.sh --table` prints the queue with the
   `catches:` column for the docs.
4. Rig facts (hosts, trees, model paths) go into `battery.env.example` as a new `DS4_*` key with
   a placeholder, never into the script.
