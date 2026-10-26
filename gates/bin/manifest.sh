#!/usr/bin/env bash
# gates/bin/manifest.sh — render gates/battery.manifest from the legs' header blocks (the legs are the
# source of truth; the rendered file is checked in so the whole queue reads in one place and the
# orchestrator stays a flat-file consumer).
#
# A battery member declares itself in its first 40 lines:
#   # battery: <leg-name> tier=<0|1|2|2b|3|3m> stop=<0|1> server=<33|15|mac> driver=<33|15|mac> trigger=<always|"free text"> est=<minutes> [seq=<order within the tier, default 50>]
#   # verdict: <regex matching the leg's OWN final line; VERDICT-LINE law: nothing looser>
#   # catches: <the failure class only this leg sees; one line, used by the README table>
#   # cmd: <optional command run on the driver in its tree; default "bash gates/legs/<sub>/<file>">
# Files without a "# battery:" line are standing gates (run by hand or by trigger), not members.
#
# Usage: bash gates/bin/manifest.sh            render to gates/battery.manifest
#        bash gates/bin/manifest.sh --check    exit 1 if the checked-in manifest is stale (make check-manifest)
#        bash gates/bin/manifest.sh --table    print the README table (leg | tier | boxes | est | catches)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); GATES=$(cd "$HERE/.." && pwd)
mode=${1:-render}
python3 - "$GATES" "$mode" <<'PY'
import os, re, shlex, sys
gates, mode = sys.argv[1], sys.argv[2]
legs = []
for root, _, files in os.walk(os.path.join(gates, "legs")):
    for f in sorted(files):
        if not f.endswith(".sh"): continue
        p = os.path.join(root, f); rel = os.path.relpath(p, os.path.dirname(gates))
        head = open(p, errors="replace").read().split("\n")[:40]
        bat = next((l for l in head if l.startswith("# battery:")), None)
        if not bat: continue
        toks = shlex.split(bat[len("# battery:"):])
        name, kv = toks[0], dict(t.split("=", 1) for t in toks[1:] if "=" in t)
        def hdr(k):
            l = next((l for l in head if l.startswith(f"# {k}:")), None)
            return l[len(k) + 3:].strip() if l else ""
        for k in ("tier", "stop", "server", "driver", "trigger", "est"):
            if k not in kv: sys.exit(f"{rel}: battery header lacks {k}=")
        verdict = hdr("verdict") or sys.exit(f"{rel}: no '# verdict:' line")
        cmd = hdr("cmd") or f"bash {rel}"
        m = re.match(r"(\d+)([a-z]*)", kv["tier"]); tkey = (int(m.group(1)), m.group(2))
        legs.append((tkey, int(kv.get("seq", 50)), name, kv, verdict, hdr("catches"), cmd, rel))
legs.sort(key=lambda x: (x[0], x[1], x[2]))
if mode == "--table":
    print("| Leg | Tier | Server | Driver | Trigger | Est (min) | What only it catches | File |"); print("|---|---|---|---|---|---|---|---|")
    for _, _, name, kv, _, catches, _, rel in legs:
        print(f"| {name} | {kv['tier']} | .{kv['server']} | .{kv['driver']} | {kv['trigger']} | {kv['est']} | {catches} | `{rel}` |")
    sys.exit(0)
out = ["# gates/battery.manifest — RENDERED by gates/bin/manifest.sh from the legs' '# battery:' headers; do not edit",
       "# (edit the leg's header, re-render, commit both).  Fields separated by ' @@ ':",
       "#   tier @@ stop @@ leg @@ server @@ driver @@ trigger @@ est_min @@ verdict_regex @@ command",
       "# command runs on the DRIVER box in its ds4 tree with RB_LEG RB_OUT RB_SERVER RB_TREE RB_GGUF RB_VBASE RB_ENC RB_RUN RB_NONCE RB_LOCK_DIR set.",
       "# Triggered legs run only when named in RUN_TRIGGERED (or ALL=1); otherwise index.md records SKIPPED (trigger: ...).",
       "# Tier 3m rows are attended pointers the orchestrator never launches."]
cur = None
for tkey, _, name, kv, verdict, catches, cmd, rel in legs:
    if kv["tier"] != cur: cur = kv["tier"]; out.append(f"# ---- tier {cur} ----")
    out.append(" @@ ".join([kv["tier"], kv["stop"], name, kv["server"], kv["driver"], kv["trigger"], kv["est"], verdict, cmd]))
text = "\n".join(out) + "\n"
target = os.path.join(gates, "battery.manifest")
if mode == "--check":
    old = open(target).read() if os.path.exists(target) else ""
    if old != text: print("gates/battery.manifest is STALE: run gates/bin/manifest.sh"); sys.exit(1)
    print(f"gates/battery.manifest up to date ({len(legs)} legs)"); sys.exit(0)
open(target, "w").write(text); print(f"rendered {target}: {len(legs)} legs")
PY
