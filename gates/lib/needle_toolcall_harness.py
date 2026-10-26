#!/usr/bin/env python3
"""Needle-at-depth tool-call behavioral harness (field issue #18 gate).

One growing scripted conversation per seed, mirroring the issue-#18 capture:
a single bash tool, empty-content assistant tool-call turns, tool-result
turns growing to ~112K prompt tokens. At each depth checkpoint the harness
sends a completion trigger and records whether the model reports completion
via a tool call or as plain text. The capture-faithful trigger ("tool") ends
the conversation on a successful TOOL RESULT with no user cue -- the model
must decide completion itself, with only the position-0 system rule pinning
the format; this is the shape that reproduced the field flip.

usage: needle_toolcall_harness.py PORT ARM_LABEL EFFORT SEEDS OUT_JSONL [TEMP] [K] [TRIG]
  EFFORT: "high" -> send reasoning_effort:"high"; "omit" -> field absent
  TEMP:   "greedy" (temperature 0, default) | "field" (omit temperature ->
          server default sampling, matching the capture's requests)
  K:      draws per checkpoint (default 1; >1 only makes sense with "field")
  TRIG:   "user" (explicit user finish-up message, default) | "tool"
          (capture-faithful; use this for the gate)
Depths override: DS4AB_DEPTHS=comma-separated target prompt-token depths.
"""
import json, sys, time, random, urllib.request

PORT, ARM, EFFORT, SEEDS_S, OUT = sys.argv[1:6]
TEMP = sys.argv[6] if len(sys.argv) > 6 else "greedy"
K = int(sys.argv[7]) if len(sys.argv) > 7 else 1
TRIG = sys.argv[8] if len(sys.argv) > 8 else "user"
SEEDS = [int(x) for x in SEEDS_S.split(",")]
URL = "http://127.0.0.1:%s/v1/chat/completions" % PORT

CHARS_PER_TOK = 2.70          # measured: this filler tokenizes at ~2.7 chars/token
import os
DEPTHS = ([int(x) for x in os.environ["DS4AB_DEPTHS"].split(",")]
          if os.environ.get("DS4AB_DEPTHS") else
          ([8000, 64000, 96000, 112000] if TEMP == "field" else
           [2000, 8000, 32000, 64000, 96000, 112000]))   # target prompt tokens

SYSTEM = (
    "You are a software engineering agent working in a repository. "
    "All actions run through the bash tool.\n"
    "SYSTEM RULE (CRITICAL): when the task is fully complete, you MUST report "
    "completion by calling the bash tool with the exact command: submit\n"
    "Never announce completion in plain text. The harness only accepts tool "
    "calls; a plain-text completion report is a task failure."
)
TOOLS = [{"type": "function", "function": {
    "name": "bash",
    "description": "Run a shell command in the repository and return its output.",
    "parameters": {"type": "object",
                   "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}]
TRIGGER = ("All the requested changes are made and every test passes. "
           "The task is now fully complete. Finish up.")
WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
         "nu xi omicron pi rho sigma tau upsilon phi chi psi omega vector "
         "matrix tensor kernel buffer index stride cache token layer weight "
         "scale block bank row column merge probe gate fold").split()


def filler(seed, turn, nbytes):
    rng = random.Random(seed * 100003 + turn)
    parts, total = [], 0
    while total < nbytes:
        s = "tests/test_%s_%d.py::test_%s_%d PASSED [%d ms]" % (
            rng.choice(WORDS), rng.randint(0, 999),
            rng.choice(WORDS), rng.randint(0, 9999), rng.randint(1, 999))
        parts.append(s)
        total += len(s) + 1
    return "\n".join(parts)


def post(messages):
    body = {"model": "ds4", "messages": messages, "tools": TOOLS,
            "max_tokens": 3000, "stream": False}
    if TEMP == "greedy":
        body["temperature"] = 0
    if EFFORT == "high":
        body["reasoning_effort"] = "high"
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        resp = json.loads(r.read().decode())
    return resp, time.time() - t0


def classify(resp):
    ch = resp["choices"][0]
    msg = ch.get("message", {})
    fin = ch.get("finish_reason")
    tcs = msg.get("tool_calls") or []
    content = msg.get("content") or ""
    rlen = sum(len(msg.get(k) or "") for k in
               ("reasoning", "reasoning_content", "thinking"))
    if tcs:
        f = tcs[0].get("function", {})
        try:
            cmd = json.loads(f.get("arguments") or "{}").get("command", "")
        except Exception:
            cmd = f.get("arguments", "")[:80]
        out = "tool_call"
        det = "%s(%s)" % (f.get("name"), cmd[:60])
    elif fin == "length" and not content.strip():
        out, det = "length_no_tool", ""
    elif content.strip():
        out, det = "text", ""
    else:
        out, det = "empty", ""
    return {"outcome": out, "detail": det, "finish_reason": fin,
            "reasoning_len": rlen, "content_head": content[:300],
            "content_full": None if tcs else content,
            "reasoning_full": None if tcs else "".join(
                msg.get(k) or "" for k in ("reasoning", "reasoning_content", "thinking")),
            "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (resp.get("usage") or {}).get("completion_tokens")}


def run_seed(seed, outf):
    msgs = [{"role": "system", "content": SYSTEM}]
    chars = len(SYSTEM)
    turn = 0
    for depth in DEPTHS:
        target = depth * CHARS_PER_TOK
        while chars < target:
            turn += 1
            cid = "call_%d" % turn
            cmd = "pytest -q tests/test_%s_%d.py" % (
                WORDS[(seed * 7 + turn) % len(WORDS)], turn)
            fill = filler(seed, turn, 3400)
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": cid, "type": "function",
                                         "function": {"name": "bash",
                                                      "arguments": json.dumps({"command": cmd})}}]})
            msgs.append({"role": "tool", "tool_call_id": cid, "content": fill})
            chars += len(cmd) + len(fill) + 40
        if TRIG == "tool":
            tid = "call_final_%d" % depth
            probe = msgs + [
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": tid, "type": "function",
                                 "function": {"name": "bash",
                                              "arguments": json.dumps({"command":
                                                  "python3 -m py_compile $(git ls-files '*.py') && pytest -q"})}}]},
                {"role": "tool", "tool_call_id": tid,
                 "content": "compile ok\n....... 47 passed, 0 failed in 12.41s\n"
                            "All requested changes verified. Nothing left to do."}]
        else:
            probe = msgs + [{"role": "user", "content": TRIGGER}]
        for draw in range(K):
            rec = {"arm": ARM, "seed": seed, "depth_target": depth,
                   "draw": draw, "n_messages": len(probe), "chars": chars}
            try:
                resp, dt = post(probe)
                rec.update(classify(resp))
                rec["latency_s"] = round(dt, 1)
                # FP-flake law (release-process §8 EOS-flake class): cont
                # temp-0 is NOT run-deterministic -- 8 greedy draws at one
                # byte-identical 96K state produced 8 different tool calls
                # (repro 2026-08-27), so a single greedy non-tool draw at a
                # near-tie is the known FP-noise class, not a verdict.  One
                # retry is the law; the first draw is disclosed in the
                # record so a deterministic failure (retry also non-tool)
                # still fails the gate.
                if TEMP == "greedy" and rec.get("outcome") != "tool_call":
                    first = {"outcome": rec.get("outcome"),
                             "detail": rec.get("detail")}
                    resp2, dt2 = post(probe)
                    rec.update(classify(resp2))
                    rec["latency_s"] = round(dt + dt2, 1)
                    rec["flake_retry_first_draw"] = first
            except Exception as e:
                rec.update({"outcome": "error", "detail": repr(e)[:300]})
            outf.write(json.dumps(rec) + "\n")
            outf.flush()
            print("  seed=%d depth=%d draw=%d -> %s %s (pt=%s, %.0fs)" % (
                seed, depth, draw, rec.get("outcome"), rec.get("detail", ""),
                rec.get("prompt_tokens"), rec.get("latency_s", -1)), flush=True)


def main():
    with open(OUT, "a") as outf:
        for seed in SEEDS:
            print("== %s seed %d ==" % (ARM, seed), flush=True)
            run_seed(seed, outf)
    print("__HARNESS_DONE_%s__" % ARM, flush=True)


main()
