#!/usr/bin/env python3
"""Streaming client for the deep-ctx capacity gate: sends one chat request
(SSE), timestamps first/last content delta, prints TTFT (≈ admit+prefill
wall), decode rate, and the answer. Optionally writes the FULL result as JSON
(needed by memory/deep_ctx.sh to build the turn-2 request from the ACTUAL
stage-1 answer — cont temp-0 is not run-to-run deterministic, so a scripted
follow-up turn must never bake in a previous run's answer).

usage: sse_probe_client.py <req.json> <url> <tag> [out.json]
"""
import json, sys, time, urllib.request

req_path, url, tag = sys.argv[1], sys.argv[2], sys.argv[3]
out_json = sys.argv[4] if len(sys.argv) > 4 else None
body = json.load(open(req_path))
body["stream"] = True
body["stream_options"] = {"include_usage": True}
data = json.dumps(body).encode()

t0 = time.time()
t_first = None
n_deltas = 0
text = []
usage = None
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=7200) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                piece = ch.get("delta", {}).get("content") or ""
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    n_deltas += 1
                    text.append(piece)
except Exception as e:
    print(f"[{tag}] TRANSPORT FAIL after {time.time()-t0:.1f}s: {e}")
    sys.exit(1)

t_end = time.time()
ans = "".join(text)
ttft = (t_first - t0) if t_first else -1
dec = (t_end - t_first) if t_first else 0
rate = (n_deltas - 1) / dec if dec > 0 and n_deltas > 1 else 0
print(f"[{tag}] ttft={ttft:.1f}s decode={n_deltas} deltas in {dec:.1f}s "
      f"({rate:.2f} tok/s, {1000/rate if rate else 0:.1f} ms/tok) total={t_end-t0:.1f}s")
print(f"[{tag}] usage={usage}")
print(f"[{tag}] answer={ans[:300]!r}")
if out_json:
    with open(out_json, "w") as f:
        json.dump({"tag": tag, "text": ans, "usage": usage, "ttft_s": round(ttft, 1),
                   "decode_deltas": n_deltas, "decode_s": round(dec, 1)}, f)
