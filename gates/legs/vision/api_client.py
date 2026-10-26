#!/usr/bin/env python3
"""Track B inc 7 API gate client (stdlib only; runs ON .33 beside ds4-server).
Commands: identity NAME | surfaces | repeat | refusals | concurrency NAME KIND | needles SURFACE
Every command writes RD/<name>.jsonl (or .json) and prints one summary line."""
import sys, os, json, base64, time, threading, urllib.request, urllib.error

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8090")
RD = os.environ.get("RD", ".")
IMG = os.environ.get("IMG", ".")
NEEDLE = os.environ.get("NEEDLE", ".")
MODEL = "deepseek-v4-flash"
AFTER = "Describe this image in one sentence."
AFTER_JPG = "Describe this photograph in one sentence."   # distinct prompt length for the seed compare
TEXT_PROMPTS = [
    "Explain how a hash table resolves collisions, in two sentences.",
    "Name three prime numbers and say why they are prime.",
]
IMAGES = ["tiny_16x16.png", "odd_17x9.png", "typical_532x280.png", "typical_532x280.jpg",
          "portrait_600x900.png", "checker_640x480.png", "wide_8to1_2000x250.png", "large_2400x1800.png"]
CONC_TEXT = [
    "What is the capital of France? Answer in one sentence.",
    "Name three primary colors, comma separated.",
    "Write the word hello backwards.",
    "How many legs does a spider have? One sentence.",
    "Give one synonym for quick.",
    "What year did the first moon landing happen? One sentence.",
    "Spell the word rhythm letter by letter.",
]

def data_uri(path):
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    b = base64.b64encode(open(path, "rb").read()).decode()
    return mime, b, f"data:{mime};base64,{b}"

def post(path, body, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode()), time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try: j = json.loads(raw)
        except Exception: j = {"raw": raw}
        return e.code, j, time.time() - t0

def post_stream(path, body, timeout=600):
    """Streaming chat: collect delta text, token ids and usage from the SSE chunks."""
    body = dict(body); body["stream"] = True; body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode(errors="replace"); st = r.status
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try: j = json.loads(raw)
        except Exception: j = {"raw": raw}
        return e.code, j, time.time() - t0
    text, ids, usage = [], [], None
    for line in raw.split("\n"):
        if not line.startswith("data: ") or line.strip() == "data: [DONE]": continue
        try: c = json.loads(line[6:])
        except Exception: continue
        if c.get("usage"): usage = c["usage"]
        for ch in c.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content"): text.append(d["content"])
            if ch.get("token_ids"): ids.extend(ch["token_ids"])
    j = {"choices": [{"message": {"content": "".join(text)}, "token_ids": ids}], "usage": usage or {}}
    return st, j, time.time() - t0

def rejects():
    """ds4_cont_admit_rejects_total from /metrics (-1 when unreadable).  A request whose delta is > 0 was
    REFUSED by the continuous lane (memory floor) and served on the serial lane instead: the twin and
    surfaces checks compare within a lane only (batched-vs-serial divergence is a known FP class; the
    09-04 re-gate's "surfaces: DIFF" was exactly that after floor refusals at bank 21)."""
    try:
        with urllib.request.urlopen(BASE + "/metrics", timeout=20) as r:
            for line in r.read().decode(errors="replace").split("\n"):
                if line.startswith("ds4_cont_admit_rejects_total "): return int(float(line.split()[1]))
    except Exception:
        pass
    return -1

def lane_probe(fn, *a):
    """Run a request; return (status, json, seconds, fallback) with fallback=True when the cont lane refused it.
    The counter is GLOBAL, so this attribution is exact only while requests run one at a time (the identity,
    surfaces and repeat phases do; the concurrency phase does not use it)."""
    r0 = rejects(); st, j, dt = fn(*a); r1 = rejects()
    return st, j, dt, (r0 >= 0 and r1 > r0)

def call(path, body, lane):
    """lane: 'serial' = non-streaming + return_token_ids (the server routes that
    shape to the serial lane); 'batched' = streaming + return_token_ids (the
    continuous lane); 'plain' = non-streaming, no ids (continuous lane)."""
    if lane == "batched": return post_stream(path, body)
    return post(path, body)

def walk(o, key):
    """First value of `key` anywhere in a JSON object."""
    if isinstance(o, dict):
        if key in o: return o[key]
        for v in o.values():
            r = walk(v, key)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = walk(v, key)
            if r is not None: return r
    return None

def out_text(surface, j):
    if surface == "chat":
        return (j.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if surface == "anthropic":
        return "".join(c.get("text", "") for c in j.get("content", []) if c.get("type") == "text")
    if surface == "responses":
        parts = []
        for item in j.get("output", []):
            for c in item.get("content", []) if isinstance(item, dict) else []:
                if c.get("type") == "output_text": parts.append(c.get("text", ""))
        return "".join(parts)
    return ""

def build(surface, images, before, after, max_tokens, token_ids=False):
    """One user turn: before [image...] after (images = list of paths)."""
    if surface == "chat":
        content = [{"type": "text", "text": before}] if before or not images else []
        if not images:
            content = before + after
        else:
            for p in images:
                content.append({"type": "image_url", "image_url": {"url": data_uri(p)[2]}})
            content.append({"type": "text", "text": after})
        body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens, "temperature": 0, "thinking": {"type": "disabled"}}
        if token_ids: body["return_token_ids"] = True
        return "/v1/chat/completions", body
    if surface == "anthropic":
        content = [{"type": "text", "text": before}] if before or not images else []
        if not images:
            content = [{"type": "text", "text": before + after}]
        else:
            for p in images:
                mime, b, _ = data_uri(p)
                content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b}})
            content.append({"type": "text", "text": after})
        body = {"model": MODEL, "max_tokens": max_tokens, "temperature": 0,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": content}]}
        return "/v1/messages", body
    if surface == "responses":
        content = [{"type": "input_text", "text": before}] if before or not images else []
        if not images:
            content = [{"type": "input_text", "text": before + after}]
        else:
            for p in images:
                content.append({"type": "input_image", "image_url": data_uri(p)[2], "detail": "auto"})
            content.append({"type": "input_text", "text": after})
        body = {"model": MODEL, "max_output_tokens": max_tokens, "temperature": 0,
                "reasoning": {"effort": "none"},   # the Responses surface's thinking control
                "input": [{"type": "message", "role": "user", "content": content}]}
        return "/v1/responses", body
    raise SystemExit("bad surface")

def identity_prompts():
    ps = [("-", "", t) for t in TEXT_PROMPTS]
    for f in IMAGES:
        ps.append((os.path.join(IMG, f), "", AFTER_JPG if f.endswith(".jpg") else AFTER))
    return ps

def cmd_identity(name, lane="serial", ntok=64):
    n_ok = 0
    with open(os.path.join(RD, name + ".jsonl"), "w") as out:
        for image, before, after in identity_prompts():
            imgs = [] if image == "-" else [image]
            path, body = build("chat", imgs, before, after, ntok, token_ids=True)
            st, j, dt, fb = lane_probe(call, path, body, lane)
            rec = {"image": image, "before": before, "after": after, "status": st, "sec": round(dt, 2),
                   "prompt_tokens": walk(j, "prompt_tokens"), "completion_tokens": walk(j, "completion_tokens"),
                   "token_ids": walk(j, "token_ids"), "text": out_text("chat", j) if st == 200 else json.dumps(j)[:300],
                   "fallback": fb}
            out.write(json.dumps(rec) + "\n"); out.flush()
            n_ok += st == 200
    print(f"identity {name} ({lane}): {n_ok}/{len(identity_prompts())} ok")

def cmd_surfaces(ntok=48):
    img = os.path.join(IMG, "typical_532x280.png")
    res = {}
    for s in ("chat", "anthropic", "responses"):
        path, body = build(s, [img], "Look carefully. ", AFTER, ntok)
        st, j, dt, fb = lane_probe(post, path, body)
        res[s] = {"status": st, "sec": round(dt, 2), "text": out_text(s, j) if st == 200 else json.dumps(j)[:300],
                  "prompt_tokens": walk(j, "input_tokens") if s == "anthropic" else walk(j, "prompt_tokens"),
                  "fallback": fb}
    ok = all(res[s]["status"] == 200 for s in ("chat", "anthropic", "responses"))
    lanes = {res[s]["fallback"] for s in ("chat", "anthropic", "responses")}
    same_lane = {fb: {res[s]["text"] for s in ("chat", "anthropic", "responses") if res[s]["fallback"] == fb} for fb in lanes}
    ident = ok and all(len(t) == 1 for t in same_lane.values())
    res["verdict"] = ("IDENTICAL" if len(lanes) == 1 else "IDENTICAL-WITHIN-LANE (lane-mixed: %s)" % ",".join(
        s for s in ("chat", "anthropic", "responses") if res[s]["fallback"])) if ident else "DIFF"
    json.dump(res, open(os.path.join(RD, "surfaces.json"), "w"), indent=1)
    print(f"surfaces: {res['verdict']} " + " ".join(f"{s}={res[s]['status']}{'/serial' if res[s]['fallback'] else ''}" for s in ("chat", "anthropic", "responses")))

def cmd_repeat(lane="batched", ntok=48):
    img = os.path.join(IMG, "checker_640x480.png")
    recs = []
    for i in range(2):
        path, body = build("chat", [img], "Look carefully. ", AFTER, ntok, token_ids=True)
        st, j, dt, fb = lane_probe(call, path, body, lane)
        recs.append({"run": i, "status": st, "sec": round(dt, 2), "token_ids": walk(j, "token_ids"),
                     "text": out_text("chat", j) if st == 200 else json.dumps(j)[:300], "fallback": fb})
    path, body = build("chat", [], "What is the capital of France? Answer in one sentence.", "", ntok, token_ids=True)
    st, j, dt, fb = lane_probe(call, path, body, lane)
    recs.append({"run": "text-after", "status": st, "sec": round(dt, 2), "token_ids": walk(j, "token_ids"),
                 "text": out_text("chat", j) if st == 200 else json.dumps(j)[:300], "fallback": fb})
    same = recs[0]["status"] == 200 and recs[0]["token_ids"] == recs[1]["token_ids"]
    if recs[0]["fallback"] != recs[1]["fallback"]: same = "lane-mixed"
    with open(os.path.join(RD, "repeat.jsonl"), "w") as out:
        for r in recs: out.write(json.dumps(r) + "\n")
    print(f"repeat ({lane}): {'LANE-MIXED' if same == 'lane-mixed' else 'IDENTICAL' if same else 'DIFF'} statuses={[r['status'] for r in recs]} fallbacks={[r['fallback'] for r in recs]}")

def cmd_twincheck(n1, n2, n3, lane="batched", ntok=64):
    """Third draw for the prompts whose two identity draws differ.  The continuous lane at temperature 0
    is not bit-deterministic across placements (capture-arc law), so a near-tie prompt can split on its
    first draw: when the third draw matches either of the first two the split is disclosed as a
    first-draw flake (2 of 3 agree); three distinct draws stay a twin FAIL (the 3l gate's rule)."""
    a = [json.loads(l) for l in open(os.path.join(RD, n1 + ".jsonl")) if l.strip().startswith("{")]
    b = [json.loads(l) for l in open(os.path.join(RD, n2 + ".jsonl")) if l.strip().startswith("{")]
    def differ(x, y):   # ids when both draws carry them (batched lane); text when they do not (serial lane: SSE-only ids)
        xi, yi = x.get("token_ids") or [], y.get("token_ids") or []
        return xi != yi if (xi and yi) else (x.get("text") != y.get("text"))
    redraw = [(x["image"], x["before"], x["after"]) for x, y in zip(a, b) if differ(x, y)]
    with open(os.path.join(RD, n3 + ".jsonl"), "w") as out:
        for image, before, after in redraw:
            imgs = [] if image == "-" else [image]
            path, body = build("chat", imgs, before, after, ntok, token_ids=True)
            st, j, dt, fb = lane_probe(call, path, body, lane)
            out.write(json.dumps({"image": image, "before": before, "after": after, "status": st, "sec": round(dt, 2),
                                  "prompt_tokens": walk(j, "prompt_tokens"), "token_ids": walk(j, "token_ids"),
                                  "text": out_text("chat", j) if st == 200 else json.dumps(j)[:300], "fallback": fb}) + "\n")
    print(f"twincheck {n1}/{n2} ({lane}): {len(redraw)} mismatch(es) re-drawn into {n3}")

def cmd_refusals():
    img = os.path.join(IMG, "typical_532x280.png")
    mime, b, uri = data_uri(img)
    cases = [
        ("chat_http_url", "/v1/chat/completions", {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]}, 400, "not fetched"),
        ("chat_file_path", "/v1/chat/completions", {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "/etc/hostname"}}]}]}, 400, "not fetched"),
        ("chat_assistant_image", "/v1/chat/completions", {"model": MODEL, "max_tokens": 8, "messages": [
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": [{"type": "image_url", "image_url": {"url": uri}}]},
            {"role": "user", "content": "and?"}]}, 400, "user messages"),
        ("chat_input_audio", "/v1/chat/completions", {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]}]}, 400, "not supported"),
        ("chat_not_an_image", "/v1/chat/completions", {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJDREVGRw=="}}]}]}, 400, "JPEG or PNG"),
        ("anthropic_url_source", "/v1/messages", {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]}]}, 400, "not fetched"),
        ("anthropic_document", "/v1/messages", {"model": MODEL, "max_tokens": 8, "messages": [{"role": "user", "content": [
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "QUJD"}}]}]}, 400, "not supported"),
        ("responses_tool_output_image", "/v1/responses", {"model": MODEL, "max_output_tokens": 8, "input": [
            {"type": "function_call_output", "call_id": "c1", "output": [{"type": "input_image", "image_url": uri}]}]}, 400, "user messages"),
        ("completions_no_images", "/v1/completions", {"model": MODEL, "max_tokens": 4, "prompt": "Say hi."}, 200, ""),
    ]
    n_ok = 0
    with open(os.path.join(RD, "refusals.jsonl"), "w") as out:
        for name, path, body, want, sub in cases:
            st, j, dt = post(path, body)
            raw = json.dumps(j)
            ok = st == want and (sub in raw)
            n_ok += ok
            out.write(json.dumps({"case": name, "status": st, "want": want, "substr": sub, "ok": ok, "body": raw[:400]}) + "\n")
    print(f"refusals: {n_ok}/{len(cases)} as expected")

def cmd_concurrency(name, kind, lane="batched", ntok=48):
    """8 concurrent chat requests: 7 text + (image | an 8th text)."""
    img = os.path.join(IMG, "typical_532x280.png")
    reqs = [build("chat", [], t, "", ntok, token_ids=True) for t in CONC_TEXT]
    if kind == "img":
        reqs.append(build("chat", [img], "Look carefully. ", AFTER, ntok, token_ids=True))
    else:
        reqs.append(build("chat", [], "Which planet is largest in the solar system? One sentence.", "", ntok, token_ids=True))
    results = [None] * len(reqs)
    def run(i):
        st, j, dt = call(reqs[i][0], reqs[i][1], lane)
        results[i] = {"row": i, "status": st, "sec": round(dt, 2), "token_ids": walk(j, "token_ids"),
                      "text": out_text("chat", j) if st == 200 else json.dumps(j)[:300]}
    ths = [threading.Thread(target=run, args=(i,)) for i in range(len(reqs))]
    for t in ths: t.start()
    for t in ths: t.join()
    with open(os.path.join(RD, name + ".jsonl"), "w") as out:
        for r in results: out.write(json.dumps(r) + "\n")
    print(f"concurrency {name} ({kind}, {lane}): statuses={[r['status'] for r in results]}")

def cmd_needles(surface, ntok=48):
    m = json.load(open(os.path.join(NEEDLE, "manifest.json")))
    items = m["items"] if isinstance(m, dict) else m
    n_ok = 0
    with open(os.path.join(RD, f"needle_{surface}.replies.jsonl"), "w") as out:
        for it in items:
            path, body = build(surface, [os.path.join(NEEDLE, it["image"])], "", it["question"], ntok)
            st, j, dt = post(path, body)
            reply = out_text(surface, j) if st == 200 else ""
            n_ok += st == 200
            out.write(json.dumps({"id": it["id"], "reply": reply, "status": st, "sec": round(dt, 2)}) + "\n"); out.flush()
    print(f"needles {surface}: {n_ok}/{len(items)} answered")

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "identity": cmd_identity(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "serial")
    elif c == "surfaces": cmd_surfaces()
    elif c == "repeat": cmd_repeat(sys.argv[2] if len(sys.argv) > 2 else "batched")
    elif c == "twincheck": cmd_twincheck(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "batched")
    elif c == "refusals": cmd_refusals()
    elif c == "concurrency": cmd_concurrency(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "batched")
    elif c == "needles": cmd_needles(sys.argv[2])
    else: raise SystemExit("unknown command")
