#!/usr/bin/env python3
"""Build a deep-context needle chat request for the KV-capacity gate
(gates/legs/memory/deep_ctx.sh).

Filler = real prose (the repo's markdown docs), a needle sentence buried at a
chosen depth, question at the end. Aims at a target token count via a
chars/token estimate; the authoritative count comes back in the response's
usage.prompt_tokens. LAW (2026-07-14): the 3.4 chars/tok estimate drifts
±4% per corpus rotation — leave >=6K tokens of seq_cap margin (a 500,034-token
estimate landed at 518,263 actual).

usage: needle_prompt.py <target_tokens> <out.json> [depth_frac] [needle_code]
                        [model] [rotate]

rotate (int, default 0) rotates the source-file order so two prompts of the
same size share no byte prefix — forces a COLD admit instead of a warm/fork
match against an earlier prompt built from the same corpus.

Corpus: $DS4_NEEDLE_CORPUS (colon-separated dirs/globs) or, by default, the
repo's local/docs/**/*.md plus top-level *.md relative to this script.
"""
import glob, json, os, sys

target_tokens = int(sys.argv[1])
out_path      = sys.argv[2]
depth_frac    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.55
code          = sys.argv[4] if len(sys.argv) > 4 else "7391-ALPHA-DELTA"
model         = sys.argv[5] if len(sys.argv) > 5 else "deepseek-chat"
rotate        = int(sys.argv[6]) if len(sys.argv) > 6 else 0

CHARS_PER_TOK = 3.4
target_chars = int(target_tokens * CHARS_PER_TOK)

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
corpus = os.environ.get("DS4_NEEDLE_CORPUS")
if corpus:
    srcs = []
    for part in corpus.split(":"):
        srcs += sorted(glob.glob(os.path.join(part, "**/*.md"), recursive=True)
                       if os.path.isdir(part) else glob.glob(part))
else:
    srcs = sorted(glob.glob(os.path.join(repo, "local/docs/**/*.md"), recursive=True))
    srcs += sorted(glob.glob(os.path.join(repo, "*.md")))
if not srcs:
    sys.exit("needle_prompt: no corpus files found")
rotate %= len(srcs)
srcs = srcs[rotate:] + srcs[:rotate]

parts, total, sect = [], 0, 0
while total < target_chars:
    grew = False
    for p in srcs:
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not t.strip():
            continue
        sect += 1
        blob = f"\n\n===== ARCHIVE SECTION {sect} ({os.path.basename(p)}) =====\n\n{t}"
        parts.append(blob)
        total += len(blob)
        grew = True
        if total >= target_chars:
            break
    if not grew:
        sys.exit("needle_prompt: corpus produced no text")

filler = "".join(parts)[:target_chars]
cut = int(len(filler) * depth_frac)
nl = filler.find("\n\n", cut)
if nl < 0: nl = cut
needle = (f"\n\nIMPORTANT ARCHIVE NOTE: The secret keystone code hidden in this "
          f"archive is {code}. Remember it exactly.\n\n")
doc = filler[:nl] + needle + filler[nl:]

body = {
    "model": model,
    "temperature": 0,
    "max_tokens": 64,
    "messages": [
        {"role": "user", "content":
            "Below is a long document archive. Read it carefully; a question follows at the end.\n\n"
            + doc +
            "\n\n===== END OF ARCHIVE =====\n\n"
            "Question: What is the secret keystone code hidden in this archive? "
            "Answer with just the code and nothing else."}
    ],
}
with open(out_path, "w") as f:
    json.dump(body, f)
print(f"prompt chars={len(doc)} est_tokens~{len(doc)/CHARS_PER_TOK:.0f} "
      f"needle_at={depth_frac:.0%} code={code} rotate={rotate} -> {out_path}")
