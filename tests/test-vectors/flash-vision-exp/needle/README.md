# Vision-Exp needle validation set

This is a deterministic needle-in-image visual-QA validation set for the upcoming image-input ("Vision-Exp") feature.
Its PASS band will be established from an UPSTREAM REFERENCE BINARY's own answers on these same items, not from a fixed absolute threshold here; 24/24 is not declared as the required bar.

Manifest schema: `{"font_used": "...", "seed": 20260903, "items": [...]}` with one scored object per PNG.
Resolved font: `/System/Library/Fonts/Supplemental/Arial.ttf`.
Measured `needle/` total: 243100 bytes (237.4 KiB).

Regenerate from the repository root:
```sh
python3 tests/gen_vision_needle.py
```
Score JSONL replies, or run the built-in scorer self-test:
```sh
python3 tests/vision_needle_score.py path/to/replies.jsonl
python3 tests/vision_needle_score.py --selftest
```
