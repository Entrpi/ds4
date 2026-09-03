#!/usr/bin/env python3
"""Score JSONL replies against the deterministic Vision-Exp needle set."""

import argparse
import json
import os
import re
import sys


CATEGORIES = ("bar", "table", "screenshot", "digits", "count", "word")
DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test-vectors",
    "flash-vision-exp",
    "needle",
    "manifest.json",
)
NUMERIC_TARGET_RE = re.compile(r"[0-9]+(?:,[0-9]+)*")
NUMERIC_REPLY_RE = re.compile(
    r"(?<![\w,+.\-])"
    r"(?P<number>[0-9](?:[0-9,]*[0-9])?)"
    r"(?P<suffix>[A-Za-z%]*)"
    r"(?![\w,]|\.[0-9])"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score Vision-Exp needle replies from a JSONL file."
    )
    parser.add_argument("replies", nargs="?", help="JSONL replies file")
    parser.add_argument("--jsonl", help="JSONL replies file (alternative to positional path)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="manifest JSON path")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="ignore reply input and run internal correct/wrong-answer checks",
    )
    args = parser.parse_args()
    if not args.selftest:
        if args.replies and args.jsonl:
            parser.error("provide the replies file either positionally or with --jsonl, not both")
        if not args.replies and not args.jsonl:
            parser.error("a replies JSONL file is required unless --selftest is used")
    return args


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    items = manifest.get("items") if isinstance(manifest, dict) else manifest
    if not isinstance(items, list):
        raise ValueError("manifest must be a list or an object containing an items list")
    seen_ids = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every manifest item must be an object")
        for key in ("id", "category", "answer"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"manifest item has invalid {key!r}")
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate manifest id: {item['id']}")
        seen_ids.add(item["id"])
        aliases = item.get("answer_aliases", [])
        if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
            raise ValueError(f"invalid answer_aliases for {item['id']}")
    return items


def load_replies(path):
    replies = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            item_id = record.get("id")
            reply = record.get("reply")
            if not isinstance(item_id, str) or not isinstance(reply, str):
                raise ValueError(f"line {line_number} must contain string id and reply fields")
            replies[item_id] = reply
    return replies


def target_matches(target, reply):
    if not target:
        return False
    if NUMERIC_TARGET_RE.fullmatch(target):
        canonical_target = target.replace(",", "")
        for match in NUMERIC_REPLY_RE.finditer(reply):
            if match.group("number").replace(",", "") == canonical_target:
                return True
        return False
    pattern = r"(?<!\w)" + re.escape(target) + r"(?!\w)"
    return re.search(pattern, reply, flags=re.IGNORECASE) is not None


def item_matches(item, reply):
    targets = [item["answer"]] + item.get("answer_aliases", [])
    return any(target_matches(target, reply) for target in targets if target)


def score(items, replies):
    results = []
    for item in items:
        reply = replies.get(item["id"])
        passed = reply is not None and item_matches(item, reply)
        results.append((item, passed))
    return results


def print_report(results):
    for category in CATEGORIES:
        category_results = [passed for item, passed in results if item["category"] == category]
        print(f"{category}: {sum(category_results)}/{len(category_results)}")
    passed_total = sum(passed for _, passed in results)
    print(f"total: {passed_total}/{len(results)}")


def run_selftest(items):
    exact_replies = {item["id"]: item["answer"] for item in items}
    results = score(items, exact_replies)
    failed_ids = [item["id"] for item, passed in results if not passed]
    correct_count = len(items) - len(failed_ids)
    print(f"selftest correct-answers pass: {correct_count}/{len(items)}")
    if failed_ids:
        print("selftest unexpected failures: " + ", ".join(failed_ids))

    if not items:
        wrong_answer_fails = False
    else:
        wrong_reply = "definitely not the right answer at all"
        wrong_answer_fails = not item_matches(items[0], wrong_reply)
    print(
        "selftest wrong-answer correctly fails: "
        + ("yes" if wrong_answer_fails else "no")
    )

    passed = not failed_ids and bool(items) and wrong_answer_fails
    print("SELFTEST PASSED" if passed else "SELFTEST FAILED")
    return 0 if passed else 1


def main():
    args = parse_args()
    try:
        items = load_manifest(args.manifest)
        if args.selftest:
            return run_selftest(items)
        replies_path = args.jsonl or args.replies
        replies = load_replies(replies_path)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print_report(score(items, replies))
    return 0


if __name__ == "__main__":
    sys.exit(main())
