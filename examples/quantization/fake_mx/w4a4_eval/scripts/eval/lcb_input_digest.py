"""LCB per-problem input digest: verify the dataset snapshot matches the baseline."""

import glob
import hashlib
import json
import sys


def digest_reviews(dataset_out):
    """Digest the per-problem inputs from an EvalScope livecodebench output dir."""
    rows = []
    for f in sorted(glob.glob(f"{dataset_out}/*/reviews/qwen3.5/*.jsonl")):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            msgs = d.get("messages") or []
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            question = msgs[0].get("content", "") if msgs else ""
            rows.append(f"{d.get('index', '')}\n{question}")
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()[:16]


if __name__ == "__main__":
    print(digest_reviews(sys.argv[1]))
