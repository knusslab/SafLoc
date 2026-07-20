#!/usr/bin/env python3
"""
categorize_dataset.py — Categorize dataset_c.jsonl into modality buckets.

Usage:
    python categorize_dataset.py

Outputs:
    data/categorized/log_only.jsonl
    data/categorized/log_config.jsonl
    data/categorized/log_code.jsonl
    data/categorized/log_code_config.jsonl
    data/categorized/other.jsonl
    data/categorized/_summary.json
"""
import json
from pathlib import Path

# Input and output paths
IN_PATH = Path("data/dataset_c.jsonl")
OUT_DIR = Path("data/categorized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

buckets = {
    "log_only": [],
    "log_config": [],
    "log_code": [],
    "log_code_config": [],
    "other": []
}

with IN_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        log = rec.get("artifact_availability", {}).get("log_available", False)
        code = rec.get("artifact_availability", {}).get("code_available", False)
        config = rec.get("artifact_availability", {}).get("config_available", False)
        if log and not code and not config:
            buckets["log_only"].append(rec)
        elif log and config and not code:
            buckets["log_config"].append(rec)
        elif log and code and not config:
            buckets["log_code"].append(rec)
        elif log and code and config:
            buckets["log_code_config"].append(rec)
        else:
            buckets["other"].append(rec)

# Write out each bucket
for name, records in buckets.items():
    with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Write summary
summary = {name: len(records) for name, records in buckets.items()}
summary["total"] = sum(summary.values())
with (OUT_DIR / "_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
