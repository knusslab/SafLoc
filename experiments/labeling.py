"""LLM-assisted root-cause labeling (post-hoc oracle).

The labeler sees the runtime log AND the closing PR's fix (code/config diff),
so it has strictly more information than any method under test (which only sees
pre-fix retrieval context). Labels below a confidence threshold are flagged for
human review. This is the "gold" used to score root-cause accuracy (RCA); it is
model-assisted, not human-verified -- a documented threat to validity.
"""
import os
import json

from .common import Ollama, ROOTCAUSE_CATEGORIES, RESULTS_DIR, extract_json, LABEL_MODEL

REVIEW_THRESHOLD = 0.6

SYS = ("You are an expert software-reliability engineer labeling the ROOT CAUSE "
       "of a resolved Knative (serverless) fault. You are shown the runtime log "
       "AND the fix (diff) that closed the issue. Output STRICT JSON only, no prose.")


def normalize_category(s):
    s = (s or "").lower()
    if any(w in s for w in ("program", "logic", "api misuse", "handler", "code bug", "null", "nil")):
        return "Programming Error"
    if any(w in s for w in ("config", "resource", "permission", "rbac", "scal", "limit", "quota", "manifest")):
        return "Configuration Error"
    if any(w in s for w in ("depend", "version", "module", "import", "package", "vendor", "compat")):
        return "Dependency Error"
    for c in ROOTCAUSE_CATEGORIES:
        if c.lower() in s:
            return c
    return "Other"


def _prompt(t):
    cats = "\n".join(f"- {c}" for c in ROOTCAUSE_CATEGORIES)
    files = "\n".join(t["gold_files"][:8]) or "(none)"
    return f"""Root-cause categories:
{cats}

Runtime log / issue report:
{(t['log'] or '(none)')[:1500]}

Files changed by the fix:
{files}

Code fix (diff excerpt):
{(t['code_diff'] or '(none)')[:2500]}

Config artifact (excerpt):
{(t['config_diff'] or '(none)')[:800]}

Using the FIX as ground truth, classify the ROOT CAUSE. Output JSON exactly:
{{"root_cause": "<one category, verbatim>", "symptom": "<short: timeout|OOMKilled|permission denied|500|crashloop|...>", "primary_symbol": "<main function or k8s kind changed, or empty>", "confidence": <0.0-1.0>}}"""


def label_triplets(triplets, oll: Ollama, verbose=True):
    labels = []
    for i, t in enumerate(triplets):
        out = oll.generate(_prompt(t), system=SYS, temperature=0.0, num_predict=200,
                           model=LABEL_MODEL)
        j = extract_json(out) or {}
        rc = normalize_category(j.get("root_cause", "Other"))
        t["gold_rootcause"] = rc
        try:
            conf = float(j.get("confidence", 0) or 0)
        except (TypeError, ValueError):
            conf = 0.0
        t["rootcause_conf"] = conf
        t["gold_symptom"] = (j.get("symptom") or "").strip()
        t["label_primary_symbol"] = (j.get("primary_symbol") or "").strip()
        labels.append({"id": t["id"], "bucket": t["bucket"], "root_cause": rc,
                       "symptom": t["gold_symptom"], "confidence": conf,
                       "needs_review": conf < REVIEW_THRESHOLD})
        if verbose and (i + 1) % 20 == 0:
            print(f"  labeled {i + 1}/{len(triplets)}")
        oll.save()
    with open(os.path.join(RESULTS_DIR, "labels.jsonl"), "w") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    return labels


if __name__ == "__main__":
    from collections import Counter
    from .data import load_triplets
    oll = Ollama()
    ts = load_triplets()
    labs = label_triplets(ts, oll)
    dist = Counter(l["root_cause"] for l in labs)
    lowc = [l for l in labs if l["needs_review"]]
    print("\nroot-cause distribution:", dict(dist))
    print(f"low-confidence (<{REVIEW_THRESHOLD}) flagged for review: {len(lowc)}")
    for l in lowc[:12]:
        print(f"  REVIEW {l['id']}  rc={l['root_cause']}  conf={l['confidence']:.2f}")
