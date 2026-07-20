"""Dataset loading + deterministic gold-label extraction + retrieval corpus.

Each of the 107 records has: log, code_diff (custom unified-ish diff whose file
headers are `--- <path>`), config_diff (raw YAML manifest, often no path), and a
modality bucket (artifact_availability).

Gold fault files  -> from `--- <path>` headers in code_diff (deterministic).
Gold symbols      -> Go function/const/type names from `@@ .. @@ <ctx>` hunk
                     headers and added definition lines (heuristic).
Root-cause label  -> assigned later by LLM-assisted labeling (see labeling.py).
"""
import os
import re
import glob
import json

from .common import DATA_DIR

# A file header in this dataset's diff format is a line `--- path/to/file.go`.
_FILE_RE = re.compile(r"^---\s+(\S+)\s*$", re.M)
_HUNK_RE = re.compile(r"^@@[^@]*@@\s*(.*)$")
_GOFUNC_RE = re.compile(r"func\s*(?:\([^)]*\)\s*)?([A-Za-z_]\w+)")
_DEF_RE = re.compile(r"\b(?:func|type|const|var)\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w+)")
_CODE_EXT = (".go", ".py", ".js", ".ts", ".java", ".yaml", ".yml", ".sh",
             ".rs", ".c", ".cc", ".cpp", ".h", ".md", ".txt", ".mod", ".sum")


def parse_gold_files(code_diff):
    if not code_diff:
        return []
    out, seen = [], set()
    for m in _FILE_RE.finditer(code_diff):
        p = m.group(1).strip()
        p = re.sub(r"^[ab]/", "", p)
        if ("/" in p or p.endswith(_CODE_EXT)) and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def parse_gold_symbols(code_diff):
    syms = set()
    if not code_diff:
        return syms
    for line in code_diff.splitlines():
        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if m:
                for fm in _GOFUNC_RE.finditer(m.group(1)):
                    if len(fm.group(1)) > 2:
                        syms.add(fm.group(1))
        elif line.startswith("+") and not line.startswith("+++"):
            for dm in _DEF_RE.finditer(line[1:]):
                if len(dm.group(1)) > 2:
                    syms.add(dm.group(1))
    return syms


def prefix_code_text(code_diff, path):
    """Reconstruct approximate PRE-FIX text for one file from its diff hunks.

    Uses context (' ') and removed ('-') lines only -- i.e. the code roughly as
    it existed BEFORE the fix -- so the retrieval corpus does not simply contain
    the answer (the added fix lines are excluded)."""
    if not code_diff:
        return ""
    lines = code_diff.splitlines()
    keep = []
    cur = None
    for ln in lines:
        m = _FILE_RE.match(ln)
        if m:
            cur = re.sub(r"^[ab]/", "", m.group(1).strip())
            continue
        if cur != path:
            continue
        if ln.startswith("@@"):
            hm = _HUNK_RE.match(ln)
            if hm and hm.group(1).strip():
                keep.append(hm.group(1).strip())
        elif ln.startswith("-") and not ln.startswith("---"):
            keep.append(ln[1:])
        elif ln.startswith(" "):
            keep.append(ln[1:])
    return "\n".join(keep).strip()


def load_triplets():
    """Return list of triplet dicts with gold files/symbols attached."""
    triplets = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.jsonl"))):
        bucket = os.path.splitext(os.path.basename(f))[0]
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            code_diff = d.get("code_diff") or ""
            config_diff = d.get("config_diff") or ""
            gold_files = parse_gold_files(code_diff)
            triplets.append({
                "id": d.get("issue_id"),
                "bucket": bucket,
                "repo": d.get("repository"),
                "issue_url": d.get("issue_url"),
                "pr_url": d.get("closing_pr_url"),
                "keywords": d.get("keywords_hit") or [],
                "log": d.get("log") or "",
                "code_diff": code_diff,
                "config_diff": config_diff,
                "gold_files": gold_files,
                "gold_symbols": sorted(parse_gold_symbols(code_diff)),
                "has_code": bool(code_diff),
                "has_config": bool(config_diff),
                "has_log": bool(d.get("log")),
                # filled in by labeling.py:
                "gold_rootcause": None,
                "rootcause_conf": None,
                "gold_symptom": None,
            })
    return triplets


def query_text(t, max_chars=1600):
    """Natural-language fault query for a triplet (log + keywords)."""
    kw = " ".join(t.get("keywords") or [])
    log = (t.get("log") or "")[:max_chars]
    return (kw + "\n" + log).strip()


if __name__ == "__main__":
    ts = load_triplets()
    n = len(ts)
    with_files = sum(1 for t in ts if t["gold_files"])
    with_syms = sum(1 for t in ts if t["gold_symbols"])
    allfiles = [f for t in ts for f in t["gold_files"]]
    print(f"triplets: {n}")
    print(f"  with >=1 gold file:   {with_files}")
    print(f"  with >=1 gold symbol: {with_syms}")
    print(f"  total gold files:     {len(allfiles)} (unique {len(set(allfiles))})")
    print(f"  has_code/config/log:  "
          f"{sum(t['has_code'] for t in ts)}/"
          f"{sum(t['has_config'] for t in ts)}/"
          f"{sum(t['has_log'] for t in ts)}")
    ex = next(t for t in ts if t["gold_files"] and t["gold_symbols"])
    print("\nexample:", ex["id"])
    print("  gold_files:  ", ex["gold_files"][:5])
    print("  gold_symbols:", ex["gold_symbols"][:8])
    print("  prefix_text[0][:200]:",
          repr(prefix_code_text(ex["code_diff"], ex["gold_files"][0])[:200]))
