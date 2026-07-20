"""Metric computation. Each metric is reported only for methods whose CAPS
include it; otherwise it is N/A (None) -- never a fabricated 0."""
import re

_WORD = re.compile(r"[A-Za-z_]\w+")


def _base(p):
    return p.rsplit("/", 1)[-1] if p else p


def _tokens(s):
    return {w.lower() for w in _WORD.findall(s or "") if len(w) > 2}


def score_method(method, caps, preds, triplets, index, elapsed, n):
    r = {"method": method, "Recall@K": None, "LocPrec": None,
         "SymMatch": None, "RCA": None, "Halluc": None,
         "Time(s)": (elapsed / n) if n else None}

    if "recall" in caps:
        num = den = 0
        for t in triplets:
            if not t["gold_files"]:
                continue
            den += 1
            got = set(preds[t["id"]]["retrieved_code_paths"])
            if got & set(t["gold_files"]):
                num += 1
        r["Recall@K"] = num / den if den else None

    if "loc" in caps:
        gold_bases_all = num = den = 0
        num = den = 0
        for t in triplets:
            if not t["gold_files"]:
                continue
            den += 1
            pf = preds[t["id"]]["pred_file"]
            if pf and _base(pf) in {_base(g) for g in t["gold_files"]}:
                num += 1
        r["LocPrec"] = num / den if den else None

    if "sym" in caps:
        num = den = 0
        for t in triplets:
            if not t["gold_symbols"]:
                continue
            den += 1
            gold = {s.lower() for s in t["gold_symbols"]}
            if _tokens(preds[t["id"]]["pred_symbol"]) & gold:
                num += 1
        r["SymMatch"] = num / den if den else None

    if "rca" in caps:
        num = den = 0
        for t in triplets:
            if not t["gold_rootcause"]:
                continue
            den += 1
            if preds[t["id"]]["pred_rootcause"] == t["gold_rootcause"]:
                num += 1
        r["RCA"] = num / den if den else None

    if "halluc" in caps:
        num = den = 0
        for t in triplets:
            pf = preds[t["id"]]["pred_file"]
            if pf is None:
                continue
            den += 1
            if pf not in index.code_paths:
                num += 1
        r["Halluc"] = num / den if den else None

    return r
