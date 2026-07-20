"""Post-hoc statistical analysis: bootstrap confidence intervals, per-category
breakdowns, and paired significance tests over the saved predictions.

No models are called here -- it reads experiments/results/predictions/*.jsonl and
recomputes metrics at the item level so we can bootstrap-resample fault instances.

Outputs (experiments/results/):
  results_ci.csv / results_ci.tex   overall metrics with 95% bootstrap CIs
  percat.csv                        every method x metric x category (long form)
  percat_locprec.tex                LocPrec by root-cause category (core table)
  percat_recall.tex                 Recall@3 by root-cause category
  percat_rca.tex                    RCA by root-cause category
  significance.csv                  paired SafLoc-vs-baseline diffs (95% CI + p)
"""
import os
import csv
import json
import glob
import numpy as np

from .common import RESULTS_DIR, K, Ollama
from .data import load_triplets
from .labeling import label_triplets
from .metrics import _base, _tokens
from . import methods as M

B = 5000            # bootstrap resamples
SEED = 12345
ROOTCATS = ["Programming Error", "Configuration Error", "Dependency Error"]
METRICS = [("Recall@1", "recall1"), ("Recall@3", "recall3"), ("LocPrec", "loc"),
           ("SymMatch", "sym"), ("RCA", "rca"), ("Halluc", "halluc")]
_CAPMAP = {"recall1": "recall", "recall3": "recall", "loc": "loc",
           "sym": "sym", "rca": "rca", "halluc": "halluc"}


def load_predictions():
    preds = {}
    for f in glob.glob(os.path.join(RESULTS_DIR, "predictions", "*.jsonl")):
        name = os.path.splitext(os.path.basename(f))[0]
        preds[name] = {p["id"]: p for p in
                       (json.loads(l) for l in open(f) if l.strip())}
    return preds


def item_scores(method, metric, preds, triplets, code_paths):
    """id -> (applicable: bool, correct: 0/1) for one method+metric."""
    P = preds.get(method, {})
    out = {}
    for t in triplets:
        p = P.get(t["id"])
        appl, cor = False, 0
        if metric in ("recall1", "recall3"):
            if t["gold_files"]:
                appl = True
                topn = (p["retrieved_code_paths"][:1] if metric == "recall1"
                        else p["retrieved_code_paths"][:K]) if p else []
                cor = int(bool(set(topn) & set(t["gold_files"])))
        elif metric == "loc":
            if t["gold_files"]:
                appl = True
                pf = p["pred_file"] if p else None
                cor = int(bool(pf) and _base(pf) in {_base(g) for g in t["gold_files"]})
        elif metric == "sym":
            if t["gold_symbols"]:
                appl = True
                gold = {s.lower() for s in t["gold_symbols"]}
                cor = int(bool(p) and bool(_tokens(p["pred_symbol"]) & gold))
        elif metric == "rca":
            if t["gold_rootcause"]:
                appl = True
                cor = int(bool(p) and p["pred_rootcause"] == t["gold_rootcause"])
        elif metric == "halluc":
            pf = p["pred_file"] if p else None
            if pf is not None:
                appl = True
                cor = int(pf not in code_paths)
        out[t["id"]] = (appl, cor)
    return out


def rate_ci(items, ids, seed=SEED):
    arr = np.array([items[i][1] for i in ids if items[i][0]], float)
    if arr.size == 0:
        return None
    rng = np.random.default_rng(seed)
    boots = arr[rng.integers(0, arr.size, size=(B, arr.size))].mean(axis=1)
    return arr.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), arr.size


def paired_diff(itemsA, itemsB, ids, seed=SEED):
    common = [i for i in ids if itemsA[i][0] and itemsB[i][0]]
    if not common:
        return None
    a = np.array([itemsA[i][1] for i in common], float)
    b = np.array([itemsB[i][1] for i in common], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    d = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return a.mean() - b.mean(), float(lo), float(hi), float(min(p, 1.0)), a.size


def _cell(res, dec=3):
    if res is None:
        return "--"
    pt, lo, hi, _ = res
    return f"{pt:.{dec}f} [{lo:.2f},{hi:.2f}]"


def main():
    oll = Ollama()
    triplets = load_triplets()
    label_triplets(triplets, oll, verbose=False)  # cached -> instant
    preds = load_predictions()
    code_paths = set(f for t in triplets for f in t["gold_files"])
    all_ids = [t["id"] for t in triplets]

    # cache item-level scores
    scores = {}
    for name in M.ALL_METHODS:
        for _, mk in METRICS:
            if _CAPMAP[mk] in M.CAPS[name]:
                scores[(name, mk)] = item_scores(name, mk, preds, triplets, code_paths)

    order = ["log_only", "llm_only", "unstructured", "DeepLog", "HADES", "Eadro",
             "MEPFL", "TraceRCA", "LLMAO", "FaceIt", "RCACopilot", "MicroRCA",
             "Pinpoint", "SafLoc"]

    # ---- 1. overall table with CIs ----
    rows = []
    for name in order:
        row = {"method": name}
        for label, mk in METRICS:
            res = rate_ci(scores[(name, mk)], all_ids) if (name, mk) in scores else None
            row[label] = res
        rows.append(row)
    _write_overall(rows)

    # ---- 2. per-category (root cause) ----
    cat_ids = {c: [t["id"] for t in triplets if t["gold_rootcause"] == c] for c in ROOTCATS}
    percat_rows = []
    for name in order:
        for label, mk in METRICS:
            if (name, mk) not in scores:
                continue
            for c in ROOTCATS:
                res = rate_ci(scores[(name, mk)], cat_ids[c])
                if res:
                    percat_rows.append([name, label, c, f"{res[0]:.4f}",
                                        f"{res[1]:.4f}", f"{res[2]:.4f}", res[3]])
    with open(os.path.join(RESULTS_DIR, "percat.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "metric", "category", "point", "ci_lo", "ci_hi", "n"])
        w.writerows(percat_rows)

    loc_methods = [m for m in order if "loc" in M.CAPS.get(m, set())]
    rca_methods = [m for m in order if "rca" in M.CAPS.get(m, set())]
    _write_percat("percat_locprec.tex", "LocPrec", "loc", loc_methods, scores, cat_ids,
                  "Location precision by root-cause category (point [95\\% CI]). "
                  "Config-origin faults are where tri-context retrieval helps most.")
    _write_percat("percat_recall.tex", f"Recall@{K}", "recall3", loc_methods, scores, cat_ids,
                  f"Retrieval recall@{K} by root-cause category (point [95\\% CI]).")
    _write_percat("percat_rca.tex", "RCA", "rca", rca_methods, scores, cat_ids,
                  "Root-cause accuracy by category (point [95\\% CI]).")

    # ---- 3. per modality bucket (LocPrec) ----
    buckets = ["log_code", "log_code_config", "log_config", "log_only", "other"]
    bkt_ids = {b: [t["id"] for t in triplets if t["bucket"] == b] for b in buckets}
    print("\n=== LocPrec by modality bucket (n) ===")
    hdr = "method".ljust(13) + "".join(f"{b}({len(bkt_ids[b])})".rjust(20) for b in buckets)
    print(hdr)
    for name in loc_methods:
        line = name.ljust(13)
        for b in buckets:
            res = rate_ci(scores[(name, "loc")], bkt_ids[b]) if (name, "loc") in scores else None
            line += (_cell(res, 2)).rjust(20)
        print(line)

    # ---- 4. paired significance vs SafLoc ----
    sig = [["metric", "baseline", "safloc_minus_baseline", "ci_lo", "ci_hi", "p", "n", "significant"]]
    print("\n=== Paired bootstrap: SafLoc - baseline (95% CI, p) ===")
    for mk, label, bases in [("loc", "LocPrec", ["LLMAO", "RCACopilot", "DeepLog", "unstructured", "log_only"]),
                             ("rca", "RCA", ["MEPFL", "RCACopilot", "TraceRCA", "LLMAO", "log_only"])]:
        for base in bases:
            if ("SafLoc", mk) not in scores or (base, mk) not in scores:
                continue
            r = paired_diff(scores[("SafLoc", mk)], scores[(base, mk)], all_ids)
            if not r:
                continue
            diff, lo, hi, p, n = r
            signif = not (lo <= 0 <= hi)
            print(f"  {label:8s} SafLoc-{base:11s} {diff:+.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"p={p:.3f} n={n} {'SIGNIFICANT' if signif else 'ns'}")
            sig.append([label, base, f"{diff:.4f}", f"{lo:.4f}", f"{hi:.4f}",
                        f"{p:.4f}", n, signif])
    with open(os.path.join(RESULTS_DIR, "significance.csv"), "w", newline="") as f:
        csv.writer(f).writerows(sig)
    print(f"\nWrote results_ci.{{csv,tex}}, percat*.tex/csv, significance.csv to {RESULTS_DIR}")


def _write_overall(rows):
    with open(os.path.join(RESULTS_DIR, "results_ci.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [m for m, _ in METRICS])
        for r in rows:
            w.writerow([r["method"]] +
                       ["" if r[m] is None else f"{r[m][0]:.4f} [{r[m][1]:.4f},{r[m][2]:.4f}]"
                        for m, _ in METRICS])
    lines = [r"\begin{table}[t]", r"  \centering",
             r"  \caption{Overall results with 95\% bootstrap confidence intervals "
             r"(5000 resamples over fault instances). \textsc{SafLoc} is ours.}",
             r"  \label{tab:results_ci}", r"  \setlength{\tabcolsep}{3pt}",
             r"  \footnotesize", r"  \begin{tabular}{lcccccc}", r"    \toprule",
             rf"    Method & Recall@1 & Recall@{K} & LocPrec & SymMatch & RCA & Halluc \\",
             r"    \midrule"]
    for r in rows:
        disp = r"\textbf{\textsc{SafLoc}}" if r["method"] == "SafLoc" else r["method"].replace("_", r"\_")
        if r["method"] == "SafLoc":
            lines.append(r"    \midrule")
        cells = [_cell(r[m]) for m, _ in METRICS]
        lines.append(f"    {disp} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(os.path.join(RESULTS_DIR, "results_ci.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_percat(fname, label, mk, methods, scores, cat_ids, caption):
    cats = list(cat_ids.keys())
    ncol = "l" + "c" * len(cats)
    head = " & ".join(["Method"] + [c.replace(" Error", "").replace(" ", "~") +
                                     f"~(n={len(cat_ids[c])})" for c in cats])
    lines = [r"\begin{table}[t]", r"  \centering", r"  \caption{%s}" % caption,
             r"  \label{tab:%s}" % fname.replace(".tex", ""),
             r"  \setlength{\tabcolsep}{4pt}", r"  \footnotesize",
             r"  \begin{tabular}{%s}" % ncol, r"    \toprule",
             "    " + head + r" \\", r"    \midrule"]
    for name in methods:
        disp = r"\textbf{\textsc{SafLoc}}" if name == "SafLoc" else name.replace("_", r"\_")
        if name == "SafLoc":
            lines.append(r"    \midrule")
        cells = []
        for c in cats:
            res = rate_ci(scores[(name, mk)], cat_ids[c]) if (name, mk) in scores else None
            cells.append(_cell(res, 2))
        lines.append(f"    {disp} & " + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(os.path.join(RESULTS_DIR, fname), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
