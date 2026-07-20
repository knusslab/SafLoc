"""Run the full experiment matrix and emit results.

Usage:  python -m experiments.run [--methods m1,m2,...]

Outputs (experiments/results/):
  labels.jsonl              LLM-assisted root-cause labels (+ review flags)
  predictions/<method>.jsonl
  results.csv               all methods x metrics
  results.tex               paper-ready LaTeX table (splncs04-friendly)
"""
import os
import sys
import csv
import json
import time
import traceback

from .common import Ollama, RESULTS_DIR, K
from .data import load_triplets
from .labeling import label_triplets
from .retrieval import Index
from .context import Ctx
from .metrics import score_method
from . import methods as M

ORDER = ["llm_only", "log_only", "unstructured",
         "DeepLog", "HADES", "Eadro", "MEPFL", "TraceRCA",
         "LLMAO", "FaceIt", "RCACopilot", "MicroRCA", "Pinpoint", "SafLoc"]
COLS = ["Recall@%d" % K, "LocPrec", "SymMatch", "RCA", "Halluc", "Time(s)"]
_KEYMAP = {"Recall@%d" % K: "Recall@K"}


def _fmt(v):
    if v is None:
        return "--"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def run(methods=None):
    oll = Ollama()
    triplets = load_triplets()
    print(f"[1/4] labeling {len(triplets)} triplets (cached)...")
    label_triplets(triplets, oll, verbose=False)
    print("[2/4] building BGE-M3 index...")
    index = Index(triplets, oll)
    ctx = Ctx(triplets, index, oll)
    oll.save()

    todo = methods or M.ALL_METHODS
    rows = []
    os.makedirs(os.path.join(RESULTS_DIR, "predictions"), exist_ok=True)

    for name in ORDER:
        if name not in todo:
            continue
        if name in M.NA_METHODS:
            rows.append({"method": name, **{c: None for c in COLS}, "na": True})
            print(f"[3/4] {name:12s} N/A ({M.METHOD_NOTES[name]})")
            continue
        fn = M.REGISTRY[name]
        preds, errs, t0 = {}, 0, time.time()
        for t in triplets:
            try:
                preds[t["id"]] = fn(t, ctx)
            except Exception:
                errs += 1
                preds[t["id"]] = M._mk(name, t)  # empty prediction
        elapsed = time.time() - t0
        oll.save()
        with open(os.path.join(RESULTS_DIR, "predictions", f"{name}.jsonl"), "w") as f:
            for p in preds.values():
                f.write(json.dumps(p) + "\n")
        r = score_method(name, M.CAPS[name], preds, triplets, index, elapsed, len(triplets))
        r["na"] = False
        rows.append(r)
        disp = "  ".join(f"{c}={_fmt(r.get(_KEYMAP.get(c, c)))}" for c in COLS)
        print(f"[3/4] {name:12s} {disp}  (errs={errs})")

    _emit(rows, triplets)
    print(f"[4/4] wrote results.csv / results.tex to {RESULTS_DIR}")
    return rows


def _emit(rows, triplets):
    # CSV
    with open(os.path.join(RESULTS_DIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + COLS + ["na", "note"])
        for r in rows:
            w.writerow([r["method"]] +
                       [_fmt(r.get(_KEYMAP.get(c, c))) for c in COLS] +
                       [r.get("na", False), M.METHOD_NOTES.get(r["method"], "")])
    # LaTeX
    n_files = sum(1 for t in triplets if t["gold_files"])
    n_sym = sum(1 for t in triplets if t["gold_symbols"])
    lines = [
        r"\begin{table}[t]", r"  \centering",
        r"  \caption{Full experimental comparison on the %d-triplet Knative "
        r"dataset (localization scored over %d triplets with gold files, %d with "
        r"gold symbols; root cause over all %d). All methods run locally with "
        r"Qwen2.5-Coder-7B + BGE-M3. `\texttt{--}' = metric not applicable to the "
        r"method; \emph{N/A} rows require monitoring modalities (metrics, traces, "
        r"or a service graph) absent from this dataset. \textsc{SafLoc} is ours.}"
        % (len(triplets), n_files, n_sym, len(triplets)),
        r"  \label{tab:fullresults}",
        r"  \setlength{\tabcolsep}{5pt}",
        r"  \begin{tabular}{lcccccc}", r"    \toprule",
        r"    Method & Recall@%d & LocPrec & SymMatch & RCA & Halluc & Time (s) \\"
        % K, r"    \midrule",
    ]
    for r in rows:
        cells = []
        for c in COLS:
            v = r.get(_KEYMAP.get(c, c))
            cells.append("N/A" if (r.get("na") and c != "Time(s)") else _fmt(v))
        name = r["method"]
        disp = r"\textbf{\textsc{SafLoc}}" if name == "SafLoc" else name.replace("_", r"\_")
        row = f"    {disp} & " + " & ".join(cells) + r" \\"
        if name == "SafLoc":
            lines.append(r"    \midrule")
        lines.append(row)
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    with open(os.path.join(RESULTS_DIR, "results.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ms = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--methods"):
        ms = sys.argv[1].split("=", 1)[1].split(",") if "=" in sys.argv[1] else sys.argv[2].split(",")
    run(ms)
