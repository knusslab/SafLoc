# SafLoc — Silent Failure Localization for Serverless Applications

Artifact for the paper *"Silent Failures in Stateless Systems: A Tri-Context
RAG-Augmented LLM for Fault Localization in Serverless Applications"*
(anonymous submission, ICSOC 2026).

SafLoc is a Tri-Context Retrieval-Augmented Generation (TC-RAG) pipeline that
indexes source code, configuration, and runtime logs in three separate stores,
retrieves a fixed evidence budget from each, and prompts a local LLM for a
structured fault diagnosis. This repository contains the dataset (107 fault
triplets mined from closed Knative GitHub issues), the full experiment matrix
(SafLoc + 3 ablations + 10 adapted baselines), and the statistical analysis
that produces every number in the paper.

## Repository layout

```
data/categorized_git/      The 107-triplet dataset used in the paper (JSONL, one
                           file per modality bucket) + _summary.json (Table 2)
scripts/                   Dataset construction (GitHub/Stack Overflow mining,
                           categorization). NOT needed to reproduce the paper's
                           tables; only to regenerate the dataset from scratch.
experiments/               All experiment code (see below)
experiments/results/       Primary-run outputs: per-method predictions,
                           LLM-label + generation/embedding caches, metrics
experiments/results_q3/    Qwen3-14B re-run used for RQ5
ICSOC_main.tex             The paper (LLNCS), reference.bib
```

Key experiment modules: `common.py` (config + cached Ollama client),
`data.py` (triplet loading, gold extraction from fix diffs), `retrieval.py`
(BGE-M3 stores, stratified/unified/hybrid retrieval), `labeling.py`
(LLM-assisted root-cause labels), `methods.py` (all 14 diagnosers),
`metrics.py`, `run.py` (orchestrator), `analysis.py` (bootstrap CIs,
per-category, paired significance).

## Environment

Tested configuration (the one used for all reported numbers):

- Ubuntu Linux, Python **3.12.9**, single NVIDIA RTX 4090 (24 GB)
- [Ollama](https://ollama.com) serving on `http://localhost:11434` with models:
  - `qwen2.5-coder:7b-instruct` (digest `dae161e27b0e`) — diagnosis model
  - `bge-m3:latest` (digest `790764642607`) — embedder
  - `qwen3:14b` (digest `bdbd181c33f2`) — only for the RQ5 capacity test

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt      # fully pinned
ollama pull qwen2.5-coder:7b-instruct && ollama pull bge-m3   # + qwen3:14b for RQ5
```

No API keys are needed to reproduce the paper's results. A `GITHUB_TOKEN` is
required only if you want to re-mine the dataset from GitHub (see below).

## Reproducing the paper's results

The dataset ships with the repository, and `experiments/results/` includes the
disk caches of every LLM generation and embedding
(`gen_cache.pkl`, `emb_cache.pkl`) plus the model-assisted labels
(`labels.jsonl`). Two reproduction modes follow:

- **Exact (recommended for badge evaluation):** keep the caches in place. All
  LLM calls are replayed from cache, so every number reproduces bit-for-bit,
  no GPU required.
- **From scratch:** delete `experiments/results/*.pkl` and
  `experiments/results/labels.jsonl`. All LLM calls are re-issued at
  temperature 0. Note that greedy decoding is not bit-identical across GPUs,
  drivers, or Ollama versions, so numbers may shift by a few points; the
  paper's conclusions are supported by the bootstrap confidence intervals,
  not exact point values.

### Commands, mapped to paper results

| Paper result | Command | Output |
|---|---|---|
| Table 2 (modality distribution), RQ1 | shipped: `data/categorized_git/_summary.json` (regenerate: `scripts/categorize_dataset.py`) | counts used in Sec. 3.3 |
| Table 3 (`tab:results_ci`, overall + CIs) | `.venv/bin/python -m experiments.run` then `.venv/bin/python -m experiments.analysis` | `experiments/results/results_ci.tex` / `.csv` |
| Table 4 (`tab:sig`, paired significance) | `.venv/bin/python -m experiments.analysis` | `experiments/results/significance.csv` |
| Table 5 (`tab:percat`, per-category) | `.venv/bin/python -m experiments.analysis` | `experiments/results/percat_locprec.tex`, `percat_rca.tex`, `percat.csv` (paper merges the two tables and omits the n=4 Dependency column) |
| RQ5 (Qwen3-14B capacity test) | `SAFLOC_RESULTS=$PWD/experiments/results_q3 SAFLOC_LLM=qwen3:14b SAFLOC_LABEL_LLM=qwen2.5-coder:7b-instruct .venv/bin/python -m experiments.run` | `experiments/results_q3/predictions/` |

The RQ5 paired statistics (LocPrec −0.011, p=0.83; RCA +0.056, p=0.07) compare
`experiments/results/predictions/SafLoc.jsonl` against
`experiments/results_q3/predictions/SafLoc.jsonl` with
`experiments.analysis.paired_diff` (same seed and B as all other tests):

```python
# run from the repository root with SAFLOC_RESULTS unset (defaults to results/)
from experiments.analysis import item_scores, paired_diff, load_predictions
from experiments.data import load_triplets
from experiments.labeling import label_triplets
from experiments.common import Ollama
import json
ts = load_triplets(); label_triplets(ts, Ollama(), verbose=False)
ids = [t["id"] for t in ts]
gold = set(f for t in ts for f in t["gold_files"])
p7 = load_predictions()
p14 = {"SafLoc": {json.loads(l)["id"]: json.loads(l)
       for l in open("experiments/results_q3/predictions/SafLoc.jsonl")}}
for mk in ("loc", "rca"):
    a = item_scores("SafLoc", mk, p14, ts, gold)
    b = item_scores("SafLoc", mk, p7, ts, gold)
    print(mk, paired_diff(a, b, ids))
```

## Determinism

- Bootstrap analysis: fixed seed (`SEED=12345`), `B=5000` resamples
  (`experiments/analysis.py`); NumPy `default_rng` throughout.
- Classical ML baselines: `KMeans(random_state=0)`; leave-one-out
  `LogisticRegression`/TF-IDF and kNN are deterministic given the embeddings.
- LLM calls: temperature 0.0 everywhere, and every generation/embedding is
  cached on disk keyed by (model, prompt, options), so repeated runs replay
  identically. The only irreducible nondeterminism is greedy GPU decoding
  across different hardware/driver/Ollama versions (see "From scratch" above).

## Dataset provenance and licensing

The 107 triplets were mined from **public, closed GitHub issues and their
linked pull requests in the Knative organization** (knative/serving,
knative/eventing, and related repos), filtered as described in Sec. 3.2 of the
paper (closed by a linked PR, filed on or after 2024-12-01, containing a
fault signal, triplet-validated). Each JSONL record stores the issue id/url,
the reported log excerpt, and the code/config diff of the closing PR.
GitHub content is redistributed for research purposes with links to its
public source; Knative code excerpts are Apache-2.0. To regenerate from
scratch: `export GITHUB_TOKEN=<personal token>` then run
`scripts/mine_knative.py` followed by `scripts/categorize_dataset.py`
(results can differ from the shipped snapshot as issues get edited/deleted).
`scripts/mine_stackoverflow.py` builds the auxiliary Stack Overflow set
(`data/categorized_so/`) that is not used in the paper's experiments.

## Configuration knobs (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `SAFLOC_LLM` | `qwen2.5-coder:7b-instruct` | diagnosis model |
| `SAFLOC_EMB` | `bge-m3` | embedding model |
| `SAFLOC_LABEL_LLM` | = `SAFLOC_LLM` | labeling model |
| `SAFLOC_RESULTS` | `experiments/results` | output directory |
| `SAFLOC_K` | `3` | per-modality retrieval budget |
