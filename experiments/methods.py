"""Diagnosers: 4 ablations + 10 SOTA baselines.

Honesty policy:
  * Faithful where the method maps cleanly to our (log, code, config) artifacts.
  * "Adapted" where the original needs modalities we do not have (metrics /
    traces / service-graph); the adaptation keeps the method's CORE mechanism
    and is documented in METHOD_NOTES.
  * N/A where no faithful adaptation exists (MicroRCA, Pinpoint) -> reported as
    N/A with a reason, never as a fabricated number.

Every method returns a prediction dict:
  {method,id, pred_file, pred_symbol, pred_rootcause, retrieved_code_paths, na}
"""
import re
import numpy as np

from .common import Ollama, cosine_topk, K, extract_json
from .data import query_text
from .labeling import normalize_category

# ----------------------------------------------------------------------------
METHOD_NOTES = {
    "log_only":     "Ablation: LLM sees only the runtime log; no retrieval.",
    "llm_only":     "Ablation: LLM sees only a short NL description; no retrieval, no grounding.",
    "unstructured": "Ablation: single combined index (top-3k), no modality stratification.",
    "SafLoc":       "Ours (enhanced): hybrid dense/lexical retrieval + tri-context prompt + few-shot RCA exemplars (leave-one-out) + symbol refinement + grounding.",
    "LLMAO":        "SOTA (LLM code FL): LLM over retrieved code candidates only.",
    "FaceIt":       "SOTA (2-stage LLM): stage-1 config-vs-code routing from log, stage-2 localize.",
    "RCACopilot":   "SOTA (retrieve-similar-incident + LLM RCA): kNN incident prior + LLM.",
    "DeepLog":      "SOTA-adapted (no LSTM training seq): log-embedding nearest-code localization.",
    "HADES":        "SOTA-adapted (no metrics): log+config cross-modal score fusion for code localization.",
    "Eadro":        "SOTA-adapted (no metrics/traces): log+code+config multi-source score fusion.",
    "MEPFL":        "SOTA-adapted (no traces): TF-IDF + logistic-regression root-cause classifier (leave-one-out).",
    "TraceRCA":     "SOTA-adapted (no traces): embedding kNN root-cause classifier (leave-one-out).",
    "MicroRCA":     "SOTA-adapted (no service graph/metrics): personalized-PageRank anomaly propagation over an embedding-similarity file graph.",
    "Pinpoint":     "SOTA-adapted (no request traces): cluster corpus files, localize within the query-correlated cluster.",
}
NA_METHODS = set()
ALL_METHODS = list(METHOD_NOTES.keys())

# Which metrics each method targets (others reported as N/A, never 0).
CAPS = {
    "log_only":     {"loc", "sym", "rca", "halluc"},
    "llm_only":     {"loc", "sym", "rca", "halluc"},
    "unstructured": {"recall", "loc", "sym", "rca", "halluc"},
    "SafLoc":       {"recall", "loc", "sym", "rca", "halluc"},
    "LLMAO":        {"recall", "loc", "sym", "rca", "halluc"},
    "FaceIt":       {"recall", "loc", "rca", "halluc"},
    "RCACopilot":   {"recall", "loc", "rca", "halluc"},
    "DeepLog":      {"recall", "loc", "halluc"},
    "HADES":        {"recall", "loc", "halluc"},
    "Eadro":        {"recall", "loc", "halluc"},
    "MEPFL":        {"rca"},
    "TraceRCA":     {"rca"},
    "MicroRCA":     {"recall", "loc", "halluc"},
    "Pinpoint":     {"recall", "loc", "halluc"},
}

CATS = "Programming Error, Configuration Error, Dependency Error, Other"
SYS = ("You are a fault-localization assistant for Knative (serverless) apps. "
       "Answer with STRICT JSON only, no prose.")


def _cands_block(title, docs, n=3, width=350):
    if not docs:
        return ""
    lines = [f"{title}:"]
    for i, d in enumerate(docs[:n], 1):
        snip = d["text"][:width].replace("\n", " ")
        lines.append(f'{i}. path="{d["path"]}"  ::  {snip}')
    return "\n".join(lines) + "\n"


def _parse(out):
    j = extract_json(out) or {}
    return {
        "file": (j.get("fault_file") or j.get("file") or "").strip() or None,
        "symbol": (j.get("fault_symbol") or j.get("symbol") or "").strip() or None,
        "root": normalize_category(j.get("root_cause")) if j.get("root_cause") else None,
    }


def _llm_diagnose(oll, log, code_cands=None, config_cands=None,
                  ground_paths=None, want_symbol=True, extra=""):
    body = [f"Runtime log (symptom):\n{(log or '(none)')[:1400]}\n"]
    if extra:
        body.append(extra)
    if code_cands:
        body.append(_cands_block("Candidate CODE files", code_cands))
    if config_cands:
        body.append(_cands_block("Candidate CONFIG artifacts", config_cands))
    body.append(f"Root-cause categories: {CATS}.")
    body.append('Output STRICT JSON: {"fault_file":"<path>","fault_symbol":'
                '"<function or empty>","symptom":"<short>","root_cause":'
                '"<one category>","evidence":"<short>"}')
    if ground_paths:
        body.append("The fault_file MUST be copied verbatim from one of the "
                    "candidate paths listed above.")
    out = oll.generate("\n".join(body), system=SYS, temperature=0.0, num_predict=320)
    pred = _parse(out)
    if ground_paths:
        allow = set(ground_paths)
        if pred["file"] not in allow:
            body.append("Your previous answer used an invalid path. Choose "
                        "fault_file EXACTLY from: " + " | ".join(ground_paths))
            out = oll.generate("\n".join(body), system=SYS, temperature=0.0, num_predict=320)
            p2 = _parse(out)
            if p2["file"] in allow:
                pred = p2
            elif ground_paths:
                pred["file"] = pred["file"] if pred["file"] in allow else ground_paths[0]
    if not want_symbol:
        pred["symbol"] = None
    return pred


def _mk(method, t, file=None, symbol=None, root=None, code_paths=None, na=False):
    return {"method": method, "id": t["id"], "pred_file": file,
            "pred_symbol": symbol, "pred_rootcause": root,
            "retrieved_code_paths": code_paths or [], "na": na}


# ---- ablations -------------------------------------------------------------
def m_log_only(t, ctx):
    p = _llm_diagnose(ctx.oll, t["log"])
    return _mk("log_only", t, p["file"], p["symbol"], p["root"])


def m_llm_only(t, ctx):
    desc = " ".join(t["keywords"]) + " :: " + (t["log"] or "")[:300]
    p = _llm_diagnose(ctx.oll, desc)
    return _mk("llm_only", t, p["file"], p["symbol"], p["root"])


def m_unstructured(t, ctx):
    docs = ctx.index.unified(t, K)
    code_paths = [d["path"] for d in docs if d["modality"] == "code"]
    p = _llm_diagnose(ctx.oll, t["log"], code_cands=docs)  # mixed, unlabeled
    return _mk("unstructured", t, p["file"], p["symbol"], p["root"], code_paths)


def m_safloc(t, ctx):
    # Multi-head diagnosis so localization and root cause don't distract each other.
    code = ctx.index.hybrid_code(query_text(t), exclude_issue=t["id"], k=K)
    code_paths = [d["path"] for d in code]
    config = ctx.index.stratified(t, K)["config"]
    # Head A -- clean grounded localization (code candidates only).
    pa = _llm_diagnose(ctx.oll, t["log"], code_cands=code,
                       ground_paths=code_paths or None, want_symbol=True)
    file = pa["file"]
    # Head B -- rich root cause: full tri-context + few-shot exemplars (LOO) +
    # a supervised classifier prior the LLM arbitrates over; take root only.
    nbrs = ctx.neighbors(t["id"], k=5)
    exemplars = "\n".join(
        f'- symptom="{n.get("gold_symptom") or "?"}" -> root_cause="{n.get("gold_rootcause")}"'
        for n in nbrs)
    clf_hint = ctx.mepfl_pred(t["id"])
    pb = _llm_diagnose(ctx.oll, t["log"], code_cands=code, config_cands=config,
                       want_symbol=False,
                       extra=(f"A supervised classifier predicts root cause: {clf_hint}.\n"
                              "Similar past incidents (root-cause priors):\n" + exemplars +
                              "\nWeigh these priors with the evidence. Use the code block for "
                              "logic/API/handler faults and the config block for "
                              "resource/permission/scaling faults.\n"))
    # Head C -- symbol refinement from the chosen file's contents.
    sym = _pick_symbol(ctx.oll, t["log"], file, code) or pa["symbol"]
    return _mk("SafLoc", t, file, sym, pb["root"], code_paths)




# ---- SafLoc helper routines ----
_FUNC = re.compile(r"\b(?:func|type|const|var)\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w+)")
_IDENT = re.compile(r"\b([A-Za-z_]\w{3,})\b")


def _focus(oll, log):
    out = oll.generate(
        "Extract the key search keywords from this log: error signatures, "
        "function/identifier names, and Kubernetes/resource types. Output ONE "
        "space-separated line, no prose.\n\n" + (log or "")[:1400],
        system="You output a single line of search keywords.",
        temperature=0.0, num_predict=64)
    return " ".join((out or "").split())[:300]


def _extract_funcs(text):
    names = []
    seen = set()
    for m in _FUNC.finditer(text or ""):
        nm = m.group(1)
        if nm not in seen:
            seen.add(nm)
            names.append(nm)
    if len(names) < 3:  # fall back to CamelCase identifiers
        for m in _IDENT.finditer(text or ""):
            nm = m.group(1)
            if nm not in seen and (nm[0].isupper() or any(c.isupper() for c in nm)):
                seen.add(nm)
                names.append(nm)
    return names[:20]


def _pick_symbol(oll, log, file, code_cands):
    if not file:
        return None
    doc = next((d for d in code_cands if d["path"] == file), None)
    if not doc:
        return None
    funcs = _extract_funcs(doc["text"])
    if not funcs:
        return None
    out = oll.generate(
        f"Runtime log:\n{(log or '')[:600]}\n\nSymbols defined in {file}:\n"
        + ", ".join(funcs) + "\n\nWhich ONE symbol is most likely the fault "
        "location? Answer with only the identifier.",
        system="Answer with a single code identifier, nothing else.",
        temperature=0.0, num_predict=16)
    tok = (out or "").strip().split()
    first = tok[0] if tok else ""
    name = "".join(ch for ch in first if ch.isalnum() or ch == "_")
    return name or None



# ---- SOTA: LLM-based -------------------------------------------------------
def m_llmao(t, ctx):
    code = ctx.index.code_ranking(t, K)
    code_paths = [d["path"] for d in code]
    p = _llm_diagnose(ctx.oll, t["log"], code_cands=code, ground_paths=code_paths or None)
    return _mk("LLMAO", t, p["file"], p["symbol"], p["root"], code_paths)


def m_faceit(t, ctx):
    # stage 1: route
    route = ctx.oll.generate(
        f"Runtime log:\n{(t['log'] or '')[:1200]}\n\nIs the root cause a "
        f"CONFIGURATION problem (resource/permission/scaling/manifest) or a CODE "
        f"problem? Answer STRICT JSON: {{\"kind\":\"config\"|\"code\"}}.",
        system=SYS, temperature=0.0, num_predict=40)
    kind = ((extract_json(route) or {}).get("kind") or "code").lower()
    strat = ctx.index.stratified(t, K)
    if "config" in kind:
        p = _llm_diagnose(ctx.oll, t["log"], config_cands=strat["config"], want_symbol=False)
        root = p["root"] or "Configuration Error"
    else:
        code_paths = [d["path"] for d in strat["code"]]
        p = _llm_diagnose(ctx.oll, t["log"], code_cands=strat["code"],
                          ground_paths=code_paths or None, want_symbol=False)
        root = p["root"] or "Programming Error"
    cp = [d["path"] for d in strat["code"]]
    return _mk("FaceIt", t, p["file"], None, root, cp)


def m_rcacopilot(t, ctx):
    # kNN similar incidents (leave-one-out) -> category prior + code candidates
    nbrs = ctx.neighbors(t["id"], k=3)
    prior = "\n".join(f'- similar incident: symptom="{n["gold_symptom"]}" '
                      f'root_cause="{n["gold_rootcause"]}"' for n in nbrs)
    code = ctx.index.code_ranking(t, K)
    code_paths = [d["path"] for d in code]
    p = _llm_diagnose(ctx.oll, t["log"], code_cands=code, ground_paths=code_paths or None,
                      want_symbol=False,
                      extra="Matched historical incidents (use as root-cause prior):\n"
                            + prior + "\n")
    return _mk("RCACopilot", t, p["file"], None, p["root"], code_paths)


# ---- SOTA: retrieval / fusion (no LLM) -------------------------------------
def _fusion_rank(ctx, t, weights):
    """Rank code docs by weighted similarity of the log query to each code doc's
    (code, config-of-issue, log-of-issue) vectors. weights=(wc, wk, wl)."""
    q = ctx.q_emb[ctx.row[t["id"]]]
    wc, wk, wl = weights
    scores = []
    for ci in ctx.index.by_mod.get("code", []):
        d = ctx.index.docs[ci]
        s = wc * _cos(q, ctx.index.emb[ci])
        if wk:
            s += wk * ctx.issue_sim(q, d["issue_id"], "config")
        if wl and d["issue_id"] != t["id"]:
            s += wl * ctx.issue_sim(q, d["issue_id"], "log")
        scores.append((s, d["path"]))
    scores.sort(reverse=True)
    return [p for _, p in scores]


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def _retrieval_method(name, weights):
    def fn(t, ctx):
        ranked = _fusion_rank(ctx, t, weights)
        topk = ranked[:K]
        return _mk(name, t, file=(ranked[0] if ranked else None),
                   root=None, code_paths=topk)
    return fn


m_deeplog = _retrieval_method("DeepLog", (1.0, 0.0, 0.0))   # log->code only
m_hades = _retrieval_method("HADES", (1.0, 0.3, 0.0))       # + config cross-modal
m_eadro = _retrieval_method("Eadro", (1.0, 0.3, 0.3))       # + log-log multi-source


# ---- SOTA: ML classifiers (root-cause; leave-one-out) ----------------------
def m_mepfl(t, ctx):
    return _mk("MEPFL", t, root=ctx.mepfl_pred(t["id"]))


def m_tracerca(t, ctx):
    return _mk("TraceRCA", t, root=ctx.tracerca_pred(t["id"]))


# ---- SOTA-adapted: graph / clustering localizers ---------------------------
def m_microrca(t, ctx):
    """Personalized-PageRank anomaly propagation over an embedding-similarity
    file graph (proxy for MicroRCA's attributed service graph + metrics)."""
    idxs, En = ctx.code_normed()
    if len(idxs) == 0:
        return _mk("MicroRCA", t)
    qn = ctx.q_emb[ctx.row[t["id"]]]
    qn = qn / (np.linalg.norm(qn) + 1e-9)
    qsim = En @ qn
    top = np.argsort(-qsim)[:min(25, len(idxs))]
    S = En[top]
    A = np.clip(S @ S.T, 0, None)
    np.fill_diagonal(A, 0.0)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-9)
    pers = np.clip(qsim[top], 0, None)
    pers = pers / (pers.sum() + 1e-9)
    r = pers.copy()
    for _ in range(50):
        r = 0.15 * pers + 0.85 * (A.T @ r)
    ranked = [ctx.index.docs[idxs[top[j]]]["path"] for j in np.argsort(-r)]
    return _mk("MicroRCA", t, file=(ranked[0] if ranked else None), code_paths=ranked[:K])


def m_pinpoint(t, ctx):
    """Cluster corpus files, localize within the query-correlated cluster
    (proxy for Pinpoint's request-trace clustering)."""
    ranked = ctx.pinpoint_rank(t["id"], k=K)
    return _mk("Pinpoint", t, file=(ranked[0] if ranked else None), code_paths=ranked[:K])


REGISTRY = {
    "log_only": m_log_only, "llm_only": m_llm_only, "unstructured": m_unstructured,
    "SafLoc": m_safloc, "LLMAO": m_llmao, "FaceIt": m_faceit, "RCACopilot": m_rcacopilot,
    "DeepLog": m_deeplog, "HADES": m_hades, "Eadro": m_eadro,
    "MEPFL": m_mepfl, "TraceRCA": m_tracerca,
    "MicroRCA": m_microrca, "Pinpoint": m_pinpoint,
}
