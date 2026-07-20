"""Retrieval corpus + BGE-M3 index with stratified vs. unified retrieval.

Corpus (built from dataset artifacts, since full pre-fix repo snapshots are not
available -- a documented limitation):
  * code store   : one doc per (issue, changed file); text = pre-fix code
                   (context+removed lines only, NOT the added fix).
  * config store : one doc per issue with a config manifest.
  * log store     : one doc per issue log.

Localization target for issue i = its own code doc(s); distractors = the code
docs of the other 90 issues (165 unique real Knative files). For the log
modality we use leave-one-out (an issue never retrieves its own log).
"""
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import Ollama, cosine_topk, K
from .data import prefix_code_text, query_text

_KIND = re.compile(r"^kind:\s*(\S+)", re.M)
_NAME = re.compile(r"^\s*name:\s*(\S+)", re.M)


def _config_path(issue_id, cfg):
    kind = _KIND.search(cfg)
    name = _NAME.search(cfg)
    tag = "/".join(x.group(1) for x in (kind, name) if x)
    return f"config::{tag or issue_id}"


def build_corpus(triplets):
    docs = []  # each: dict(doc_id, issue_id, modality, path, text)
    for t in triplets:
        for path in t["gold_files"]:
            txt = prefix_code_text(t["code_diff"], path)
            docs.append({"doc_id": f"{t['id']}::code::{path}",
                         "issue_id": t["id"], "modality": "code",
                         "path": path, "text": f"{path}\n{txt}"[:2000]})
        if t["has_config"]:
            cfg = t["config_diff"]
            docs.append({"doc_id": f"{t['id']}::config",
                         "issue_id": t["id"], "modality": "config",
                         "path": _config_path(t["id"], cfg),
                         "text": cfg[:2000]})
        if t["has_log"]:
            docs.append({"doc_id": f"{t['id']}::log",
                         "issue_id": t["id"], "modality": "log",
                         "path": f"log::{t['id']}",
                         "text": t["log"][:2000]})
    return docs


class Index:
    def __init__(self, triplets, ollama: Ollama):
        self.oll = ollama
        self.docs = build_corpus(triplets)
        self.by_mod = {}
        for i, d in enumerate(self.docs):
            self.by_mod.setdefault(d["modality"], []).append(i)
        texts = [d["text"] for d in self.docs]
        self.emb = ollama.embed(texts) if texts else np.zeros((0, 1024), np.float32)
        # set of all indexed file paths (for hallucination check on code paths)
        self.code_paths = {d["path"] for d in self.docs if d["modality"] == "code"}
        # lexical (BM25-style TF-IDF) index over code docs, for hybrid retrieval
        self.code_idx = self.by_mod.get("code", [])
        code_texts = [self.docs[i]["text"] for i in self.code_idx]
        if code_texts:
            self.tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                                         sublinear_tf=True, token_pattern=r"[A-Za-z_]\w+")
            self.code_lex = self.tfidf.fit_transform(code_texts)
        else:
            self.tfidf, self.code_lex = None, None

    def _subset(self, modality, exclude_issue=None, exclude_own_log=True):
        idxs = self.by_mod.get(modality, [])
        if exclude_issue and (modality == "log" and exclude_own_log):
            idxs = [i for i in idxs if self.docs[i]["issue_id"] != exclude_issue]
        return idxs

    def _topk(self, qvec, modality, k, exclude_issue):
        idxs = self._subset(modality, exclude_issue)
        if not idxs:
            return []
        sub = self.emb[idxs]
        loc, scores = cosine_topk(qvec, sub, k)
        return [(idxs[j], float(s)) for j, s in zip(loc, scores)]

    def stratified(self, triplet, k=K):
        """Top-k per modality -> {modality: [docs]} (SafLoc retrieval)."""
        qvec = self.oll.embed(query_text(triplet))[0]
        out = {}
        for mod in ("code", "config", "log"):
            hits = self._topk(qvec, mod, k, triplet["id"])
            out[mod] = [dict(self.docs[i], score=s) for i, s in hits]
        return out

    def unified(self, triplet, k=K):
        """Top-3k over ALL modalities combined (unstructured retrieval)."""
        qvec = self.oll.embed(query_text(triplet))[0]
        pool = []
        for mod in ("code", "config", "log"):
            pool += self._subset(mod, triplet["id"])
        if not pool:
            return []
        sub = self.emb[pool]
        loc, scores = cosine_topk(qvec, sub, 3 * k)
        return [dict(self.docs[pool[j]], score=float(s)) for j, s in zip(loc, scores)]

    def code_ranking(self, triplet, k=K):
        """Ranked code docs only (for Recall@k / retrieval baselines)."""
        qvec = self.oll.embed(query_text(triplet))[0]
        return [dict(self.docs[i], score=s)
                for i, s in self._topk(qvec, "code", k, triplet["id"])]

    def hybrid_code(self, query_str, exclude_issue=None, k=K, pool=30):
        """Hybrid dense (BGE-M3) + lexical (TF-IDF) code retrieval via reciprocal
        rank fusion. Better top-k than dense-only, used by the enhanced SafLoc."""
        if not self.code_idx:
            return []
        qv = self.oll.embed(query_str)[0]
        sub = self.emb[self.code_idx]
        dloc, _ = cosine_topk(qv, sub, min(pool, len(self.code_idx)))
        lex_order = []
        if self.tfidf is not None:
            ql = self.tfidf.transform([query_str])
            lsc = np.asarray((self.code_lex @ ql.T).todense()).ravel()
            lex_order = list(np.argsort(-lsc)[:pool])
        rr = {}
        for rank, j in enumerate(dloc):
            rr[int(j)] = rr.get(int(j), 0.0) + 1.0 / (60 + rank)
        for rank, j in enumerate(lex_order):
            rr[int(j)] = rr.get(int(j), 0.0) + 1.0 / (60 + rank)
        out = []
        for j, sc in sorted(rr.items(), key=lambda x: -x[1]):
            out.append(dict(self.docs[self.code_idx[j]], score=float(sc)))
            if len(out) >= k:
                break
        return out


def recall_at_k(index, triplets, k=K):
    hits = 0
    n = 0
    for t in triplets:
        if not t["gold_files"]:
            continue
        n += 1
        got = {d["path"] for d in index.code_ranking(t, k)}
        if got & set(t["gold_files"]):
            hits += 1
    return hits / n if n else 0.0, n
