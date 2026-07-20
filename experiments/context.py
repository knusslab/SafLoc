"""Shared experiment context: embeddings, kNN neighbors, and the two
leave-one-out ML classifiers used by the MEPFL / TraceRCA baselines."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from .data import query_text


class Ctx:
    def __init__(self, triplets, index, oll):
        self.ts = triplets
        self.index = index
        self.oll = oll
        self.by_id = {t["id"]: t for t in triplets}
        self.id_list = [t["id"] for t in triplets]
        self.row = {tid: i for i, tid in enumerate(self.id_list)}
        # log-query embeddings for every triplet (kNN / fusion signals)
        self.q_emb = oll.embed([query_text(t) for t in triplets])
        self.qn = self.q_emb / (np.linalg.norm(self.q_emb, axis=1, keepdims=True) + 1e-9)
        # issue_id -> emb row for single-doc modalities (config, log)
        self._mod_row = {"config": {}, "log": {}}
        for i, d in enumerate(index.docs):
            if d["modality"] in self._mod_row:
                self._mod_row[d["modality"]].setdefault(d["issue_id"], i)
        self._mepfl = None
        self._tracerca = None
        self._pin = None
        self._code_norm = None

    # ---- similarity helpers ----
    def issue_sim(self, qvec, issue_id, modality):
        r = self._mod_row.get(modality, {}).get(issue_id)
        if r is None:
            return 0.0
        v = self.index.emb[r]
        return float(qvec @ v / ((np.linalg.norm(qvec) * np.linalg.norm(v)) + 1e-9))

    def neighbors(self, issue_id, k=3):
        i = self.row[issue_id]
        sims = self.qn @ self.qn[i]
        order = np.argsort(-sims)
        out = []
        for j in order:
            if j == i:
                continue
            out.append(self.by_id[self.id_list[j]])
            if len(out) >= k:
                break
        return out

    # ---- leave-one-out classifiers ----
    def _labels(self):
        return np.array([self.by_id[i]["gold_rootcause"] or "Other" for i in self.id_list])

    def mepfl_pred(self, issue_id):
        if self._mepfl is None:
            texts = [query_text(self.by_id[i]) for i in self.id_list]
            y = self._labels()
            preds = {}
            for h in range(len(self.id_list)):
                mask = np.ones(len(self.id_list), bool)
                mask[h] = False
                vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2),
                                      sublinear_tf=True)
                Xtr = vec.fit_transform([texts[j] for j in range(len(texts)) if mask[j]])
                clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                clf.fit(Xtr, y[mask])
                preds[self.id_list[h]] = clf.predict(vec.transform([texts[h]]))[0]
            self._mepfl = preds
        return self._mepfl[issue_id]

    def tracerca_pred(self, issue_id, k=5):
        if self._tracerca is None:
            y = self._labels()
            sims = self.qn @ self.qn.T
            np.fill_diagonal(sims, -1.0)
            preds = {}
            for i, tid in enumerate(self.id_list):
                nn = np.argsort(-sims[i])[:k]
                vals, cnt = np.unique(y[nn], return_counts=True)
                preds[tid] = vals[int(np.argmax(cnt))]
            self._tracerca = preds
        return self._tracerca[issue_id]

    # ---- Pinpoint proxy: cluster corpus files, correlate cluster with query ----
    def pinpoint_rank(self, issue_id, k=3, n_clusters=12):
        if self._pin is None:
            idxs = self.index.by_mod.get("code", [])
            E = self.index.emb[idxs]
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
            c = min(n_clusters, max(2, len(idxs)))
            km = KMeans(n_clusters=c, n_init=5, random_state=0).fit(En)
            self._pin = (idxs, En, km)
            self._code_norm = En
        idxs, En, km = self._pin
        q = self.q_emb[self.row[issue_id]]
        qn = q / (np.linalg.norm(q) + 1e-9)
        cent = km.cluster_centers_
        centn = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        cbest = int(np.argmax(centn @ qn))
        members = [i for i, lab in enumerate(km.labels_) if lab == cbest]
        if not members:
            members = list(range(len(idxs)))
        scored = sorted(((float(En[i] @ qn), self.index.docs[idxs[i]]["path"])
                         for i in members), reverse=True)
        return [p for _, p in scored][:k]

    def code_normed(self):
        """L2-normalized code-doc embedding matrix + their global doc indices."""
        idxs = self.index.by_mod.get("code", [])
        if self._code_norm is None or len(self._code_norm) != len(idxs):
            E = self.index.emb[idxs]
            self._code_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        return idxs, self._code_norm

