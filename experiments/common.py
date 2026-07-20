"""Shared config + local Ollama client (LLM generation + BGE-M3 embeddings).

Everything runs locally through Ollama's HTTP API (no external API keys):
  - LLM:        qwen2.5-coder:7b-instruct
  - Embeddings: bge-m3 (1024-d), matching the paper's embedder.

Both generation and embedding calls are cached on disk so re-runs are cheap
and deterministic (temperature 0 for generation).
"""
import os
import json
import hashlib
import pickle
import numpy as np
import requests

BASE = os.environ.get("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.environ.get("SAFLOC_LLM", "qwen2.5-coder:7b-instruct")
EMB_MODEL = os.environ.get("SAFLOC_EMB", "bge-m3")
LABEL_MODEL = os.environ.get("SAFLOC_LABEL_LLM", LLM_MODEL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "categorized_git")
RESULTS_DIR = os.environ.get("SAFLOC_RESULTS") or os.path.join(ROOT, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Per-modality retrieval budget k (paper uses k=3).
K = int(os.environ.get("SAFLOC_K", "3"))
EMB_DIM = 1024

# Root-cause taxonomy (dominant categories from Xie et al. 2025 + catch-all).
ROOTCAUSE_CATEGORIES = [
    "Programming Error",
    "Configuration Error",
    "Dependency Error",
    "Other",
]


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def extract_json(text: str):
    """Extract the first balanced JSON object from an LLM response."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        blob = text[start:i + 1]
                        try:
                            return json.loads(blob)
                        except Exception:
                            break  # try next '{'
        start = text.find("{", start + 1)
    return None


class Ollama:
    def __init__(self):
        self.emb_cache_path = os.path.join(RESULTS_DIR, "emb_cache.pkl")
        self.gen_cache_path = os.path.join(RESULTS_DIR, "gen_cache.pkl")
        self.emb_cache = self._load(self.emb_cache_path)
        self.gen_cache = self._load(self.gen_cache_path)

    @staticmethod
    def _load(path):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def save(self):
        with open(self.emb_cache_path, "wb") as f:
            pickle.dump(self.emb_cache, f)
        with open(self.gen_cache_path, "wb") as f:
            pickle.dump(self.gen_cache, f)

    def embed(self, texts):
        """Embed a list of strings -> np.ndarray (n, 1024), with per-text cache."""
        if isinstance(texts, str):
            texts = [texts]
        out = [None] * len(texts)
        todo, todo_idx = [], []
        for i, t in enumerate(texts):
            h = sha("emb::" + t)
            if h in self.emb_cache:
                out[i] = self.emb_cache[h]
            else:
                todo.append(t)
                todo_idx.append(i)
        batch = 16
        for s in range(0, len(todo), batch):
            chunk = todo[s:s + batch]
            r = requests.post(f"{BASE}/api/embed",
                              json={"model": EMB_MODEL, "input": chunk},
                              timeout=600)
            r.raise_for_status()
            embs = r.json()["embeddings"]
            for j, e in enumerate(embs):
                v = np.asarray(e, dtype=np.float32)
                gi = todo_idx[s + j]
                out[gi] = v
                self.emb_cache[sha("emb::" + chunk[j])] = v
        return np.vstack(out)

    def generate(self, prompt, system=None, temperature=0.0, num_predict=384, model=None):
        mdl = model or LLM_MODEL
        key = sha(json.dumps([mdl, system, prompt, temperature, num_predict],
                             ensure_ascii=False))
        if key in self.gen_cache:
            return self.gen_cache[key]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        r = requests.post(f"{BASE}/api/chat",
                          json={"model": mdl, "messages": messages,
                                "stream": False, "think": False,
                                "options": {"temperature": temperature,
                                            "num_predict": num_predict}},
                          timeout=900)
        r.raise_for_status()
        txt = r.json()["message"]["content"]
        self.gen_cache[key] = txt
        return txt


def cosine_topk(query_vec, mat, k):
    """Return (indices, scores) of top-k rows in mat by cosine similarity."""
    if mat.shape[0] == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = m @ q
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]
