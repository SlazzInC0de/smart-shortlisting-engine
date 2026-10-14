"""Replace semantic_match with a deterministic bag-of-words embedder.

The real module loads torch and downloads a model. Tests only need a stable
vector whose cosine similarity rises with word overlap, so pipeline logic
(parsing, scoring, ranking, explanations) runs in well under a second.
"""
import re
import sys
import types

import numpy as np

DIM = 512


def _embed(text: str) -> np.ndarray:
    vec = np.zeros(DIM)
    for word in re.findall(r"[a-z0-9+#]+", text.lower()):
        vec[hash(word) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _semantic_score(a: np.ndarray, b: np.ndarray) -> float:
    return float(max(0.0, min(1.0, float(a @ b))))


fake = types.ModuleType("semantic_match")
fake.embed_document = _embed
fake.embed_documents_batch = lambda texts: [_embed(t) for t in texts]
fake.semantic_score = _semantic_score
sys.modules.setdefault("semantic_match", fake)
