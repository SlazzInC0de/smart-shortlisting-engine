"""Semantic matching: local sentence-transformer embeddings + cosine similarity.

Captures contextual similarity even without literal keyword overlap.
Includes batched document inference and MD5 in-memory caching for high performance.
"""

import hashlib
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL_NAME

_model = None
_embedding_cache = {}

# Limit thread contention on Windows CPU
try:
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
except Exception:
    pass


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _chunk_text(text: str, chunk_words: int = 140, overlap_words: int = 25) -> list[str]:
    """Chunk text along paragraph or sentence boundaries with overlap."""
    if not text or not text.strip():
        return [""]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current_words = []

    for para in paragraphs:
        p_words = para.split()
        if len(current_words) + len(p_words) <= chunk_words:
            current_words.extend(p_words)
        else:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap_words:] if overlap_words > 0 else []
            if len(p_words) > chunk_words:
                step = max(1, chunk_words - overlap_words)
                for i in range(0, len(p_words), step):
                    sub = p_words[i:i + chunk_words]
                    if sub:
                        chunks.append(" ".join(sub))
                current_words = []
            else:
                current_words.extend(p_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks or [text.strip()]


def embed_document(text: str) -> np.ndarray:
    """Embed a single document by mean-pooling chunk embeddings with L2 normalization."""
    cleaned = (text or "").strip()
    cache_key = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    model = get_model()
    chunks = _chunk_text(cleaned)
    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    mean_vec = chunk_embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm

    _embedding_cache[cache_key] = mean_vec
    return mean_vec


def embed_documents_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed multiple documents efficiently using batched tensor encoding with caching."""
    model = get_model()
    results = [None] * len(texts)
    uncached_info = []  # (index, cache_key)
    uncached_chunks = []
    chunk_slices = []

    for idx, text in enumerate(texts):
        cleaned = (text or "").strip()
        cache_key = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
        if cache_key in _embedding_cache:
            results[idx] = _embedding_cache[cache_key]
        else:
            uncached_info.append((idx, cache_key))
            chunks = _chunk_text(cleaned)
            start = len(uncached_chunks)
            uncached_chunks.extend(chunks)
            chunk_slices.append((start, len(uncached_chunks)))

    if uncached_chunks:
        all_embeddings = model.encode(
            uncached_chunks,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for (idx, cache_key), (start, end) in zip(uncached_info, chunk_slices):
            doc_chunks = all_embeddings[start:end]
            mean_vec = doc_chunks.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            _embedding_cache[cache_key] = mean_vec
            results[idx] = mean_vec

    return results


def semantic_score(jd_embedding: np.ndarray, resume_embedding: np.ndarray) -> float:
    """Cosine similarity between JD and resume embeddings, clamped to [0, 1]."""
    sim = cosine_similarity(
        jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1)
    )[0][0]
    return float(max(0.0, min(1.0, sim)))
