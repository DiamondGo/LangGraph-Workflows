"""Group items reporting the same event, using local CPU embeddings.

LLM grouping was tried first: without thinking the local Qwen lumped unrelated stories
together, and with thinking it grouped by broad theme and took ~200s per topic.
"""

from functools import cache

import numpy as np
from fastembed import TextEmbedding

from .config import CLUSTER_THRESHOLD, DATA_DIR, EMBED_MODEL


@cache
def _model() -> TextEmbedding:
    return TextEmbedding(EMBED_MODEL, cache_dir=str(DATA_DIR / "models"))


def embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(_model().embed(texts)))
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def average_linkage(vectors: np.ndarray, threshold: float = CLUSTER_THRESHOLD) -> list[list[int]]:
    """Merge the most similar pair of clusters until no pair averages above `threshold`.

    Average linkage avoids chaining loosely related stories into one big group.
    """
    n = len(vectors)
    if n == 0:
        return []
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)
    members = {i: [i] for i in range(n)}
    active = np.ones(n, dtype=bool)
    while True:
        masked = np.where(active[:, None] & active[None, :], sim, -np.inf)
        a, b = np.unravel_index(np.argmax(masked), masked.shape)
        if masked[a, b] < threshold:
            break
        na, nb = len(members[a]), len(members[b])
        sim[a, :] = (na * sim[a, :] + nb * sim[b, :]) / (na + nb)
        sim[:, a] = sim[a, :]
        sim[a, a] = -np.inf
        members[a] += members.pop(b)
        active[b] = False
    return list(members.values())


def group_items(items: list[dict]) -> list[list[dict]]:
    if not items:
        return []
    vectors = embed([f"{i['title']}. {i['summary'][:200]}" for i in items])
    return [[items[k] for k in group] for group in average_linkage(vectors)]
