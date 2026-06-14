"""
Semantic similarity scoring using sentence-transformers.

Install with: pip install openai-compressor[semantic]
"""

from typing import Optional
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False


def _ensure_model():
    if not _SEMANTIC_AVAILABLE:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install openai-compressor[semantic]"
        )


_model: Optional["SentenceTransformer"] = None


def get_model() -> "SentenceTransformer":
    global _model
    _ensure_model()
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def score_semantic_similarity(original: str, compressed: str) -> float:
    """
    Return cosine similarity between original and compressed text embeddings.
    Range: -1 to 1, where 1 = identical meaning, 0 = orthogonal, -1 = opposite.
    """
    _ensure_model()
    model = get_model()
    emb1 = model.encode(original, convert_to_numpy=True)
    emb2 = model.encode(compressed, convert_to_numpy=True)
    return (cosine_similarity(emb1, emb2) + 1) / 2  # Normalize to 0-1


def score_message_similarity(original_messages: list, compressed_messages: list) -> float:
    """
    Score semantic similarity between original and compressed message lists.
    Compares only user/assistant content (skips system).
    """
    _ensure_model()
    model = get_model()
    orig_text = " ".join(
        m.get("content", "") for m in original_messages
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    )
    comp_text = " ".join(
        m.get("content", "") for m in compressed_messages
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    )
    if not orig_text.strip() or not comp_text.strip():
        return 0.0
    emb1 = model.encode(orig_text, convert_to_numpy=True)
    emb2 = model.encode(comp_text, convert_to_numpy=True)
    return (cosine_similarity(emb1, emb2) + 1) / 2
