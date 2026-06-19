"""drift/text — Text utilities (chunking, similarity, tokenization).

Lightweight implementations only. For real embedding-based work, swap in
a provider via drift.text.embed = your_function.
"""
import math
import re
from typing import Callable


def chunk(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks suitable for an LLM context window."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap >= max_chars:
        overlap = max_chars // 2
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i:i + max_chars])
        i += max_chars - overlap
    return out


def tokenize(text: str) -> list[str]:
    """Cheap word tokenizer. For real token counts, use the provider's tokenizer."""
    return re.findall(r"\w+", text.lower())


def similarity(a: str, b: str) -> float:
    """Jaccard similarity over word tokens. 0.0–1.0."""
    tok_a, tok_b = set(tokenize(a)), set(tokenize(b))
    if not tok_a and not tok_b:
        return 1.0
    intersection = tok_a & tok_b
    union = tok_a | tok_b
    return len(intersection) / len(union) if union else 0.0


def embed(text: str) -> list[float]:
    """Override this function to wire a real embedding model.

    The default raises so users notice when they're trying to do semantic
    work without configuring a backend."""
    raise NotImplementedError(
        "drift.text.embed is a stub. Assign drift.text.embed = your_fn to wire "
        "an embedding provider (OpenAI, Anthropic via Voyage, sentence-transformers, etc)."
    )


__all__ = ["chunk", "tokenize", "similarity", "embed"]
