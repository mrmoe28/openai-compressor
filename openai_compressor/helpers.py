"""
Standalone helpers for one-shot compression without wrapping the client.
"""

from typing import List, Dict, Optional

try:
    import tiktoken
except ImportError:
    raise ImportError("pip install tiktoken")

from .core import ContextCompressor


def compress_messages(
    messages: List[Dict[str, str]],
    target_ratio: float = 0.25,
    preserve_roles: Optional[List[str]] = None,
    query: str = "",
    prefer_llmlingua: bool = False,
) -> List[Dict[str, str]]:
    """One-shot message compression."""
    comp = ContextCompressor(prefer_llmlingua=prefer_llmlingua)
    return comp.compress_messages(messages, target_ratio, preserve_roles, query)


def estimate_tokens(messages: List[Dict[str, str]]) -> int:
    """Quick token count without compression overhead."""
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            content = "\n".join(text_parts)
        total += len(enc.encode(content))
    total += len(messages) * 3
    return total
