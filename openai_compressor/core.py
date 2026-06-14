"""
Core compression engine: native TF-IDF + optional LLMLingua backend.
"""

import re
import math
from collections import Counter
from typing import List, Dict, Optional

try:
    import tiktoken
except ImportError:
    raise ImportError("pip install tiktoken")


class NativeCompressor:
    """Pure-Python TF-IDF sentence compressor. No GPU, no heavy models."""

    STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","being","have","has","had",
        "do","does","did","will","would","could","should","may","might","must","shall",
        "can","need","dare","ought","used","to","of","in","for","on","with","at","by",
        "from","as","into","through","during","before","after","above","below","between",
        "under","again","further","then","once","here","there","when","where","why","how",
        "all","each","few","more","most","other","some","such","no","nor","not","only",
        "own","same","so","than","too","very","just","and","but","if","or","because","until",
        "while","what","which","who","whom","this","that","these","those","am","it","its",
        "it's","i","me","my","we","our","you","your","he","him","his","she","her","they",
        "them","their","i'm","i'll","i've","i'd","you're","you'll","you've","you'd","we're",
        "we'll","we've","we'd","they're","they'll","they've","they'd","that's","that'll",
        "there's","there'll","here's","here'll","where's","where'll","when's","when'll",
        "why's","why'll","how's","how'll","what's","what'll","who's","who'll","let's",
        "let'll","ain't","aren't","can't","couldn't","didn't","doesn't","don't","hadn't",
        "hasn't","haven't","isn't","mustn't","needn't","shan't","shouldn't","wasn't",
        "weren't","won't","wouldn't"
    }

    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            total += self.count_tokens(content)
        total += len(messages) * 3
        return total

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    def _sentences(self, text: str) -> List[str]:
        sents = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sents if s.strip() and len(s.strip()) > 10]

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        counts = Counter(t for t in tokens if t not in self.STOPWORDS and len(t) > 2)
        total = len(tokens) or 1
        return {word: count / total for word, count in counts.items()}

    def _compute_idf(self, docs: List[List[str]]) -> Dict[str, float]:
        N = len(docs) or 1
        idf = {}
        all_words = set()
        for doc in docs:
            all_words.update(doc)
        for word in all_words:
            df = sum(1 for doc in docs if word in doc)
            idf[word] = math.log((N + 1) / (df + 1)) + 1
        return idf

    def compress_text(self, text: str, target_ratio: float = 0.3, query: str = "") -> str:
        if not text or not text.strip():
            return text

        sents = self._sentences(text)
        if len(sents) <= 1:
            return text

        sent_tokens = [self._tokenize(s) for s in sents]
        query_tokens = self._tokenize(query) if query else []
        idf = self._compute_idf(sent_tokens)

        scores = []
        for i, tokens in enumerate(sent_tokens):
            tf = self._compute_tf(tokens)
            score = sum(tf.get(w, 0) * idf.get(w, 0) for w in tf)
            if query_tokens:
                overlap = len(set(tokens) & set(query_tokens))
                score += overlap * 0.5
            scores.append((i, score, sents[i]))

        scores.sort(key=lambda x: x[1], reverse=True)

        target_tokens = int(self.count_tokens(text) * target_ratio)
        selected = []
        current_tokens = 0

        for idx, score, sent in scores:
            sent_tokens_count = self.count_tokens(sent)
            if current_tokens + sent_tokens_count <= target_tokens or not selected:
                selected.append((idx, sent))
                current_tokens += sent_tokens_count
            if current_tokens >= target_tokens and len(selected) >= 2:
                break

        selected.sort(key=lambda x: x[0])
        compressed = " ".join(sent for _, sent in selected)

        if self.count_tokens(compressed) < 50 and self.count_tokens(text) > 100:
            target = int(len(sents) * target_ratio) or 1
            compressed = " ".join(sents[:target])

        return compressed

    def compress_messages(
        self,
        messages: List[Dict[str, str]],
        target_ratio: float = 0.3,
        preserve_roles: Optional[List[str]] = None,
        query: str = "",
    ) -> List[Dict[str, str]]:
        preserve_roles = preserve_roles or ["system"]
        compressed = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""

            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text", "")
                        part["text"] = (
                            txt if role in preserve_roles else self.compress_text(txt, target_ratio, query)
                        )
                    new_content.append(part)
                compressed.append({**msg, "content": new_content})
            else:
                compressed.append({
                    **msg,
                    "content": (
                        content if role in preserve_roles else self.compress_text(content, target_ratio, query)
                    ),
                })
        return compressed


class LLMLinguaCompressor:
    """Wrapper around LLMLingua when GPU is available."""

    def __init__(self, model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank", device_map: str = "auto"):
        from llmlingua import PromptCompressor
        import torch
        if device_map == "auto" and not torch.cuda.is_available():
            device_map = "cpu"
        self.compressor = PromptCompressor(model_name=model_name, device_map=device_map)
        self.enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def count_message_tokens(self, messages: List[Dict[str, str]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content") or ""
            if isinstance(content, list):
                text_parts = [
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            total += self.count_tokens(content)
        total += len(messages) * 3
        return total

    def compress_text(self, text: str, target_ratio: float = 0.3, query: str = "") -> str:
        if not text or not text.strip():
            return text
        MAX_MODEL_TOKENS = 256
        tokens = self.count_tokens(text)
        if tokens <= MAX_MODEL_TOKENS:
            try:
                result = self.compressor.compress_prompt(text, question=query if query else None, rate=target_ratio)
                return result.get("compressed_prompt", text)
            except Exception:
                return text
        words = text.split()
        chunks = []
        current_chunk = []
        current_tokens = 0
        for word in words:
            word_tokens = self.count_tokens(word + " ")
            if current_tokens + word_tokens > MAX_MODEL_TOKENS and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_tokens = word_tokens
            else:
                current_chunk.append(word)
                current_tokens += word_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        compressed_chunks = []
        for chunk in chunks:
            try:
                result = self.compressor.compress_prompt(chunk, question=query if query else None, rate=target_ratio)
                compressed_chunks.append(result.get("compressed_prompt", chunk))
            except Exception:
                compressed_chunks.append(chunk)
        return " ".join(compressed_chunks)

    def compress_messages(self, messages, target_ratio=0.3, preserve_roles=None, query=""):
        preserve_roles = preserve_roles or ["system"]
        compressed = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content") or ""
            if isinstance(content, list):
                new_content = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        txt = part.get("text", "")
                        part["text"] = txt if role in preserve_roles else self.compress_text(txt, target_ratio, query)
                    new_content.append(part)
                compressed.append({**msg, "content": new_content})
            else:
                compressed.append({**msg, "content": content if role in preserve_roles else self.compress_text(content, target_ratio, query)})
        return compressed


class ContextCompressor:
    """Auto-selects best available compressor."""

    def __init__(
        self,
        model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
        device_map: str = "auto",
        prefer_llmlingua: bool = False,
    ):
        self._native = NativeCompressor()
        self._llmlingua = None
        if prefer_llmlingua:
            try:
                self._llmlingua = LLMLinguaCompressor(model_name=model_name, device_map=device_map)
            except Exception:
                pass
        self._active = self._llmlingua if self._llmlingua else self._native

    def __getattr__(self, name):
        return getattr(self._active, name)
