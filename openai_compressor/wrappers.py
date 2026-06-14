"""
OpenAI client wrappers with auto-compression.
"""

import copy
from typing import Dict, Any

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    raise ImportError("pip install openai")

from .core import ContextCompressor


class _CompressedCompletions:
    """Internal proxy for chat.completions with auto-compression."""

    def __init__(self, raw_completions, compressor: ContextCompressor, config: Dict[str, Any]):
        self._raw = raw_completions
        self._compressor = compressor
        self._config = config

    def _maybe_compress(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        messages = kwargs.get("messages")
        if not messages:
            return kwargs

        total = self._compressor.count_message_tokens(messages)
        threshold = self._config["threshold_tokens"]

        if total <= threshold:
            return kwargs

        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    query = content[:500]
                break

        target = self._config["target_ratio"]
        preserve = self._config["preserve_roles"]

        compressed = self._compressor.compress_messages(
            messages, target_ratio=target, preserve_roles=preserve, query=query
        )
        new_total = self._compressor.count_message_tokens(compressed)

        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs["messages"] = compressed

        if self._config["verbose"]:
            print(f"[COMPRESS] {total} -> {new_total} tokens ({round(total/new_total,1)}x)")

        return new_kwargs

    def create(self, *args, **kwargs):
        kwargs = self._maybe_compress(kwargs)
        return self._raw.create(*args, **kwargs)


class _StreamingCompressedCompletions:
    """Sync streaming proxy — compresses prompt, then yields chunks."""

    def __init__(self, raw_completions, compressor: ContextCompressor, config: Dict[str, Any]):
        self._raw = raw_completions
        self._sync = _CompressedCompletions(raw_completions, compressor, config)

    def create(self, *args, **kwargs):
        kwargs = self._sync._maybe_compress(kwargs)
        for chunk in self._raw.create(*args, **kwargs):
            yield chunk


class CompressedOpenAI:
    """Drop-in wrapper for openai.OpenAI."""

    def __init__(
        self,
        *,
        compression_threshold: float = 0.7,
        target_ratio: float = 0.25,
        preserve_roles=None,
        verbose: bool = True,
        model_context_length: int = 128000,
        prefer_llmlingua: bool = False,
        **openai_kwargs,
    ):
        self._client = OpenAI(**openai_kwargs)
        self._compressor = ContextCompressor(prefer_llmlingua=prefer_llmlingua)
        self._config = {
            "threshold_tokens": int(model_context_length * compression_threshold),
            "target_ratio": target_ratio,
            "preserve_roles": preserve_roles or ["system"],
            "verbose": verbose,
            "model_context_length": model_context_length,
        }
        self._raw_chat = self._client.chat
        class _Chat:
            completions = None
        self.chat = _Chat()
        self.chat.completions = _CompressedCompletions(
            self._raw_chat.completions, self._compressor, self._config
        )

    def __getattr__(self, name: str):
        return getattr(self._client, name)


class _AsyncCompressedCompletions:
    """Async proxy for chat.completions with auto-compression."""

    def __init__(self, raw_completions, compressor: ContextCompressor, config: Dict[str, Any]):
        self._raw = raw_completions
        self._compressor = compressor
        self._config = config

    def _maybe_compress(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        messages = kwargs.get("messages")
        if not messages:
            return kwargs

        total = self._compressor.count_message_tokens(messages)
        threshold = self._config["threshold_tokens"]

        if total <= threshold:
            return kwargs

        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    query = content[:500]
                break

        target = self._config["target_ratio"]
        preserve = self._config["preserve_roles"]

        compressed = self._compressor.compress_messages(
            messages, target_ratio=target, preserve_roles=preserve, query=query
        )
        new_total = self._compressor.count_message_tokens(compressed)

        new_kwargs = copy.deepcopy(kwargs)
        new_kwargs["messages"] = compressed

        if self._config["verbose"]:
            print(f"[COMPRESS] {total} -> {new_total} tokens ({round(total/new_total,1)}x)")

        return new_kwargs

    async def create(self, *args, **kwargs):
        kwargs = self._maybe_compress(kwargs)
        return await self._raw.create(*args, **kwargs)


class _AsyncStreamingCompressedCompletions:
    """Async streaming proxy — compresses prompt, then yields chunks."""

    def __init__(self, raw_completions, compressor: ContextCompressor, config: Dict[str, Any]):
        self._raw = raw_completions
        self._sync = _AsyncCompressedCompletions(raw_completions, compressor, config)

    async def create(self, *args, **kwargs):
        kwargs = self._sync._maybe_compress(kwargs)
        async for chunk in await self._raw.create(*args, **kwargs):
            yield chunk


class CompressedAsyncOpenAI:
    """Drop-in async wrapper for openai.AsyncOpenAI."""

    def __init__(
        self,
        *,
        compression_threshold: float = 0.7,
        target_ratio: float = 0.25,
        preserve_roles=None,
        verbose: bool = True,
        model_context_length: int = 128000,
        prefer_llmlingua: bool = False,
        **openai_kwargs,
    ):
        self._client = AsyncOpenAI(**openai_kwargs)
        self._compressor = ContextCompressor(prefer_llmlingua=prefer_llmlingua)
        self._config = {
            "threshold_tokens": int(model_context_length * compression_threshold),
            "target_ratio": target_ratio,
            "preserve_roles": preserve_roles or ["system"],
            "verbose": verbose,
            "model_context_length": model_context_length,
        }
        self._raw_chat = self._client.chat
        class _Chat:
            pass
        self.chat = _Chat()
        self.chat.completions = _AsyncCompressedCompletions(
            self._raw_chat.completions, self._compressor, self._config
        )

    def __getattr__(self, name: str):
        return getattr(self._client, name)
