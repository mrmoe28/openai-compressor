"""
FastAPI middleware for transparent prompt compression.
"""

import json
from typing import List, Dict, Optional, Any

from .core import ContextCompressor


class CompressionMiddleware:
    """
    FastAPI ASGI middleware that intercepts outbound LLM request bodies
    and compresses prompt content before forwarding.

    Usage:
        from fastapi import FastAPI
        from openai_compressor.middleware import CompressionMiddleware

        app = FastAPI()
        app.add_middleware(
            CompressionMiddleware,
            llm_endpoints=["/v1/chat/completions"],
            compression_threshold=0.7,
            target_ratio=0.25,
            model_context_length=128000,
        )
    """

    def __init__(
        self,
        app,
        llm_endpoints: Optional[List[str]] = None,
        compression_threshold: float = 0.7,
        target_ratio: float = 0.25,
        preserve_roles: Optional[List[str]] = None,
        verbose: bool = True,
        model_context_length: int = 128000,
        prefer_llmlingua: bool = False,
    ):
        self.app = app
        self.endpoints = set(llm_endpoints or ["/v1/chat/completions"])
        self.compressor = ContextCompressor(prefer_llmlingua=prefer_llmlingua)
        self.config = {
            "threshold_tokens": int(model_context_length * compression_threshold),
            "target_ratio": target_ratio,
            "preserve_roles": preserve_roles or ["system"],
            "verbose": verbose,
            "model_context_length": model_context_length,
        }

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        if path not in self.endpoints or method != "POST":
            await self.app(scope, receive, send)
            return

        # Capture body
        body_parts = []
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                body_parts.append(msg.get("body", b""))
                if not msg.get("more_body", False):
                    break
            else:
                break

        raw_body = b"".join(body_parts)
        body_str = raw_body.decode("utf-8", errors="replace")

        # Parse and compress
        modified = False
        try:
            data = json.loads(body_str)
            messages = data.get("messages")
            if messages and isinstance(messages, list):
                total = self.compressor.count_message_tokens(messages)
                if total > self.config["threshold_tokens"]:
                    query = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            content = msg.get("content") or ""
                            if isinstance(content, str):
                                query = content[:500]
                            break

                    compressed = self.compressor.compress_messages(
                        messages,
                        target_ratio=self.config["target_ratio"],
                        preserve_roles=self.config["preserve_roles"],
                        query=query,
                    )
                    new_total = self.compressor.count_message_tokens(compressed)
                    data["messages"] = compressed
                    modified = True

                    if self.config["verbose"]:
                        print(f"[MW-COMPRESS] {total} -> {new_total} tokens ({round(total/new_total,1)}x)")

            if modified:
                raw_body = json.dumps(data, separators=(",", ":")).encode("utf-8")
                scope["headers"] = [
                    (k, v) for k, v in scope.get("headers", [])
                    if k.lower() != b"content-length"
                ]
        except (json.JSONDecodeError, Exception):
            pass

        # Rebuild receive
        body_exhausted = False

        async def new_receive():
            nonlocal body_exhausted
            if not body_exhausted:
                body_exhausted = True
                return {"type": "http.request", "body": raw_body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, new_receive, send)
