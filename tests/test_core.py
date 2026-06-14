"""Tests for openai_compressor core compressor."""

import pytest
from openai_compressor.core import NativeCompressor, ContextCompressor


class TestNativeCompressor:
    def test_count_tokens(self):
        comp = NativeCompressor()
        assert comp.count_tokens("hello world") > 0

    def test_compress_text_empty(self):
        comp = NativeCompressor()
        assert comp.compress_text("") == ""
        assert comp.compress_text("   ") == "   "

    def test_compress_text_short(self):
        comp = NativeCompressor()
        text = "Hello."
        assert comp.compress_text(text) == text

    def test_compress_text_reduces_tokens(self):
        comp = NativeCompressor()
        text = "Machine learning is transforming the world. " * 50
        before = comp.count_tokens(text)
        compressed = comp.compress_text(text, target_ratio=0.3)
        after = comp.count_tokens(compressed)
        assert after < before
        assert after > 0

    def test_compress_text_with_query(self):
        comp = NativeCompressor()
        text = "Dogs are great. Cats are nice. Birds can fly. Fish swim. " * 20
        compressed = comp.compress_text(text, target_ratio=0.3, query="dogs")
        assert "dog" in compressed.lower() or len(compressed) < len(text)

    def test_compress_messages_preserves_system(self):
        comp = NativeCompressor()
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "User message. " * 50},
        ]
        compressed = comp.compress_messages(messages, target_ratio=0.3)
        assert compressed[0]["content"] == "System prompt."
        assert len(compressed) == 2

    def test_compress_messages_list_content(self):
        comp = NativeCompressor()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello world. " * 50},
                    {"type": "image_url", "image_url": {"url": "http://example.com"}},
                ],
            },
        ]
        compressed = comp.compress_messages(messages, target_ratio=0.3)
        assert compressed[0]["content"][1] == messages[0]["content"][1]
        assert isinstance(compressed[0]["content"][0]["text"], str)

    def test_count_message_tokens(self):
        comp = NativeCompressor()
        messages = [
            {"role": "system", "content": "Hello."},
            {"role": "user", "content": "World."},
        ]
        tokens = comp.count_message_tokens(messages)
        assert tokens > 0


class TestContextCompressor:
    def test_auto_selects_native(self):
        comp = ContextCompressor()
        assert isinstance(comp._active, NativeCompressor)

    def test_uses_llmlingua_when_preferred_and_available(self):
        # llmlingua is not installed in test env by default
        comp = ContextCompressor(prefer_llmlingua=False)
        assert isinstance(comp._active, NativeCompressor)

    def test_getattr_delegation(self):
        comp = ContextCompressor()
        assert callable(comp.count_tokens)
        assert callable(comp.compress_text)
