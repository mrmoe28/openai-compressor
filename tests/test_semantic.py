"""Tests for semantic similarity scoring."""

import pytest
from openai_compressor.core import NativeCompressor
from openai_compressor.semantic import score_semantic_similarity, score_message_similarity


class TestSemanticSimilarity:
    def test_identical_text(self):
        text = "Large Language Models have transformed NLP."
        score = score_semantic_similarity(text, text)
        assert 0.99 <= score <= 1.0

    def test_different_text(self):
        text1 = "Machine learning is revolutionizing healthcare."
        text2 = "The weather is sunny today."
        score = score_semantic_similarity(text1, text2)
        assert 0.0 <= score < 0.8

    def test_compressed_preserved(self):
        comp = NativeCompressor()
        text = "Machine learning models require large amounts of training data. " * 50
        compressed = comp.compress_text(text, target_ratio=0.3)
        score = score_semantic_similarity(text, compressed)
        assert score > 0.5

    def test_message_similarity(self):
        from openai_compressor import compress_messages
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Machine learning is transforming the world. " * 50},
        ]
        compressed = compress_messages(messages, target_ratio=0.3)
        score = score_message_similarity(messages, compressed)
        assert 0.5 <= score <= 1.0
