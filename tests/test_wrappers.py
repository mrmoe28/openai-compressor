"""Tests for OpenAI client wrappers."""

import pytest
from unittest.mock import MagicMock, patch
from openai_compressor.wrappers import CompressedOpenAI, CompressedAsyncOpenAI


class TestCompressedOpenAI:
    @patch("openai_compressor.wrappers.OpenAI")
    def test_init(self, mock_openai):
        client = CompressedOpenAI(api_key="test-key")
        assert client._config["threshold_tokens"] == int(128000 * 0.7)
        assert client._config["target_ratio"] == 0.25

    @patch("openai_compressor.wrappers.OpenAI")
    def test_compresses_above_threshold(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = {"id": "test"}

        client = CompressedOpenAI(
            api_key="test-key",
            compression_threshold=0.1,
            target_ratio=0.3,
            model_context_length=1000,
        )

        long_text = "Hello world. " * 200
        messages = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": long_text},
        ]

        client.chat.completions.create(model="gpt-4o", messages=messages)
        call_args = mock_client.chat.completions.create.call_args
        sent_messages = call_args.kwargs["messages"]
        assert len(sent_messages) == 2
        assert sent_messages[0]["content"] == "System prompt."

    @patch("openai_compressor.wrappers.OpenAI")
    def test_no_compress_below_threshold(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value = {"id": "test"}

        client = CompressedOpenAI(
            api_key="test-key",
            compression_threshold=1.0,
            model_context_length=100000,
        )

        messages = [
            {"role": "user", "content": "Short message."},
        ]
        client.chat.completions.create(model="gpt-4o", messages=messages)
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["messages"][0]["content"] == "Short message."

    @patch("openai_compressor.wrappers.OpenAI")
    def test_delegates_other_attributes(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.models.list.return_value = ["model1"]

        client = CompressedOpenAI(api_key="test-key")
        result = client.models.list()
        assert result == ["model1"]
