"""
OpenAI-Compatible Auto-Compression Wrapper

Drop-in replacements for openai.OpenAI and openai.AsyncOpenAI
with automatic prompt compression.

Usage:
    from openai_compressor import CompressedOpenAI, CompressedAsyncOpenAI
    from openai_compressor.middleware import CompressionMiddleware
"""

from .core import ContextCompressor, NativeCompressor, LLMLinguaCompressor
from .wrappers import CompressedOpenAI, CompressedAsyncOpenAI
from .helpers import compress_messages, estimate_tokens

__version__ = "0.1.0"
__all__ = [
    "ContextCompressor",
    "NativeCompressor",
    "LLMLinguaCompressor",
    "CompressedOpenAI",
    "CompressedAsyncOpenAI",
    "compress_messages",
    "estimate_tokens",
]
