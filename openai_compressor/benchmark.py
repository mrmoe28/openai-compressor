"""
Benchmark tool for openai-compressor.
"""

import time
import json
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass, asdict

from .core import ContextCompressor

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models require large amounts of training data.",
    "Natural language processing has revolutionized how we interact with computers.",
    "Deep neural networks can learn hierarchical representations of data.",
    "Transformers use self-attention mechanisms to process sequences in parallel.",
    "Prompt engineering is the practice of designing inputs to get desired outputs from LLMs.",
    "Reinforcement learning from human feedback improves model alignment.",
    "Tokenization splits text into subword units for efficient processing.",
    "Fine-tuning adapts pre-trained models to specific downstream tasks.",
    "Retrieval augmented generation combines parametric and non-parametric knowledge.",
    "Large language models exhibit emergent capabilities at scale.",
    "Context windows limit how much information a model can process at once.",
    "Quantization reduces model size by using lower precision weights.",
    "Distillation transfers knowledge from large teacher models to smaller students.",
    "Gradient checkpointing trades computation for memory during training.",
    "KV caching avoids redundant computation during autoregressive generation.",
    "Beam search explores multiple candidate sequences simultaneously.",
    "Temperature controls randomness in probabilistic sampling.",
    "Top-p nucleus sampling dynamically truncates the output distribution.",
    "Constitutional AI trains models to be helpful, harmless, and honest.",
]


def generate_text(target_tokens: int, compressor: ContextCompressor) -> str:
    """Generate synthetic text approximating target token count."""
    tokens_per_sentence = sum(compressor.count_tokens(s) for s in SENTENCES) / len(SENTENCES)
    needed = int(target_tokens / tokens_per_sentence)
    chunks = [random.choice(SENTENCES) for _ in range(needed)]
    return " ".join(chunks)


@dataclass
class BenchmarkResult:
    target_tokens: int
    actual_tokens: int
    compressed_tokens: int
    ratio: float
    compression_time_ms: float
    semantic_score: float


def run_benchmark(
    compressor: ContextCompressor,
    sizes: Optional[Tuple[int, ...]] = None,
    target_ratios: Optional[Tuple[float, ...]] = None,
    iterations: int = 3,
) -> List[BenchmarkResult]:
    sizes = sizes or (1000, 5000, 10000, 30000)
    target_ratios = target_ratios or (0.3, 0.25, 0.2)
    results = []

    for size in sizes:
        text = generate_text(size, compressor)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
        actual = compressor.count_message_tokens(messages)

        for tr in target_ratios:
            compressed = None
            times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                compressed = compressor.compress_messages(messages, target_ratio=tr, query="What is this about?")
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            comp_tokens = compressor.count_message_tokens(compressed) if compressed else 0
            ratio = actual / comp_tokens if comp_tokens else 0
            avg_time = sum(times) / len(times)

            original_words = set(text.lower().split())
            compressed_text = compressed[1]["content"] if compressed else ""
            compressed_words = set(compressed_text.lower().split())
            retention = len(original_words & compressed_words) / len(original_words) if original_words else 0

            results.append(BenchmarkResult(
                target_tokens=size,
                actual_tokens=actual,
                compressed_tokens=comp_tokens,
                ratio=ratio,
                compression_time_ms=avg_time,
                semantic_score=retention,
            ))

    return results


def print_results(results: List[BenchmarkResult]):
    print("\n" + "=" * 90)
    print(f"{'Target':>8} | {'Actual':>8} | {'Compressed':>10} | {'Ratio':>6} | {'Time(ms)':>10} | {'Retention':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r.target_tokens:>8} | {r.actual_tokens:>8} | {r.compressed_tokens:>10} | {r.ratio:>6.1f}x | {r.compression_time_ms:>10.1f} | {r.semantic_score:>10.1%}")
    print("=" * 90)

    ratios = [r.ratio for r in results]
    times = [r.compression_time_ms for r in results]
    retentions = [r.semantic_score for r in results]
    print(f"\nAvg compression ratio: {sum(ratios)/len(ratios):.1f}x")
    print(f"Avg compression time:  {sum(times)/len(times):.1f} ms")
    print(f"Avg word retention:    {sum(retentions)/len(retentions):.1%}")


def export_json(results: List[BenchmarkResult], path: str = "benchmark_results.json"):
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults exported to {path}")


def main():
    print("Loading compressor...")
    compressor = ContextCompressor()
    print("Running benchmarks...")
    results = run_benchmark(compressor, sizes=(1000, 5000, 10000, 30000, 60000), target_ratios=(0.3, 0.25, 0.2))
    print_results(results)
    export_json(results)


if __name__ == "__main__":
    main()
