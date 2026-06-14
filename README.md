# OpenAI Compressor

Auto-compress prompts for OpenAI API calls. Zero GPU dependency. CPU-native TF-IDF scoring.

## Quick Start

```bash
pip install openai-compressor
```

```python
from openai_compressor import CompressedOpenAI

client = CompressedOpenAI(api_key="sk-...")

# Prompts are automatically compressed when they exceed 70% of context window
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Very long document here..." * 500},
    ],
)
# [COMPRESS] 15234 -> 4120 tokens (3.7x)
```

## Features

| Feature | Status |
|---------|--------|
| Sync OpenAI wrapper | [OK] |
| Async OpenAI wrapper | [OK] |
| Streaming (sync + async) | [OK] |
| FastAPI middleware | [OK] |
| CPU-native compressor | [OK] |
| Optional LLMLingua backend | [OK] |
| Semantic similarity scoring | [OK] |
| Benchmark tool | [OK] |

## Install Options

```bash
# Base (CPU only)
pip install openai-compressor

# With FastAPI middleware
pip install openai-compressor[fastapi]

# With LLMLingua GPU backend
pip install openai-compressor[llmlingua]

# With semantic similarity scoring
pip install openai-compressor[semantic]

# Everything
pip install openai-compressor[all]
```

## Usage

### Standalone compression

```python
from openai_compressor import compress_messages, estimate_tokens

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Very long text..."},
]

print(estimate_tokens(messages))  # quick token count
compressed = compress_messages(messages, target_ratio=0.3)
```

### Async

```python
from openai_compressor import CompressedAsyncOpenAI

client = CompressedAsyncOpenAI(api_key="sk-...")
resp = await client.chat.completions.create(model="gpt-4o", messages=[...])
```

### FastAPI Middleware

```python
from fastapi import FastAPI
from openai_compressor.middleware import CompressionMiddleware

app = FastAPI()
app.add_middleware(
    CompressionMiddleware,
    llm_endpoints=["/v1/chat/completions"],
    compression_threshold=0.7,
    target_ratio=0.25,
)
```

### Semantic Scoring

```python
from openai_compressor.semantic import score_semantic_similarity

score = score_semantic_similarity(original, compressed)
print(f"Semantic retention: {score:.2%}")
```

## Benchmark

```bash
openai-compressor-benchmark
```

Or programmatically:

```python
from openai_compressor.benchmark import run_benchmark, print_results

results = run_benchmark()
print_results(results)
```

## License

MIT
