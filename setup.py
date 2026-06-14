import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="openai-compressor",
    version="0.1.0",
    description="Auto-compress prompts for OpenAI API calls with zero GPU dependency",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Eko Solar Ops",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "tiktoken",
    ],
    extras_require={
        "fastapi": ["fastapi", "uvicorn"],
        "llmlingua": ["llmlingua", "torch"],
        "dev": ["pytest", "pytest-asyncio"],
        "semantic": ["sentence-transformers>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "openai-compressor-benchmark=benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
