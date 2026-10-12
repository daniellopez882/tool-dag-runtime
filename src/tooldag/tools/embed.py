"""Sentence embeddings: MiniLM through onnxruntime, plus a hashing embedder for tests.

The ONNX session is created with one intra-op thread and shared by every worker
thread (``InferenceSession.run`` is thread-safe). Parallelism then comes from the
scheduler running several embedding tasks at once, which is what the makespan
benchmark measures; see docs/adr/0002.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from tooldag.tools.wordpiece import WordPieceTokenizer, basic_tokens

if TYPE_CHECKING:
    from pathlib import Path

DIM = 384
MAX_LENGTH = 256


@dataclass(frozen=True, slots=True, eq=False)
class Vector:
    """A unit-length embedding, or ``values=None`` when there was no text to embed."""

    values: np.ndarray | None

    @property
    def empty(self) -> bool:
        return self.values is None

    def cosine(self, other: Vector) -> float | None:
        if self.values is None or other.values is None:
            return None
        return float(np.dot(self.values, other.values))

    def preview(self) -> dict[str, Any]:
        return {"dim": DIM, "empty": self.empty}

    def __str__(self) -> str:
        return "<empty vector>" if self.empty else f"<vector dim={DIM}>"


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> Vector: ...


def _unit(row: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(row))
    return (row / norm).astype(np.float32) if norm > 0 else row.astype(np.float32)


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 (int8 ONNX export) with the from-scratch WordPiece tokenizer."""

    def __init__(self, model_dir: Path, *, intra_op_threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / "model_quantized.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = WordPieceTokenizer.from_vocab_file(model_dir / "vocab.txt")
        self.intra_op_threads = intra_op_threads
        self.name = f"all-MiniLM-L6-v2 int8 ONNX (intra-op threads={intra_op_threads})"

    def embed(self, text: str) -> Vector:
        if not text.strip():
            return Vector(None)
        ids, mask, types = self.tokenizer.batch_arrays([text], MAX_LENGTH)
        (hidden,) = self.session.run(
            ["last_hidden_state"],
            {"input_ids": ids, "attention_mask": mask, "token_type_ids": types},
        )
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)
        return Vector(_unit(pooled[0]))


class HashingEmbedder:
    """Signed feature hashing of lowercase word tokens into 384 dimensions.

    Deterministic and dependency-free; used by the test suite and the offline demo when
    the MiniLM files are not present. It captures word overlap only.
    """

    name = "feature hashing (384 dims, word tokens)"

    def embed(self, text: str) -> Vector:
        tokens = [t for t in basic_tokens(text) if t.isalnum()]
        if not tokens:
            return Vector(None)
        acc = np.zeros(DIM, dtype=np.float64)
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % DIM
            acc[bucket] += 1.0 if digest[4] & 1 else -1.0
        return Vector(_unit(acc))


class CachingEmbedder:
    """Memoises another embedder by text. Used for accuracy runs, never for timing runs."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = f"{inner.name} (cached)"
        self._cache: dict[str, Vector] = {}
        self._lock = threading.Lock()
        self.misses = 0

    def embed(self, text: str) -> Vector:
        key = hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        vector = self.inner.embed(text)
        with self._lock:
            self._cache[key] = vector
            self.misses += 1
        return vector
