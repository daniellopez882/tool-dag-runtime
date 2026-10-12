"""Download the benchmark data and the embedding model, verified against pinned sha256.

The checksums below were recorded on the first download (2026-10-11). Every fetch
streams to a ``.part`` file, checks size and sha256, and only then renames it into
place; a mismatch deletes the partial file and raises.
"""

from __future__ import annotations

import hashlib
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

BIGBENCH_COMMIT = "092b196c1f8f14a54bbc62f24759d43bde46dd3b"
MINILM_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
_MINILM = f"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/{MINILM_REVISION}"


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    relpath: str
    url: str
    sha256: str
    size: int
    license: str
    purpose: str


ASSETS: tuple[Asset, ...] = (
    Asset(
        name="bigbench-movie-recommendation",
        relpath="raw/bigbench/movie_recommendation/task.json",
        url=(
            "https://raw.githubusercontent.com/google/BIG-bench/"
            f"{BIGBENCH_COMMIT}/bigbench/benchmark_tasks/movie_recommendation/task.json"
        ),
        sha256="94e9b3bafa38735d5d44bdca5449314cd7241b6ed0bbb83d8d061efb573d2b46",
        size=132_837,
        license="Apache-2.0",
        purpose="500 multiple-choice questions (benchmark)",
    ),
    Asset(
        name="movielens-latest-small",
        relpath="raw/movielens/ml-latest-small.zip",
        url="https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
        sha256="696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
        size=978_202,
        license="MovieLens usage licence (research use, no redistribution)",
        purpose="titles, genres and ratings for resolution and co-occurrence",
    ),
    Asset(
        name="cmu-movie-summaries",
        relpath="raw/cmu/MovieSummaries.tar.gz",
        url="http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz",
        sha256="6c3bd3d42046c4b87e2267f2e2599ba6f651b8ad23a8998237c3d2bb338ac559",
        size=48_002_242,
        license="CC BY-SA (CMU Movie Summary Corpus)",
        purpose="Wikipedia plot summaries for the embedding signal",
    ),
    Asset(
        name="minilm-onnx-quantized",
        relpath="models/all-MiniLM-L6-v2/model_quantized.onnx",
        url=f"{_MINILM}/onnx/model_quantized.onnx",
        sha256="afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
        size=22_972_370,
        license="Apache-2.0",
        purpose="sentence embeddings (int8 ONNX export of all-MiniLM-L6-v2)",
    ),
    Asset(
        name="minilm-vocab",
        relpath="models/all-MiniLM-L6-v2/vocab.txt",
        url=f"{_MINILM}/vocab.txt",
        sha256="07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
        size=231_508,
        license="Apache-2.0",
        purpose="WordPiece vocabulary read by the from-scratch tokenizer",
    ),
    Asset(
        name="minilm-tokenizer-json",
        relpath="models/all-MiniLM-L6-v2/tokenizer.json",
        url=f"{_MINILM}/tokenizer.json",
        sha256="da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
        size=711_661,
        license="Apache-2.0",
        purpose="reference tokenizer for the WordPiece parity test only",
    ),
)

ASSETS_BY_NAME = {asset.name: asset for asset in ASSETS}


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_path(root: Path, name: str) -> Path:
    return root / ASSETS_BY_NAME[name].relpath


def is_present(root: Path, asset: Asset, *, verify: bool = True) -> bool:
    path = root / asset.relpath
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    return not verify or sha256_of(path) == asset.sha256


def _download(asset: Asset, dest: Path, opener: Callable[[str], object], attempts: int) -> None:
    partial = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        digest = hashlib.sha256()
        size = 0
        try:
            with opener(asset.url) as response, partial.open("wb") as out:  # type: ignore[attr-defined]
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    digest.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            time.sleep(min(2.0**attempt, 10.0))
            continue
        got = digest.hexdigest()
        if size != asset.size or got != asset.sha256:
            partial.unlink(missing_ok=True)
            raise ChecksumError(
                f"{asset.name}: got {size} bytes sha256 {got}; "
                f"expected {asset.size} bytes sha256 {asset.sha256}"
            )
        partial.replace(dest)
        return
    raise RuntimeError(f"{asset.name}: download failed after {attempts} attempts: {last_error}")


def _urlopen(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "tool-dag-runtime-fetch/0.1"})
    return urllib.request.urlopen(request, timeout=120)


def fetch(
    root: Path,
    names: Iterable[str] | None = None,
    *,
    opener: Callable[[str], object] = _urlopen,
    attempts: int = 3,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Fetch the named assets (all by default) into ``root``; skip ones already verified."""
    selected = [ASSETS_BY_NAME[n] for n in names] if names else list(ASSETS)
    paths: list[Path] = []
    for asset in selected:
        dest = root / asset.relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if is_present(root, asset):
            log(f"ok       {asset.name} ({asset.size:,} bytes, sha256 verified)")
        else:
            log(f"fetching {asset.name} from {asset.url}")
            _download(asset, dest, opener, attempts)
            log(f"ok       {asset.name} ({asset.size:,} bytes, sha256 verified)")
        paths.append(dest)
    return paths
