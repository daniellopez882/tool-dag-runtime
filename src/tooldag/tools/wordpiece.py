"""BERT WordPiece tokenizer written from scratch (uncased, as used by all-MiniLM-L6-v2).

Pipeline, matching the reference ``BertNormalizer`` + ``BertPreTokenizer`` + ``WordPiece``:

1. clean: drop NUL, U+FFFD and control characters; map whitespace to a space;
2. put spaces around CJK ideographs;
3. strip accents (NFD, drop non-spacing marks) and lowercase;
4. split on whitespace and isolate every punctuation character;
5. greedy longest-match-first WordPiece with ``##`` continuations; words longer than
   100 characters, or with any unmatched remainder, become ``[UNK]``.

Long inputs are processed window by window and stop once ``max_length`` tokens exist,
because a plot summary can run to thousands of words and only the first 254 pieces
are kept.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from pathlib import Path

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B920, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)
_WINDOW_CHARS = 1024


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def _is_whitespace(ch: str) -> bool:
    return ch in "\t\n\r" or ch.isspace()


def _is_cjk(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def normalize(text: str, *, lowercase: bool = True) -> str:
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or _is_control(ch):
            continue
        if _is_whitespace(ch):
            out.append(" ")
        elif cp >= 0x3400 and _is_cjk(cp):
            out.extend((" ", ch, " "))
        else:
            out.append(ch)
    cleaned = "".join(out)
    if not lowercase:
        return cleaned
    decomposed = unicodedata.normalize("NFD", cleaned)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return stripped.lower()


def pre_tokenize(normalized: str) -> list[str]:
    words: list[str] = []
    for chunk in normalized.split(" "):
        if not chunk:
            continue
        current: list[str] = []
        for ch in chunk:
            if _is_punctuation(ch):
                if current:
                    words.append("".join(current))
                    current = []
                words.append(ch)
            elif ch.isspace():
                # Whitespace the cleaning step did not map (none expected) still splits.
                if current:
                    words.append("".join(current))
                    current = []
            else:
                current.append(ch)
        if current:
            words.append("".join(current))
    return words


def basic_tokens(text: str, *, lowercase: bool = True) -> list[str]:
    return pre_tokenize(normalize(text, lowercase=lowercase))


@dataclass(frozen=True, slots=True)
class Encoding:
    ids: list[int]
    tokens: list[str]


class WordPieceTokenizer:
    def __init__(
        self,
        vocab: Mapping[str, int],
        *,
        lowercase: bool = True,
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        continuing_prefix: str = "##",
        max_input_chars_per_word: int = 100,
    ) -> None:
        self.vocab = dict(vocab)
        self.lowercase = lowercase
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.prefix = continuing_prefix
        self.max_word_chars = max_input_chars_per_word
        for token in (unk_token, cls_token, sep_token, pad_token):
            if token not in self.vocab:
                raise ValueError(f"vocabulary lacks special token {token!r}")
        self.unk_id = self.vocab[unk_token]
        self.cls_id = self.vocab[cls_token]
        self.sep_id = self.vocab[sep_token]
        self.pad_id = self.vocab[pad_token]
        self._cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_vocab_file(cls, path: Path, **kwargs: object) -> WordPieceTokenizer:
        tokens = path.read_text(encoding="utf-8").split("\n")
        if tokens and tokens[-1] == "":
            tokens.pop()
        vocab = {token.rstrip("\r"): i for i, token in enumerate(tokens)}
        return cls(vocab, **kwargs)  # type: ignore[arg-type]

    def wordpiece(self, word: str) -> tuple[str, ...]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        pieces: list[str] = []
        if len(word) > self.max_word_chars:
            pieces = [self.unk_token]
        else:
            start = 0
            while start < len(word):
                end = len(word)
                found = None
                while start < end:
                    piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                    if piece in self.vocab:
                        found = piece
                        break
                    end -= 1
                if found is None:
                    pieces = [self.unk_token]
                    break
                pieces.append(found)
                start = end
        result = tuple(pieces)
        if len(self._cache) < 200_000:
            self._cache[word] = result
        return result

    def _words(self, text: str) -> Iterator[str]:
        start = 0
        while start < len(text):
            end = min(start + _WINDOW_CHARS, len(text))
            # Only cut at whitespace so no word or combining sequence is split.
            while end < len(text) and not _is_whitespace(text[end]):
                end += 1
            yield from basic_tokens(text[start:end], lowercase=self.lowercase)
            start = end

    def tokenize(self, text: str, max_tokens: int | None = None) -> list[str]:
        out: list[str] = []
        for word in self._words(text):
            out.extend(self.wordpiece(word))
            if max_tokens is not None and len(out) >= max_tokens:
                return out[:max_tokens]
        return out

    def encode(self, text: str, max_length: int = 256) -> Encoding:
        """[CLS] tokens [SEP], truncated so the whole sequence fits ``max_length``."""
        tokens = self.tokenize(text, max_tokens=max_length - 2)
        full = [self.cls_token, *tokens, self.sep_token]
        return Encoding([self.vocab.get(t, self.unk_id) for t in full], full)

    def batch_arrays(
        self, texts: Sequence[str], max_length: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Padded ``input_ids``, ``attention_mask`` and ``token_type_ids`` (int64)."""
        encodings = [self.encode(t, max_length).ids for t in texts]
        width = max(len(e) for e in encodings)
        ids = np.full((len(encodings), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, enc in enumerate(encodings):
            ids[row, : len(enc)] = enc
            mask[row, : len(enc)] = 1
        return ids, mask, np.zeros_like(ids)
