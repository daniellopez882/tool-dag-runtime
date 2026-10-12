from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tooldag.tools.wordpiece import WordPieceTokenizer, basic_tokens, normalize

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "un",
    "##aff",
    "##able",
    "the",
    "heist",
    "##s",
    ",",
    ".",
    "!",
    "cafe",
    "a",
    "##b",
    "我",
]


def tokenizer() -> WordPieceTokenizer:
    return WordPieceTokenizer({t: i for i, t in enumerate(VOCAB)})


def test_normalize_strips_accents_controls_and_spaces_cjk() -> None:
    assert normalize("Café\u0000\tX") == "cafe x"
    assert normalize("a我b") == "a 我 b"
    assert normalize("Café", lowercase=False) == "Café"


def test_basic_tokens_isolate_punctuation() -> None:
    assert basic_tokens("The heists, unaffable!") == ["the", "heists", ",", "unaffable", "!"]


def test_greedy_longest_match_with_continuations() -> None:
    tok = tokenizer()
    assert tok.tokenize("unaffable heists.") == ["un", "##aff", "##able", "heist", "##s", "."]
    assert tok.tokenize("unknownword") == ["[UNK]"]
    assert tok.tokenize("a" * 101) == ["[UNK]"]


def test_encode_adds_specials_and_truncates() -> None:
    tok = tokenizer()
    enc = tok.encode("the heists the heists", max_length=5)
    assert enc.tokens == ["[CLS]", "the", "heist", "##s", "[SEP]"]
    assert enc.ids[0] == VOCAB.index("[CLS]")
    assert enc.ids[-1] == VOCAB.index("[SEP]")


def test_batch_arrays_pad_and_mask() -> None:
    tok = tokenizer()
    ids, mask, types = tok.batch_arrays(["the", "the heists"], max_length=16)
    assert ids.shape == mask.shape == types.shape == (2, 5)
    assert ids.dtype == np.int64
    assert mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert ids[0, 3] == VOCAB.index("[PAD]")
    assert not types.any()


def test_missing_special_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="special token"):
        WordPieceTokenizer({"a": 0})


@given(st.text(alphabet="abuneafl hist.,!我 ", max_size=3000))
def test_windowed_tokenization_equals_whole_text(text: str) -> None:
    tok = tokenizer()
    whole = [p for w in basic_tokens(text) for p in tok.wordpiece(w)]
    assert tok.tokenize(text) == whole
    assert tok.tokenize(text, max_tokens=7) == whole[:7]
