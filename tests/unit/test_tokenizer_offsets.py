"""Unit tests for tokenizer token offsets."""

from abctokz.tokenizer import AugenblickTokenizer
from abctokz.models.bpe import BPEModel
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.vocab import Vocabulary


def test_tokenizer_offsets_align_with_tokens() -> None:
    vocab = Vocabulary({"<unk>": 0, "a": 1, "##b": 2})
    model = BPEModel(vocab, MergeTable([]))
    tokenizer = AugenblickTokenizer(model)

    enc = tokenizer.encode("ab")
    assert enc.tokens == ["a", "##b"]
    assert enc.offsets == [(0, 1), (1, 2)]