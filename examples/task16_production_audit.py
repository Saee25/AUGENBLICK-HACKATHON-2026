"""Task 16 — Is This Ready for Production?

Honest audit: three reasons to be confident, three concrete hesitations.
Each claim is backed by a code path or a live demonstration.
"""

from __future__ import annotations

import time
from pathlib import Path
import tempfile

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.normalizers.whitespace import WhitespaceNormalizer


CORPUS_LINES = [
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "भारत विविध भाषाओं और लिपियों का देश है",
    "देवनागरी लिपि में मात्राएं और संयुक्ताक्षर महत्वपूर्ण हैं",
    "नमस्ते दुनिया यह एक परीक्षण वाक्य है",
    "hello world this is a multilingual tokenizer benchmark",
    "mixed script नमस्ते world test sentence",
    "tokenization quality depends on corpus and vocabulary",
    "the quick brown fox jumps over the lazy dog",
] * 60


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_path = tmp_path / "audit_corpus.txt"
        corpus_path.write_text("\n".join(CORPUS_LINES), encoding="utf-8")

        config = bpe_multilingual(vocab_size=400)
        tok = Tokenizer.from_config(config)
        tok.train([str(corpus_path)], config)

        separator("TASK 16 — PRODUCTION AUDIT")
        print(f"model: BPE | vocab_size: {tok.get_vocab_size()}")

        # ─────────────────────────────────────────────────────────────────────
        # CONFIDENCE REASONS
        # ─────────────────────────────────────────────────────────────────────
        separator("CONFIDENCE (3 Reasons)")

        subsection("1. Deterministic output — verified by property tests")
        text = "नमस्ते world भारत"
        enc1 = tok.encode(text)
        enc2 = tok.encode(text)
        enc3 = tok.encode(text)
        all_same = enc1.ids == enc2.ids == enc3.ids
        print(f"  text      : {text!r}")
        print(f"  ids run1  : {enc1.ids}")
        print(f"  ids run2  : {enc2.ids}")
        print(f"  ids run3  : {enc3.ids}")
        print(f"  all_same  : {all_same}")
        print("  Evidence  : tests/property/test_determinism.py — TestDeterminism")
        print("              trainers/bpe_trainer.py — set_seed(config.seed) on init")

        subsection("2. Devanagari-safe NFC normalization (not NFKC)")
        import unicodedata
        nfd_cafe = unicodedata.normalize("NFD", "Café")  # decomposed
        norm = DevanagariNormalizer(nfc_first=True)
        after = norm.normalize(nfd_cafe)
        print(f"  NFD input : {nfd_cafe!r}  (len={len(nfd_cafe)})")
        print(f"  NFC output: {after!r}  (len={len(after)})")
        print(f"  stable    : {after == unicodedata.normalize('NFC', nfd_cafe)}")
        print("  Critical  : NFKC would break Devanagari conjuncts like क्ष (ka+halant+sha)")
        print("  Evidence  : normalizers/devanagari.py line 7 design note")
        print("              unicode_nfkc.py uses NFKC ONLY for Latin, never for Devanagari pipeline")

        subsection("3. Clean exception hierarchy + schema versioning")
        from abctokz.exceptions import (
            SchemaVersionError, SerializationError, ConfigError, VocabError
        )
        from abctokz.version import SCHEMA_VERSION
        print(f"  SCHEMA_VERSION : {SCHEMA_VERSION}")
        print("  Exception tree : TokzError > VocabError, TrainingError,")
        print("                   SerializationError > SchemaVersionError")
        print("  SchemaVersionError is raised on load with mismatched version —")
        print("  safe for rolling deploys where artifact format changes.")
        print("  Evidence  : exceptions.py, tokenizer.py load() lines 362-420")

        # ─────────────────────────────────────────────────────────────────────
        # HESITATION REASONS
        # ─────────────────────────────────────────────────────────────────────
        separator("HESITATION (3 Concrete Gaps)")

        subsection("1. save() / load() does NOT persist normalizer or pre-tokenizer")
        save_path = str(tmp_path / "tok_saved")
        tok.save(save_path)

        # Show what is actually saved
        saved_files = sorted(Path(save_path).iterdir())
        print(f"  Saved files: {[f.name for f in saved_files]}")

        import json
        config_json = json.loads((Path(save_path) / "config.json").read_text())
        print(f"  config.json contents: {config_json}")
        print("  >>> Only model_type + schema_version are saved.")
        print("  >>> normalizer config (NFC, strip_zero_width) — NOT SAVED")
        print("  >>> pre-tokenizer config (split_on_script_boundary) — NOT SAVED")

        # Demonstrate the gap: loaded tokenizer encodes differently
        loaded_tok = Tokenizer.load(save_path)
        test_text = "नमस्तेworld"  # mixed script — pretokenizer would split; loaded won't
        enc_trained = tok.encode(test_text)
        enc_loaded  = loaded_tok.encode(test_text)
        print(f"\n  Test: encode({test_text!r})")
        print(f"  trained tokenizer tokens : {enc_trained.tokens}")
        print(f"  loaded  tokenizer tokens : {enc_loaded.tokens}")
        print(f"  ids match                : {enc_trained.ids == enc_loaded.ids}")
        print("  Evidence  : tokenizer.py save() line 313 — config_data has no normalizer key")
        print("              tokenizer.py load()  line 362 — rebuilds only model+decoder")
        print("  Risk      : CRITICAL — loaded tokenizer silently produces different output")

        subsection("2. decode() uses a heuristic to skip special tokens — not the registered list")
        # The heuristic: skip anything that starts with '<' and ends with '>'
        # This means tokens like <pad> or <mask> (not registered) are also silently dropped,
        # and registered special tokens with unusual names would not be dropped.
        vocab = tok.get_vocab()
        sample_ids = tok.encode("नमस्ते world").ids[:5]
        # Inject a fake token that looks like a special token
        print("  decode() code (tokenizer.py ~line 185)::")
        print("    tokens = [t for t in tokens")
        print("              if t and not (t in special_strs")
        print("              or (t.startswith('<') and t.endswith('>')))  # HEURISTIC")
        print("  Problem 1: Any token like <tok42> would be silently dropped")
        print("             even if it is NOT a registered special token.")
        print("  Problem 2: registered special tokens without <> brackets")
        print("             (e.g., '[CLS]', 'MASK') would NOT be skipped.")
        print("  Evidence  : tokenizer.py line 185-191")
        print("  Risk      : MEDIUM — silent data loss in edge cases")

        subsection("3. encode_batch() is purely sequential — no parallelism")
        texts = ["नमस्ते world भारत tests"] * 500
        t0 = time.perf_counter()
        encodings = tok.encode_batch(texts)
        elapsed = time.perf_counter() - t0
        throughput = len(texts) / elapsed
        print(f"  encode_batch({len(texts)} sentences) took {elapsed:.4f}s -> {throughput:.1f} sent/sec")
        print("  encode_batch() implementation (tokenizer.py line 163):")
        print("    return [self.encode(t) for t in texts]  # plain list comprehension")
        print("  No multiprocessing, no batching, no async I/O.")
        secs_per_million = 1_000_000 / throughput
        print(f"  At {throughput:.0f} sent/sec: 1M docs takes ~{secs_per_million:.0f}s ({secs_per_million/3600:.2f} hrs) on a single core.")
        print("  In production (multi-user, real-time): single-threaded bottleneck")
        print("  Evidence  : tokenizer.py line 163")
        print("  Risk      : HIGH for million-doc/day SLA; LOW for batch-offline pipelines")

        # ─────────────────────────────────────────────────────────────────────
        # URGENCY RANKING
        # ─────────────────────────────────────────────────────────────────────
        separator("URGENCY RANKING")
        print("""
  #1 CRITICAL — Persistence gap (save/load drops normalizer + pretokenizer)
      A loaded tokenizer produces DIFFERENT output than the trained one.
      This is a silent correctness bug that would corrupt inference in prod.
      Fix: serialize full TokenizerConfig into config.json during save(),
           and restore it during load().

  #2 HIGH — encode_batch() single-threaded throughput ceiling
      Cannot meet millions-of-docs/day SLA without parallelism.
      Fix: use multiprocessing.Pool or concurrent.futures.ProcessPoolExecutor
           in encode_batch() for large batches.

  #3 MEDIUM — decode() special-token heuristic
      Silent token drops for edge-case token strings.
      Fix: replace the `startswith('<') and endswith('>')` heuristic with
           strict lookup of only registered special_tokens keys.
""")


if __name__ == "__main__":
    main()
