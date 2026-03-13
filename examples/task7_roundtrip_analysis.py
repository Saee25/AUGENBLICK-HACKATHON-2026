from pathlib import Path
import tempfile
import unicodedata

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
from abctokz.eval.metrics import round_trip_success_rate


def main() -> None:
    nfd_case = unicodedata.normalize("NFD", "Café नमस्ते world")

    corpus_lines = [
        "नमस्ते world",
        "hello दुनिया",
        "Café नमस्ते world",
        "जन गण मन",
        "मराठी भाषेत परीक्षण",
        "mixed script tokenization works",
        nfd_case,
    ] * 40

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task7_corpus.txt"
        corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

        config = bpe_multilingual(vocab_size=260)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)

        test_cases = [
            ("exact_simple", "नमस्ते world"),
            ("exact_mixed", "hello दुनिया"),
            ("lossy_spaces", "नमस्ते   world"),
            ("lossy_nfd", nfd_case),
            ("exact_devanagari", "जन गण मन"),
            ("exact_marathi", "मराठी भाषेत परीक्षण"),
        ]

        print("=== Task 7 Round-Trip Experiment ===")
        print(f"model_type: {config.model.type}")
        print(f"vocab_size: {tokenizer.get_vocab_size()}")
        print(f"num_cases : {len(test_cases)}")

        originals: list[str] = []
        decoded: list[str] = []
        normalized_originals: list[str] = []

        for name, text in test_cases:
            normalized = tokenizer._normalizer.normalize(text) if tokenizer._normalizer else text
            enc = tokenizer.encode(text)
            dec = tokenizer.decode(enc.ids)

            originals.append(text)
            decoded.append(dec)
            normalized_originals.append(normalized)

            print(f"\n--- {name} ---")
            print("raw_repr       :", repr(text))
            print("normalized_repr:", repr(normalized))
            print("decoded_repr   :", repr(dec))
            print("token_count    :", len(enc))
            print("matches_raw    :", dec == text)
            print("matches_norm   :", dec == normalized)

        raw_rate = round_trip_success_rate(originals, decoded)
        norm_rate = round_trip_success_rate(originals, decoded, normalized_originals)

        print("\n=== Round-Trip Metric Summary ===")
        print(f"round_trip_success_rate_raw_targets        : {raw_rate:.4f}")
        print(f"round_trip_success_rate_normalized_targets : {norm_rate:.4f}")


if __name__ == "__main__":
    main()
