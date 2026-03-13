from pathlib import Path
import tempfile

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual


def main() -> None:
    mantra = (
        "\u0950 \u092d\u0942\u0930\u094d\u092d\u0941\u0935\u0903 "
        "\u0938\u094d\u0935: \u0924\u0924\u094d\u0938\u0935\u093f\u0924\u0941\u0930\u094d\u0935\u0930\u0947\u0923\u094d\u092f\u0902 "
        "\u092d\u0930\u094d\u0917\u094b \u0926\u0947\u0935\u0938\u094d\u092f \u0927\u0940\u092e\u0939\u093f "
        "\u0927\u093f\u092f\u094b \u092f\u094b \u0928\u0903 \u092a\u094d\u0930\u091a\u094b\u0926\u092f\u093e\u0924\u094d "
        "\u0965"
    )

    corpus_lines = [
        mantra,
        "simple english line",
        "another mixed line",
        "devanagari and english mixed",
    ] * 30

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task1_corpus.txt"
        corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

        config = bpe_multilingual(vocab_size=220)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)

        print("=== Stage 0: Raw Input ===")
        print(mantra)

        normalized = tokenizer._normalizer.normalize(mantra) if tokenizer._normalizer else mantra
        print("\n=== Stage 1: Normalized ===")
        print(normalized)

        pre_tokens = (
            tokenizer._pretokenizer.pre_tokenize(normalized)
            if tokenizer._pretokenizer
            else [normalized]
        )
        print("\n=== Stage 2: Pre-Tokens ===")
        for i, pre_tok in enumerate(pre_tokens, start=1):
            print(f"{i:02d}. {pre_tok}")

        print("\n=== Stage 3: Model Pieces (Per Pre-Token) ===")
        for pre_tok in pre_tokens:
            pairs = tokenizer._model.tokenize(pre_tok)
            print(f"[{pre_tok}]")
            print("  pieces:", [t for t, _ in pairs])
            print("  ids   :", [i for _, i in pairs])

        encoding = tokenizer.encode(mantra)
        print("\n=== Stage 4: Final Encoding ===")
        print("tokens:", encoding.tokens)
        print("ids   :", encoding.ids)

        decoded = tokenizer.decode(encoding.ids)
        print("\n=== Decode(ids) ===")
        print(decoded)

        print("\n=== Sanity Checks ===")
        print("normalized_equals_raw:", normalized == mantra)
        print("pre_token_count:", len(pre_tokens))
        print("final_token_count:", len(encoding.tokens))


if __name__ == "__main__":
    main()
