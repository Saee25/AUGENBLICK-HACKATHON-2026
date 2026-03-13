"""Task 9 — Measuring Phrase Difficulty.

Trains BPE and Unigram tokenizers on a Devanagari-rich corpus, then measures
fertility for two phrases from Task 8 at multiple vocabulary sizes.
"""

from pathlib import Path
import tempfile

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual, unigram_multilingual
from abctokz.eval.metrics import fertility

PHRASE_I = "आयो लाल, सभई चायो, झूलेलाल!"  # Sindhi folk phrase
PHRASE_II = "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"  # Marathi chant

PHRASES = {
    "I_sindhi": PHRASE_I,
    "II_marathi": PHRASE_II,
}

VOCAB_SIZES = [100, 400, 800]

# Devanagari-rich corpus (Hindi + Marathi + Sindhi + Sanskrit-flavored lines)
# Repeated to stabilize frequency counts for small experiments.
BASE_DEVANAGARI_LINES = [
    "आयो लाल, सभई चायो, झूलेलाल!",
    "झूलेलाल सिंधु संस्कृति में आदर के साथ गाया जाता है",
    "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!",
    "मराठी भक्तीगीतांमध्ये गणपतीचे अनेक उल्लेख आढळतात",
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "भारत विविध भाषाओं और लिपियों का देश है",
    "देवनागरी लिपि में मात्राएं और संयुक्ताक्षर महत्वपूर्ण हैं",
    "क्ष त्र ज्ञ जैसे संयुक्ताक्षर ध्वनि और अर्थ पर प्रभाव डालते हैं",
    "शब्दसंग्रह जितना समृद्ध होगा उतना बेहतर उपशब्द विभाजन मिलेगा",
    "भाषिक सामान्यीकरण से इनपुट स्थिर और तुलना योग्य बनता है",
    "नमस्ते दुनिया यह एक परीक्षण वाक्य है",
    "हिंदी मराठी सिंधी संस्कृत में ध्वन्यात्मक विविधता पाई जाती है",
    "पावसाळ्यात डोंगर हिरवेगार दिसतात आणि नद्या भरतात",
    "सण उत्सवांच्या काळात गावोगावी कीर्तन आणि भजन होतात",
    "शब्दरचना आणि प्रत्यय मराठीमध्ये समृद्ध पद्धतीने वापरले जातात",
    "उच्चारण बदलल्यास अर्थ बदलू शकतो म्हणून लिप्यात्मक अचूकता आवश्यक आहे",
]


def build_corpus_lines() -> list[str]:
    lines: list[str] = []
    lines.extend(BASE_DEVANAGARI_LINES * 40)

    # Add phrase-focused variants to make the task phrases well represented.
    lines.extend(
        [
            PHRASE_I,
            PHRASE_II,
            "आयो लाल झूलेलाल जय झूलेलाल",
            "गणपती बप्पा मोरया मंगलमूर्ती मोरया",
            "पुढच्या वर्षी लवकर या गणराजा",
        ]
        * 60
    )
    return lines


def train_and_measure(model_name: str, vocab_size: int, corpus_path: str) -> dict[str, dict[str, float]]:
    if model_name == "bpe":
        config = bpe_multilingual(vocab_size=vocab_size)
    elif model_name == "unigram":
        config = unigram_multilingual(vocab_size=vocab_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    tokenizer = Tokenizer.from_config(config)
    tokenizer.train([corpus_path], config)

    row: dict[str, dict[str, float]] = {}
    for phrase_name, phrase in PHRASES.items():
        enc = tokenizer.encode(phrase)
        words = len(phrase.split())
        fert = fertility([enc], [words])
        row[phrase_name] = {
            "tokens": float(len(enc)),
            "words": float(words),
            "fertility": fert,
        }
    return row


def main() -> None:
    corpus_lines = build_corpus_lines()

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task9_devanagari_rich_corpus.txt"
        corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

        results: dict[str, dict[int, dict[str, dict[str, float]]]] = {
            "bpe": {},
            "unigram": {},
        }

        print("=" * 78)
        print("TASK 9 — MEASURING PHRASE DIFFICULTY")
        print("=" * 78)
        print(f"corpus_lines: {len(corpus_lines)}")
        print(f"phrase_I_words: {len(PHRASE_I.split())}, phrase_II_words: {len(PHRASE_II.split())}")

        for model_name in ["bpe", "unigram"]:
            print("\n" + "-" * 78)
            print(f"MODEL: {model_name.upper()}")
            print("-" * 78)

            for vs in VOCAB_SIZES:
                row = train_and_measure(model_name, vs, str(corpus_path))
                results[model_name][vs] = row

                i = row["I_sindhi"]
                ii = row["II_marathi"]
                print(f"vocab_size={vs:>3} | I_sindhi: tokens={int(i['tokens']):>2}, fert={i['fertility']:.4f}"
                      f" | II_marathi: tokens={int(ii['tokens']):>2}, fert={ii['fertility']:.4f}")

            # Per-model difficulty conclusion by mean fertility across vocab sizes
            mean_i = sum(results[model_name][vs]["I_sindhi"]["fertility"] for vs in VOCAB_SIZES) / len(VOCAB_SIZES)
            mean_ii = sum(results[model_name][vs]["II_marathi"]["fertility"] for vs in VOCAB_SIZES) / len(VOCAB_SIZES)
            if abs(mean_i - mean_ii) < 1e-12:
                harder = "tie"
            else:
                harder = "I_sindhi" if mean_i > mean_ii else "II_marathi"
            print(f"\n{model_name.upper()} mean fertility: I={mean_i:.4f}, II={mean_ii:.4f} -> harder={harder}")

        # Cross-model trend summary
        print("\n" + "=" * 78)
        print("TREND SUMMARY")
        print("=" * 78)
        for model_name in ["bpe", "unigram"]:
            i_100 = results[model_name][100]["I_sindhi"]["fertility"]
            i_800 = results[model_name][800]["I_sindhi"]["fertility"]
            ii_100 = results[model_name][100]["II_marathi"]["fertility"]
            ii_800 = results[model_name][800]["II_marathi"]["fertility"]

            print(
                f"{model_name.upper()}: I_sindhi fert {i_100:.4f} -> {i_800:.4f} (delta {i_800 - i_100:+.4f}), "
                f"II_marathi fert {ii_100:.4f} -> {ii_800:.4f} (delta {ii_800 - ii_100:+.4f})"
            )

        print("\nInterpretation hint:")
        print("- Lower fertility = fewer tokens per word = more efficient tokenization.")
        print("- If fertility drops with larger vocab, model is learning larger reusable subwords.")
        print("- Phrase-level differences usually come from morphology, recurring subword patterns,")
        print("  and how well those patterns are represented in the training corpus.")


if __name__ == "__main__":
    main()
