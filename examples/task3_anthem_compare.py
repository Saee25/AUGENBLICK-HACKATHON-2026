from pathlib import Path
import tempfile

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
from abctokz.eval.metrics import fertility


DEVANAGARI_ANTHEM = (
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता पंजाब सिंधु गुजरात मराठा "
    "द्राविड उत्कल बंगा विंध्य हिमाचल यमुना गंगा उच्छल जलधि तरंग "
    "तव शुभ नामे जागे तव शुभ आशीष मागे गाहे तव जय गाथा "
    "जन गण मंगलदायक जय हे भारत भाग्य विधाता जय हे जय हे जय हे "
    "जय जय जय जय हे"
)

TRANSLITERATION_ANTHEM = (
    "Jana Gana Mana Adhinayaka Jaya He Bharata Bhagya Vidhata Punjab Sindhu Gujarat Maratha "
    "Dravida Utkala Banga Vindhya Himachala Yamuna Ganga Ucchala Jaladhi Taranga "
    "Tava Shubha Name Jage Tava Shubha Ashisha Mage Gahe Tava Jaya Gatha "
    "Jana Gana Mangaladayaka Jaya He Bharata Bhagya Vidhata Jaya He Jaya He Jaya He "
    "Jaya Jaya Jaya Jaya He"
)

EXTRA_CORPUS_LINES = [
    "hello world this is a multilingual tokenizer benchmark",
    "देवनागरी और अंग्रेज़ी मिश्रित पाठ",
    "tokenization quality depends on corpus and vocabulary",
    "भारत एक विशाल देश है",
    "multilingual bpe handles shared and script specific patterns",
    "हिन्दी मराठी और संस्कृत में संयुक्ताक्षर महत्वपूर्ण हैं",
]


def print_summary(label: str, text: str, tokenizer: Tokenizer) -> dict[str, object]:
    encoding = tokenizer.encode(text)
    word_count = len(text.split())
    fert = fertility([encoding], [word_count])

    print(f"\n=== {label} ===")
    print(f"word_count   : {word_count}")
    print(f"token_count  : {len(encoding)}")
    print(f"fertility    : {fert:.4f}")
    print(f"first_tokens : {encoding.tokens[:25]}")
    print(f"decoded_same : {tokenizer.decode(encoding.ids) == text}")

    return {
        "encoding": encoding,
        "word_count": word_count,
        "token_count": len(encoding),
        "fertility": fert,
    }


def print_tiktoken_bonus() -> None:
    try:
        import tiktoken
    except ImportError:
        print("\n=== Bonus: tiktoken ===")
        print("tiktoken not installed, skipping bonus comparison.")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    dev_ids = enc.encode(DEVANAGARI_ANTHEM)
    lat_ids = enc.encode(TRANSLITERATION_ANTHEM)

    print("\n=== Bonus: tiktoken (cl100k_base) ===")
    print(f"transliteration_token_count : {len(lat_ids)}")
    print(f"devanagari_token_count     : {len(dev_ids)}")
    print(f"transliteration_fertility  : {len(lat_ids) / len(TRANSLITERATION_ANTHEM.split()):.4f}")
    print(f"devanagari_fertility       : {len(dev_ids) / len(DEVANAGARI_ANTHEM.split()):.4f}")


def main() -> None:
    corpus_lines = [
        DEVANAGARI_ANTHEM,
        TRANSLITERATION_ANTHEM,
        "जन गण मन अधिनायक जय हे",
        "Jana Gana Mana Adhinayaka Jaya He",
        "भारत भाग्य विधाता",
        "Bharata Bhagya Vidhata",
        *EXTRA_CORPUS_LINES,
    ] * 40

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "task3_corpus.txt"
        corpus_path.write_text("\n".join(corpus_lines), encoding="utf-8")

        config = bpe_multilingual(vocab_size=320)
        tokenizer = Tokenizer.from_config(config)
        tokenizer.train([str(corpus_path)], config)

        print("=== Task 3 Setup ===")
        print(f"model_type   : {config.model.type}")
        print(f"vocab_size   : {tokenizer.get_vocab_size()}")
        print(f"corpus_lines : {len(corpus_lines)}")

        translit = print_summary("English Transliteration", TRANSLITERATION_ANTHEM, tokenizer)
        dev = print_summary("Devanagari", DEVANAGARI_ANTHEM, tokenizer)

        print("\n=== Direct Comparison ===")
        print(f"token_count_difference : {translit['token_count']} - {dev['token_count']} = {translit['token_count'] - dev['token_count']}")
        print(f"fertility_difference   : {translit['fertility']:.4f} - {dev['fertility']:.4f} = {translit['fertility'] - dev['fertility']:.4f}")

        print_tiktoken_bonus()


if __name__ == "__main__":
    main()
