"""Task 10 — The Compression Trade-off.

Compares two BPE configurations that differ only by vocabulary size.
Shows a concrete case where one metric improves while another gets worse.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import time

from abctokz import Tokenizer
from abctokz.config.defaults import bpe_multilingual
from abctokz.eval.intrinsic import evaluate_tokenizer

# Same two phrases from Task 8/9 are included in evaluation focus.
PHRASE_I = "आयो लाल, सभई चायो, झूलेलाल!"
PHRASE_II = "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!"

TRAIN_LINES = [
    "आयो लाल, सभई चायो, झूलेलाल!",
    "झूलेलाल सिंधु संस्कृति में आदर के साथ गाया जाता है",
    "गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!",
    "मराठी भक्तीगीतांमध्ये गणपतीचे अनेक उल्लेख आढळतात",
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "भारत विविध भाषाओं और लिपियों का देश है",
    "देवनागरी लिपि में मात्राएं और संयुक्ताक्षर महत्वपूर्ण हैं",
    "क्ष त्र ज्ञ जैसे संयुक्ताक्षर ध्वनि और अर्थ पर प्रभाव डालते हैं",
    "भाषिक सामान्यीकरण से इनपुट स्थिर और तुलना योग्य बनता है",
    "नमस्ते दुनिया यह एक परीक्षण वाक्य है",
    "हिंदी मराठी सिंधी संस्कृत में ध्वन्यात्मक विविधता पाई जाती है",
    "पावसाळ्यात डोंगर हिरवेगार दिसतात आणि नद्या भरतात",
    "सण उत्सवांच्या काळात गावोगावी कीर्तन आणि भजन होतात",
    "शब्दरचना आणि प्रत्यय मराठीमध्ये समृद्ध पद्धतीने वापरले जातात",
    "उच्चारण बदलल्यास अर्थ बदलू शकतो म्हणून लिप्यात्मक अचूकता आवश्यक आहे",
    "hello world this is a multilingual tokenizer benchmark",
    "tokenization quality depends on corpus and vocabulary",
    "mixed script input नमस्ते world often appears in user data",
] * 50

EVAL_SENTENCES = [
    PHRASE_I,
    PHRASE_II,
    "जन गण मन अधिनायक जय हे भारत भाग्य विधाता",
    "क्ष त्र ज्ञ जैसे संयुक्ताक्षर चुनौतीपूर्ण हो सकते हैं",
    "mixed script नमस्ते world test sentence",
    "मराठी आणि हिंदी दोन्हीमध्ये देवनागरी वापरली जाते",
    "tokenization quality must balance compression and efficiency",
] * 20


def directory_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def run_experiment(vocab_size: int, corpus_path: Path, out_dir: Path) -> dict[str, float]:
    config = bpe_multilingual(vocab_size=vocab_size)
    tokenizer = Tokenizer.from_config(config)

    t0 = time.perf_counter()
    tokenizer.train([str(corpus_path)], config)
    train_seconds = time.perf_counter() - t0

    result = evaluate_tokenizer(
        tokenizer=tokenizer,
        sentences=EVAL_SENTENCES,
        name=f"bpe_vs{vocab_size}",
        language="mixed-devanagari",
    )

    save_path = out_dir / f"tok_vs_{vocab_size}"
    tokenizer.save(str(save_path))
    artifact_size = directory_size_bytes(save_path)

    return {
        "vocab_size": float(vocab_size),
        "actual_vocab": float(tokenizer.get_vocab_size()),
        "train_seconds": train_seconds,
        "fertility": result.fertility,
        "mean_tokens_per_sentence": result.mean_tokens_per_sentence,
        "throughput_sps": result.throughput_sps,
        "round_trip_success_rate": result.round_trip_success_rate,
        "unk_rate": result.unk_rate,
        "artifact_size_bytes": float(artifact_size),
    }


def pct_change(old: float, new: float) -> float:
    if abs(old) < 1e-12:
        return 0.0
    return ((new - old) / old) * 100.0


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_path = tmp_path / "task10_corpus.txt"
        corpus_path.write_text("\n".join(TRAIN_LINES), encoding="utf-8")

        small = run_experiment(100, corpus_path, tmp_path)
        large = run_experiment(800, corpus_path, tmp_path)

        print("=" * 86)
        print("TASK 10 — COMPRESSION TRADE-OFF (BPE vocab_size: 100 -> 800)")
        print("=" * 86)
        print(f"train_lines={len(TRAIN_LINES)}, eval_sentences={len(EVAL_SENTENCES)}")

        print("\nConfiguration A (small vocab)")
        print(f"  vocab_size target={int(small['vocab_size'])}, actual={int(small['actual_vocab'])}")
        print(f"  fertility={small['fertility']:.4f}")
        print(f"  mean_tokens_per_sentence={small['mean_tokens_per_sentence']:.4f}")
        print(f"  throughput_sps={small['throughput_sps']:.2f}")
        print(f"  train_seconds={small['train_seconds']:.4f}")
        print(f"  artifact_size_bytes={int(small['artifact_size_bytes'])}")

        print("\nConfiguration B (large vocab)")
        print(f"  vocab_size target={int(large['vocab_size'])}, actual={int(large['actual_vocab'])}")
        print(f"  fertility={large['fertility']:.4f}")
        print(f"  mean_tokens_per_sentence={large['mean_tokens_per_sentence']:.4f}")
        print(f"  throughput_sps={large['throughput_sps']:.2f}")
        print(f"  train_seconds={large['train_seconds']:.4f}")
        print(f"  artifact_size_bytes={int(large['artifact_size_bytes'])}")

        print("\nDelta (B - A)")
        print(f"  fertility: {large['fertility'] - small['fertility']:+.4f} ({pct_change(small['fertility'], large['fertility']):+.2f}%)")
        print(
            f"  mean_tokens_per_sentence: {large['mean_tokens_per_sentence'] - small['mean_tokens_per_sentence']:+.4f} "
            f"({pct_change(small['mean_tokens_per_sentence'], large['mean_tokens_per_sentence']):+.2f}%)"
        )
        print(f"  throughput_sps: {large['throughput_sps'] - small['throughput_sps']:+.2f} ({pct_change(small['throughput_sps'], large['throughput_sps']):+.2f}%)")
        print(f"  train_seconds: {large['train_seconds'] - small['train_seconds']:+.4f} ({pct_change(small['train_seconds'], large['train_seconds']):+.2f}%)")
        print(
            f"  artifact_size_bytes: {int(large['artifact_size_bytes'] - small['artifact_size_bytes']):+d} "
            f"({pct_change(small['artifact_size_bytes'], large['artifact_size_bytes']):+.2f}%)"
        )

        print("\nTrade-off verdict")
        improved = []
        worsened = []

        if large["fertility"] < small["fertility"]:
            improved.append("fertility (lower is better)")
        else:
            worsened.append("fertility")

        if large["mean_tokens_per_sentence"] < small["mean_tokens_per_sentence"]:
            improved.append("mean_tokens_per_sentence (lower is better)")
        else:
            worsened.append("mean_tokens_per_sentence")

        if large["throughput_sps"] > small["throughput_sps"]:
            improved.append("throughput_sps (higher is better)")
        else:
            worsened.append("throughput_sps (lower encoding throughput)")

        if large["artifact_size_bytes"] > small["artifact_size_bytes"]:
            worsened.append("artifact_size_bytes (larger model artifact)")
        else:
            improved.append("artifact_size_bytes")

        if large["train_seconds"] > small["train_seconds"]:
            worsened.append("train_seconds (longer training)")
        else:
            improved.append("train_seconds")

        print("  Improved with larger vocab:")
        for item in improved:
            print(f"    - {item}")
        print("  Worsened with larger vocab:")
        for item in worsened:
            print(f"    - {item}")

        print("\nConclusion")
        print("  This is a genuine trade-off: larger vocab improves compression")
        print("  but increases artifact size and can reduce encode throughput.")
        print("  Whether to ship it depends on deployment constraints:")
        print("  edge/on-device systems may prefer smaller artifacts, while")
        print("  server-side systems may accept larger artifacts for better compression.")


if __name__ == "__main__":
    main()
