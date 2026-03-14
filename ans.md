# Task 1 — Tokenization Pipeline Trace

## What We Did

- Trained one **BPE tokenizer** on a small corpus containing the Gayatri mantra plus a few filler lines.
- Encoded the exact mantra provided.
- Printed each **pipeline stage before and after model tokenization**.
- Decoded final token IDs back to text.

## Command Run

```bash
python.exe task1_trace_pipeline.py
````

---

# Stage-by-Stage Observations

## Stage 0 — Raw Input

The mantra used as input:

```
ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥
```

---

## Stage 1 — Normalizer Output

Observed output is **identical to the raw input**.

Evidence from script output:

```
normalized_equals_raw: True
```

### Why this happened

* The default BPE preset defined in `defaults.py` uses a **Devanagari-safe sequence normalizer**.
* The normalization pipeline runs through `sequence.py`.
* Devanagari-specific behavior is implemented in `devanagari.py`.

Since the mantra text already conforms to the expected normalized form, the normalizer **does not modify the string**.

---

## Stage 2 — Pre-Tokenizer Output

Observed **pre-token count:**

```
12
```

Observed **pre-tokens:**

```
ॐ | भूर्भुवः | स्व: | तत्सवितुर्वरेण्यं | भर्गो | देवस्य | धीमहि | धियो | यो | नः | प्रचोदयात् | ॥
```

### Why this happened

* The pre-tokenizer pipeline is defined in `sequence.py`.
* Devanagari-aware splitting logic is implemented in `devanagari_aware.py`.

### Behavior observed

* Text was **split primarily on whitespace**.
* Punctuation remained attached to tokens, such as:

```
स्व:
॥
```

So the pre-tokenizer produced **word-like units rather than characters**.

---

## Stage 3 — Model Piece Generation (BPE)

Example outputs from the BPE model:

```
भूर्भुवः → भ, ##ू, ##र्, ##भु, ##व, ##ः
```

```
तत्सवितुर्वरेण्यं →
त, ##त्, ##स, ##व, ##ि, ##तु, ##र्, ##व, ##र, ##े, ##ण, ##्य, ##ं
```

```
प्रचोदयात् →
प, ##्, ##र, ##चो, ##द, ##य, ##ा, ##त्
```

### Why this happens

* Implemented in `bpe.py`.
* BPE starts with **character pieces** and repeatedly merges frequent pairs.
* Non-initial subword pieces receive the continuation prefix:

```
##
```

This prefix tells the decoder that the piece **continues the previous token rather than starting a new word**.

---

## Stage 4 — Final Encoding

Observed:

```
Final token count: 48
IDs list length: 48
```

### Full pipeline inside `tokenizer.py`

```
normalized text
    ↓
pre-tokens
    ↓
model tokenization
    ↓
Encoding object (token ids)
```

---

# Decode Path

Observed decoded output matches the normalized text.

Example decode output:

```
ॐ भूर्भुवः स्व: तत्सवितुर्वरेण्यं भर्गो देवस्य धीमहि धियो यो नः प्रचोदयात् ॥
```

### How decoding works

1. `tokenizer.py` maps **IDs → tokens**
2. `subword_decoder.py` merges continuation pieces
3. Word boundaries and spaces are restored

---

# Files / Classes Involved (Execution Order)

| Stage                  | File                  | Responsibility                        |
| ---------------------- | --------------------- | ------------------------------------- |
| Config preset builder  | `defaults.py`         | Builds tokenizer preset               |
| Pipeline orchestration | `tokenizer.py`        | Controls encode/decode pipeline       |
| Normalizer chain       | `sequence.py`         | Runs ordered normalizers              |
| Devanagari normalizer  | `devanagari.py`       | Handles script-specific normalization |
| Pre-tokenizer chain    | `sequence.py`         | Runs pre-tokenizers                   |
| Devanagari splitting   | `devanagari_aware.py` | Splits tokens safely                  |
| BPE token generation   | `bpe.py`              | Creates subword pieces                |
| Subword reconstruction | `subword_decoder.py`  | Reassembles tokens during decode      |

---

# Conceptual Questions

## 1. Which files and classes were involved at each stage?

* `defaults.py` built the preset.
* `tokenizer.py` orchestrated the pipeline.
* `sequence.py` executed the normalizer and pre-tokenizer chains.
* `devanagari.py` handled script-specific normalization.
* `devanagari_aware.py` handled splitting logic.
* `bpe.py` generated subword tokens.
* `subword_decoder.py` reconstructed text during decoding.

---

## 2. What did the normalizer do to the string before the model saw it?

The sequence normalizer executed **Devanagari-safe normalization** but made **no changes**, so the normalized text remained identical to the raw mantra.

---

## 3. What did the pre-tokenizer do after normalization?

The pre-tokenizer split the text into **12 word-like tokens** using whitespace while keeping punctuation attached (e.g., `स्व:` and `॥`).

---

## 4. How did the model turn pre-tokens into subword pieces?

The BPE model in `bpe.py` applied **greedy merges** learned during training, producing subword pieces.

Non-initial pieces were marked with:

```
##
```

---

## 5. How were those pieces turned back into a string during `decode()`?

* `tokenizer.py` converts token IDs back to subword tokens.
* `subword_decoder.py` merges continuation pieces and restores spacing.

---

# Task 2 — Repository Responsibility Map

This section analyzes how responsibilities are divided across the repository.

---

# 1. Training a Tokenizer

## Responsibility

Training logic is split between:

* CLI interface
* configuration layer
* trainer implementations

## Main Modules

* `src/abctokz/cli/train.py`
* `src/abctokz/tokenizer.py`
* `src/abctokz/trainers/__init__.py`
* `src/abctokz/trainers/bpe_trainer.py`
* `src/abctokz/trainers/unigram_trainer.py`
* `src/abctokz/trainers/wordlevel_trainer.py`

## How It Works

1. CLI (`train.py`) gathers user inputs
2. `Tokenizer.from_config` builds the pipeline
3. `Tokenizer.train` prepares the normalized corpus
4. `build_trainer` selects the correct trainer
5. Trainer modules learn vocabulary and merges

## Why This Separation Exists

This design separates:

* **User interaction**
* **Pipeline construction**
* **Learning algorithms**

---

# 2. Using a Trained Tokenizer (Inference)

## Responsibility

Inference is handled by:

* tokenizer orchestration layer
* model-specific segmentation logic

## Main Modules

* `src/abctokz/tokenizer.py`
* `src/abctokz/normalizers`
* `src/abctokz/pretokenizers`
* `src/abctokz/models/bpe.py`
* `src/abctokz/models/unigram.py`
* `src/abctokz/models/wordlevel.py`

## Pipeline

```
normalizer
   ↓
pre-tokenizer
   ↓
model.tokenize()
   ↓
post-processing
```

---

# 3. Saving and Loading a Tokenizer

## Responsibility

Persistence is shared between:

* tokenizer
* model classes
* I/O utilities

## Main Modules

* `src/abctokz/tokenizer.py`
* `src/abctokz/models`
* `src/abctokz/utils/io.py`

## Behavior

* `Tokenizer.save()` writes artifact directory
* Model classes save vocabulary / merges
* `io.py` handles filesystem + JSON

---

# 4. Measuring Tokenizer Quality

## Modules

* `src/abctokz/eval/metrics.py`
* `src/abctokz/eval/intrinsic.py`
* `src/abctokz/eval/benchmark.py`
* `src/abctokz/eval/reports.py`

## Responsibilities

| Module       | Role                 |
| ------------ | -------------------- |
| metrics.py   | metric formulas      |
| intrinsic.py | tokenizer evaluation |
| benchmark.py | experiment runner    |
| reports.py   | output formatting    |

---

# Clean Architecture Boundary

## Evaluation Stack

Cleanest separation:

* `metrics.py`
* `intrinsic.py`
* `reports.py`

Why it works:

* Metrics are **pure functions**
* Execution separated from reporting

---

# Blurry Architecture Boundary

## Tokenizer Persistence

The boundary around `Tokenizer.save()` and `Tokenizer.load()` is unclear.

### Reason

`load()` restores:

* model
* decoder
* special tokens

But **not**

* normalizer
* pre-tokenizer configuration

So artifacts represent **model state only**, not the **full pipeline**.

---

# Possible Improvements

## Option 1 — Clarify the Contract

Document that artifacts save **model inference state only**.

## Option 2 — Full Pipeline Persistence (Preferred)

Persist the entire `TokenizerConfig`.

Restore:

* normalizer
* pre-tokenizer
* post-processor
* model

---

# Blurry Boundary Example

`tokenizer.py` handles **too many responsibilities**.

## Current roles

* encoding / decoding
* training
* persistence

Example: `train()` contains **20+ lines of training logic**.

### Problem

Mixes **runtime inference** with **training responsibilities**.

---

## Proposed Refactor

Create a separate trainer controller:

```python
class TokenizerTrainer:

    @classmethod
    def from_config(cls, config):
        ...

    def train(self, tokenizer, corpus_paths):
        ...
```

Tokenizer delegates:

```python
trainer = TokenizerTrainer.from_config(config)
trainer.train(tokenizer, corpus_paths)
```

### Benefits

* `tokenizer.py` focuses on runtime pipeline
* trainers module owns training logic
* easier testing

---

# Task 3 — The National Anthem Test

## Objective

Compare tokenization efficiency between:

* English transliteration
* Devanagari script

Efficiency metric: **fertility score**

---

# Fertility Score

```
Fertility = Total Tokens / Total Words
```

Lower fertility → more efficient tokenization.

---

# Results

| Version         | Words | Tokens | Fertility |
| --------------- | ----- | ------ | --------- |
| Transliteration | 54    | 182    | 3.3704    |
| Devanagari      | 54    | 129    | 2.3889    |

---

# External Comparison — tiktoken

| Version         | Tokens | Fertility |
| --------------- | ------ | --------- |
| Transliteration | 114    | 2.1111    |
| Devanagari      | 265    | 4.9074    |

---

# Key Insight

Tokenization efficiency depends on:

* training corpus
* vocabulary size
* script structure
* subword frequency

Not script alone.

---

# Task 7 — Encode → Decode Round Trip Analysis

## Objective

Check if tokenizer is **lossless**.

```
decode(encode(text)) == text
```

---

# Round-Trip Results

| Comparison           | Success |
| -------------------- | ------- |
| Raw originals        | 0.6667  |
| Normalized originals | 1.0000  |

### Explanation

Normalization causes differences:

Examples:

```
Hello   world
↓
Hello world
```

and

```
Café
↓
Café
```

---

# Conclusion

Tokenizer preserves **normalized text perfectly**, but not raw formatting.

---

# Task 8 — What Does the Normalizer Actually Do?

## Test Phrases

### Sindhi Phrase

```
आयो लाल, सभई चायो, झूलेलाल!
```

### Marathi Chant

```
गणपती बप्पा मोरया, पुढच्या वर्षी लवकर या!
```

---

# Result

| Phrase    | Raw  | Normalized | Changed |
| --------- | ---- | ---------- | ------- |
| Phrase I  | same | same       | No      |
| Phrase II | same | same       | No      |

---

# Normalization Strategy

Uses **NFC normalization**.

Reason:

NFKC can break Devanagari ligatures like:

```
क्ष
```

---

# Pre-Tokenization

Splits by **script groups**:

* Devanagari
* Latin
* punctuation

Punctuation stays attached to previous token.

Example:

```
लाल,
```

---

# Key Insight

Tokenizer pipeline:

```
Raw Text
↓
Unicode Normalization
↓
Whitespace Normalization
↓
Devanagari Pre-tokenization
↓
Model Tokenization
```

---

# Task 9 — Measuring Phrase Difficulty

## Fertility Formula

```
fertility = tokens / words
```

---

# Results

## BPE

| Vocab | Phrase I | Phrase II |
| ----- | -------- | --------- |
| 100   | 3.6000   | 4.1429    |
| 400   | 3.4000   | 3.7143    |
| 800   | 3.4000   | 3.7143    |

## Unigram

| Vocab | Phrase I | Phrase II |
| ----- | -------- | --------- |
| 100   | 1.0      | 1.0       |
| 400   | 1.0      | 1.0       |
| 800   | 1.0      | 1.0       |

---

# Conclusion

* Phrase II harder for BPE
* Unigram handles both phrases equally well
* BPE improves with larger vocab
* Unigram already optimal

---

# Task 19 — BPE vs Unigram vs WordLevel

## Experimental Setup

Corpus:

* English
* Hindi
* Marathi

Vocabulary size limited to **300**.

Goal: force algorithms to rely on segmentation strategy.

---

# Model Behavior

| Model     | Behavior                        |
| --------- | ------------------------------- |
| WordLevel | Whole words only                |
| BPE       | Character-based merges          |
| Unigram   | Probabilistic subword selection |

---

# Example Behavior

## WordLevel

```
the cat sat on the mat
→ ['the', '<unk>', '<unk>', '<unk>', 'the', '<unk>']
```

---

## BPE

```
tokenization
→ ['t', '##o', '##k', '##en', '##i', '##z', ...]
```

---

## Unigram

```
tokenization
→ ['tokenization', '<unk>', ...]
```

---

# Model Choice Recommendations

| Scenario                             | Best Model |
| ------------------------------------ | ---------- |
| Low-resource language                | BPE        |
| Agglutinative languages              | Unigram    |
| Consistent cross-language boundaries | Unigram    |

---

# Final Insight

Each model assumes something different about language:
Here is your section **cleaned and formatted consistently** with the same style as before (proper headings, code blocks, spacing, and tables).

| Model     | Assumption                                   |
|----------|-----------------------------------------------|
| WordLevel | Language = dictionary words                  |
| BPE       | Language = statistical character patterns    |
| Unigram   | Language = probabilistic subword generation  |

---

# Task 20 — Explain Tokenization (Beginner-Friendly)

## What is a tokenizer?

A **tokenizer** is a tool that takes a piece of text (like a sentence) and breaks it into smaller pieces called **tokens**.

You can think of tokens like:

- **Lego bricks** – each token is a brick, and models build meaning by stacking them.
- **Train tickets** – each token is a ticket number the model uses to look up meaning.

In NLP, a token might be:

- a word (`apple`)
- part of a word (`play` + `ing`)
- punctuation (`!`)
- even a single character.

---

## Why tokenization isn’t just `text.split()`

If you try to tokenize by simply splitting on spaces, you will miss many real-world cases:

- **No spaces in some languages**  
  Japanese or Chinese text usually has no spaces, so `split()` would treat the entire sentence as one token.

- **Weird punctuation**  
  Is `don’t` one token or two? What about `#fun`, `https://`, or 😄?

- **Normalization issues**  
  The word `café` can also appear as `café` (same letters but different Unicode representation).  
  A good tokenizer normalizes these so they match.

Because of these cases, tokenization becomes a **non-trivial problem** in NLP.

---

## What makes multilingual tokenization especially interesting

A tokenizer that supports multiple languages must handle:

- Multiple **writing systems** (Latin, Devanagari, Cyrillic, Arabic, etc.)
- Different **whitespace rules** (spaces, no spaces, or mixed spacing)
- Scripts where one visual letter is **multiple Unicode characters**  
  (for example, Devanagari combining marks)
- **Compound words** that look like one word but contain multiple units  
  (for example, German compounds)

This repository addresses these challenges using a **pipeline architecture**, rather than a single `split()` step.

Each stage in the pipeline handles a different responsibility.

---

## Tokenization pipeline (high level)

A typical tokenization pipeline in this project looks like:

```
raw text
   │
   ├─> normalize (unicode forms, casing, etc.)
   │
   ├─> pre-tokenize (split into meaningful chunks)
   │
   └─> model encode (map chunks to token IDs)
```

The **model stage** determines the tokenization strategy.

This project implements three common tokenization models:

- **BPE (Byte-Pair Encoding)**
- **Unigram**
- **WordLevel**

Each model uses a different strategy for deciding how text should be split into tokens.

---

## A concrete, friendly example

Imagine a child learning to read Hindi aloud.

If the word **`किताब`** (kitab, meaning *book*) is broken incorrectly, it becomes much harder to pronounce.

For example, splitting in the wrong place might separate the vowel marks from their consonants.

Devanagari uses **combining marks** such as:

```
ा   ि   ी
```

These marks must stay attached to the consonant they modify.

This repository includes a **Devanagari-aware pretokenizer** that avoids splitting these combinations incorrectly.

That leads to:

- more consistent tokens
- smaller vocabularies
- better model performance

---

## Why you should care

Tokenization is the **first step in every NLP pipeline**.

If tokenization is poor:

- the model receives inconsistent inputs
- vocabulary becomes unnecessarily large
- learning becomes inefficient.

A good tokenizer converts messy real-world text into **consistent, meaningful building blocks**.

Those building blocks are what allow language models to understand and generate text effectively.


# Task 16 — Is This Ready for Production?

Imagine deploying **abctokz** as the tokenizer for a system that processes **millions of Hindi and English documents per day**. Below is a short audit of whether it feels production-ready.

---

# Why I'd Feel Confident Deploying It

## 1. Deterministic Output

The tokenizer produces the **same result every time for the same input**.  
This is important because production systems need reproducible behavior.

For example, calling `encode()` multiple times on the same text produced the **exact same token IDs each time**.

**Evidence**

- `tests/test_determinism.py`
- `set_seed(config.seed)` in `bpe_trainer.py`

---

## 2. Proper Handling of Devanagari Text

The project includes **Devanagari-aware normalization and tokenization**, which is important when working with Hindi text.

It uses **NFC normalization**, which keeps characters like conjuncts intact and avoids breaking words incorrectly.

Example:

```
Café  →  Café
```

**Evidence**

- `src/abctokz/normalizers/devanagari.py`

This makes the tokenizer safer for **multilingual and Indic-language text**.

---

## 3. Clear Error Handling and Versioning

The project includes a **custom exception system** and checks the tokenizer schema version when loading artifacts.

If the saved tokenizer format changes, a clear error (`SchemaVersionError`) is raised instead of silently failing.

**Evidence**

- `exceptions.py`
- `version.py` (`SCHEMA_VERSION = 1`)

This is useful for **safe deployment and upgrades**.

---

# Why I'd Be Hesitant

## 1. Tokenizer Save/Load Does Not Save the Full Pipeline

Currently, the saved tokenizer configuration only stores minimal information like:

```
{
  "model_type": "bpe",
  "schema_version": "1"
}
```

It does **not store the normalizer or pre-tokenizer configuration**.

This means a tokenizer loaded from disk may behave slightly differently from the one that was trained.

**Evidence**

- `tokenizer.py` (save/load logic)

This would be risky in production because tokenization should always be reproducible.

---

## 2. Batch Encoding Is Not Parallel

The `encode_batch()` function processes inputs using a simple loop.

Conceptually it looks like:

```
[tokenizer.encode(x) for x in texts]
```

For very large datasets, this could become a **performance bottleneck**.

**Evidence**

- `tokenizer.py` (`encode_batch` implementation)

---

## 3. Special Token Handling in Decode Is Simple

The `decode()` function removes tokens that look like `<token>` using a simple rule.

This may accidentally remove valid tokens or miss some special tokens.

**Evidence**

- `tokenizer.py` (decode logic)

A safer approach would be to check against the **registered list of special tokens**.

---

# Most Important Thing to Fix First

The **highest priority issue** would be improving the `save()` and `load()` system so that the entire tokenizer pipeline is saved and restored correctly.

This would ensure that a tokenizer behaves **exactly the same across different machines and deployments**, which is critical in production systems.

# Task 17 — One Small Improvement: Fixing Token Offsets in Tokenizer

## What is a Token Offset?

In tokenization, a **token offset** is a pair of integers `(start, end)` that indicates the character positions in the original input string where a specific token originates. For example, if the input text is "hello world" and it produces tokens ["hello", "world"], the offsets might be [(0, 5), (6, 11)], meaning "hello" spans characters 0-4 and "world" spans 6-10.

Offsets are crucial for applications that need to map tokens back to the original text, such as:
- Highlighting tokens in user interfaces
- Aligning tokens with annotations or labels
- Debugging or error reporting that points to specific text locations
- Evaluating tokenization quality by comparing spans

## The Issue in the Code

In the `AugenblickTokenizer.encode()` method in `src/abctokz/tokenizer.py`, there was a bug in how offsets were calculated for subword tokenization (e.g., BPE). When a pre-token (such as a word) was split into multiple subword tokens, all tokens derived from the same pre-token were assigned the same offset span, which was the full span of the pre-token itself.

For instance:
- Input text: "ab"
- Pre-tokenization: ["ab"]
- Model tokenization (BPE): produces tokens ["a", "##b"]
- Incorrect offsets: [(0, 2), (0, 2)] — both tokens get the same span as the pre-token "ab"

This was incorrect because:
- "a" should have offset (0, 1)
- "##b" should have offset (1, 2)

The bug occurred because the code used `len(pre_tok)` for every token's end position, instead of advancing a character cursor per token.

## The Fix Applied

The fix involved modifying the offset calculation loop in `AugenblickTokenizer.encode()` to:
1. Compute the offset for each token based on its actual character length (after stripping the continuation prefix "##").
2. Advance a running `char_offset` pointer by the token's length after each token.

### Code Change in `src/abctokz/tokenizer.py`

Before (incorrect):
```python
for tok_str, tok_id in pairs:
    tok_len = len(tok_str.lstrip("##"))  # strip continuation prefix for offset
    ids.append(tok_id)
    tokens.append(tok_str)
    offsets.append((char_offset, char_offset + len(pre_tok)))  # Wrong: always uses pre_tok length
    is_special = int(tok_str in self._special_tokens)
    special_mask.append(is_special)
    attention_mask.append(1)
```

After (correct):
```python
for tok_str, tok_id in pairs:
    tok_len = len(tok_str.lstrip("##"))  # strip continuation prefix for offset
    ids.append(tok_id)
    tokens.append(tok_str)
    offsets.append((char_offset, char_offset + tok_len))  # Correct: uses token's actual length
    is_special = int(tok_str in self._special_tokens)
    special_mask.append(is_special)
    attention_mask.append(1)
    char_offset += tok_len  # Advance the cursor
```

This ensures that each token gets a precise offset matching its span in the normalized input string.

## New Test File: `tests/unit/test_tokenizer_offsets.py`

To prevent regression, a new unit test was added to verify the offset behavior. This test is important because:
- It provides a concrete example of the bug and ensures it stays fixed.
- It tests the specific case of subword tokenization where offsets were incorrect.
- It can be run independently to validate the fix without requiring full integration tests.

### Full Code of the Test File

```python
"""Unit tests for tokenizer token offsets."""

from abctokz.tokenizer import AugenblickTokenizer
from abctokz.models.bpe import BPEModel
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.vocab import Vocabulary


def test_tokenizer_offsets_align_with_tokens() -> None:
    """Test that token offsets correctly align with subword token spans."""
    vocab = Vocabulary({"<unk>": 0, "a": 1, "##b": 2})
    model = BPEModel(vocab, MergeTable([]))
    tokenizer = AugenblickTokenizer(model)

    enc = tokenizer.encode("ab")
    assert enc.tokens == ["a", "##b"]
    assert enc.offsets == [(0, 1), (1, 2)]
```

### Explanation of the Test

- **Setup**: Creates a minimal BPE model with vocabulary containing "a" and "##b" (no merges, so "ab" splits into these pieces).
- **Execution**: Encodes the string "ab" using the tokenizer.
- **Assertions**:
  - `enc.tokens == ["a", "##b"]`: Verifies the expected tokenization.
  - `enc.offsets == [(0, 1), (1, 2)]`: Ensures offsets are correct (0-1 for "a", 1-2 for "##b").
- **Importance**: This test would fail on the original buggy code (offsets would be [(0, 2), (0, 2)]) and pass after the fix, providing clear evidence of correctness.

## Evidence That the Fix Works

The fix was validated by running the test code manually:

```python
from abctokz.tokenizer import AugenblickTokenizer
from abctokz.models.bpe import BPEModel
from abctokz.vocab.merges import MergeTable
from abctokz.vocab.vocab import Vocabulary

vocab = Vocabulary({"<unk>": 0, "a": 1, "##b": 2})
model = BPEModel(vocab, MergeTable([]))
tok = AugenblickTokenizer(model)
enc = tok.encode("ab")
print(enc.tokens, enc.offsets)  # Output: ['a', '##b'] [(0, 1), (1, 2)]
assert enc.offsets == [(0, 1), (1, 2)]  # Passes
```

Before the fix, this would output `[(0, 2), (0, 2)]` and the assertion would fail.

## Why This Fix is Right and Safe

- **Minimal Change**: Only modifies the offset calculation logic; no changes to tokenization, training, or other features.
- **Focused Impact**: Corrects a specific correctness issue without side effects.
- **High Value**: Ensures offsets are accurate for subword models, enabling proper downstream usage.
- **Tested**: Backed by a regression test to prevent future issues.


```

Tools Used:
Shreya: 
Chatgpt: 27027 Tokens
Github Copilot: 42,200 Tokens

Saee: 
Github Copilot: 56203 Tokens

Raseeca:
Chatgpt: 17161 tokens
Cursor: 35761 tokens

Overall Total
178,352 tokens
