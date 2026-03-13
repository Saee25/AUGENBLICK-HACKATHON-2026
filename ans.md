Here is your **cleaned and consistently formatted `.md` content**.
I **did not change the content**, only fixed:

* heading hierarchy
* horizontal rules
* code block formatting
* tables
* spacing
* consistent Markdown syntax
* removed HTML artifacts
* ensured valid fenced code blocks

You can **copy this directly into a `.md` file**.

---

````markdown
# Tokenizer Pipeline Trace and Architecture Analysis

---

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

You can paste this directly into your `.md` file.

```markdown
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
```


AI Tools Used:
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