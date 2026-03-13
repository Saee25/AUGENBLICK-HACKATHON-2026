"""Task 8 — What Does the Normalizer Actually Do?

Analyzes how DevanagariNormalizer transforms two specific phrases before
the model sees them, then shows what DevanagariAwarePreTokenizer does next.
"""

import unicodedata
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from abctokz.normalizers.devanagari import DevanagariNormalizer
from abctokz.normalizers.whitespace import WhitespaceNormalizer
from abctokz.pretokenizers.devanagari_aware import DevanagariAwarePreTokenizer

# ── Phrases ──────────────────────────────────────────────────────────────────
PHRASE_I  = "\u0906\u092F\u094B \u0932\u093E\u0932, \u0938\u092D\u0908 \u091A\u093E\u092F\u094B, \u091D\u0942\u0932\u0947\u0932\u093E\u0932!"
PHRASE_II = "\u0917\u0923\u092A\u0924\u0940 \u092C\u092A\u094D\u092A\u093E \u092E\u094B\u0930\u092F\u093E, \u092A\u0941\u0922\u091A\u094D\u092F\u093E \u0935\u0930\u094D\u0937\u0940 \u0932\u0935\u0915\u0930 \u092F\u093E!"

phrases = {
    "I  (Sindhi folk)": PHRASE_I,
    "II (Marathi Ganesh)": PHRASE_II,
}

norm_dev = DevanagariNormalizer(nfc_first=True, strip_zero_width=False)
norm_ws  = WhitespaceNormalizer(collapse=True, strip=True)
pretok   = DevanagariAwarePreTokenizer(split_on_whitespace=True, split_on_script_boundary=True)


def char_detail(text: str) -> list[str]:
    return [f"U+{ord(c):04X} {unicodedata.name(c, '?')} ({c!r})" for c in text]


def show_diff(raw: str, norm: str, label: str) -> None:
    if raw == norm:
        print(f"  [{label}] IDENTICAL after normalization")
    else:
        print(f"  [{label}] CHANGED after normalization")
        # find first differing position
        min_len = min(len(raw), len(norm))
        for i in range(min_len):
            if raw[i] != norm[i]:
                print(f"    First difference at index {i}:")
                print(f"      raw : U+{ord(raw[i]):04X}  {unicodedata.name(raw[i], '?')} ({raw[i]!r})")
                print(f"      norm: U+{ord(norm[i]):04X}  {unicodedata.name(norm[i], '?')} ({norm[i]!r})")
                break
        if len(raw) != len(norm):
            print(f"    Length changed: {len(raw)} → {len(norm)} chars")


print("=" * 70)
print("TASK 8 — NORMALIZER + PRE-TOKENIZER ANALYSIS")
print("=" * 70)

for name, phrase in phrases.items():
    print(f"\n{'─' * 70}")
    print(f"Phrase {name}")
    print(f"{'─' * 70}")

    # ── Step 1: raw input ────────────────────────────────────────────────────
    print(f"\nRAW INPUT  : {phrase!r}")
    print(f"           : {phrase}")
    print(f"  len      : {len(phrase)} code points")
    print(f"  NFC form : {unicodedata.normalize('NFC', phrase)!r}")
    print(f"  NFKC form: {unicodedata.normalize('NFKC', phrase)!r}")
    nfc_eq  = phrase == unicodedata.normalize("NFC",  phrase)
    nfkc_eq = phrase == unicodedata.normalize("NFKC", phrase)
    print(f"  is NFC?  : {nfc_eq}   |   is NFKC? : {nfkc_eq}")

    # ── Step 2: DevanagariNormalizer ─────────────────────────────────────────
    after_dev = norm_dev.normalize(phrase)
    print(f"\nAFTER DevanagariNormalizer (NFC + exotic-whitespace fix):")
    print(f"  result   : {after_dev!r}")
    show_diff(phrase, after_dev, "DevanagariNormalizer")

    # ── Step 3: WhitespaceNormalizer ─────────────────────────────────────────
    after_ws = norm_ws.normalize(after_dev)
    print(f"\nAFTER WhitespaceNormalizer (collapse + strip):")
    print(f"  result   : {after_ws!r}")
    show_diff(after_dev, after_ws, "WhitespaceNormalizer")

    # ── Step 4: Pre-tokenize ─────────────────────────────────────────────────
    pre_tokens = pretok.pre_tokenize(after_ws)
    print(f"\nAFTER DevanagariAwarePreTokenizer:")
    for i, tok in enumerate(pre_tokens):
        print(f"  [{i:2d}] {tok!r:30s}  (len={len(tok)})")

    # ── Step 5: Fate of special chars ────────────────────────────────────────
    print(f"\nFATE OF SPECIAL CHARACTERS in pre-tokens:")
    special_map = {",": "COMMA", "!": "EXCLAMATION MARK", " ": "SPACE"}
    for char, cname in special_map.items():
        indices = [i for i, tok in enumerate(pre_tokens) if char in tok]
        if indices:
            print(f"  {cname!r} ({char!r}) found in pre-tokens at indices: {indices}")
            for idx in indices:
                print(f"       pre-token[{idx}] = {pre_tokens[idx]!r}")
        else:
            print(f"  {cname!r} ({char!r}) : NOT present in any pre-token")

print(f"\n{'=' * 70}")
print("NFC vs NFKC — KEY DIFFERENCE FOR DEVANAGARI")
print("=" * 70)
print("""
NFC  (Canonical Decomposition + Canonical Composition):
  - Decomposes characters into canonical equivalents, then recomposes.
  - Preserves combining marks like matras (e.g. 'ा', 'ी') as they are.
  - SAFE for Devanagari: matras and half-consonants stay intact.

NFKC (Compatibility Decomposition + Canonical Composition):
  - More aggressive: also decomposes compatibility-equivalent characters
    (e.g. ligatures, superscripts, half-width forms).
  - For Devanagari, NFKC can break conjunct consonants (e.g. 'क्ष' → k+sh),
    changing the phoneme the character encodes. That would corrupt the text.
  - THIS LIBRARY USES NFC (see devanagari.py line 7: "We apply NFC (not NFKC)
    because NFKC can collapse Devanagari combining marks in ways that change
    the visual and phonetic form of the character.")

Why NFC over identity?
  - Ensures consistent canonical form so that the same visual character
    always maps to the same byte sequence, regardless of how it was entered
    (e.g. via keyboard vs. copy-paste from different editors).
""")

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Both phrases are already in NFC form, so DevanagariNormalizer is a no-op
on the actual Unicode codepoints. WhitespaceNormalizer also leaves them
unchanged (single spaces, no leading/trailing whitespace).

DevanagariAwarePreTokenizer:
  - Splits on whitespace → splits each word.
  - Commas and exclamation marks are classified as 'other' script tokens.
    Per _split_by_script(), 'other' clusters attach to the CURRENT script
    group rather than forcing a split. So "लाल," becomes ONE pre-token
    ["लाल,"] — the comma is appended to the Devanagari word, not split off.
  - Spaces are consumed by text.split() in the whitespace pass — they
    DISAPPEAR from the pre-token list entirely.
""")
 