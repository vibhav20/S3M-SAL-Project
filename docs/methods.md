# Methods and caveats

How we build the word table and the word pairs (steps 01–02), what we copied from the
authors' code ([juice500ml/phonetic_semantic_probing](https://github.com/juice500ml/phonetic_semantic_probing)),
and the known weak spots. Numbers are from our run (seed 0 unless stated).

## 1. Data

| Source | Used for |
|---|---|
| LibriSpeech dev-clean + test-clean (80 speakers, ~10.8 h) | the spoken words |
| TextGrid alignments (Lugosch, Zenodo 2619474) | start/end time of every word |
| CMU Pronouncing Dictionary (`cmudict==1.0.15` pip package) | how a word sounds (phones) |
| WordNet (via NLTK 3.8.1) | what a word means (synonyms) |

CMUdict comes from the pip package (as in the authors' code), not the `cmudict-0.7b` file
linked in the proposal. It is the newer, maintained version of the same dictionary.

## 2. Word table (`scripts/01_clean_data.py`)

One row per spoken word from the `words` tier of every TextGrid. A word is kept only if:

1. it is in CMUdict, and
2. WordNet gives it at least one synonym.

Phones = the **first** CMUdict pronunciation with stress digits removed (`D AO1 G` → `D AO G`).

Result: 70,746 kept, 36,281 dropped (9,788 distinct words). Dropped words include silences,
`<unk>` (words the aligner did not know) and words with no WordNet entry (many names, and
function words like "the", "of", "and").

## 3. Phonetic distance

Edit distance (insertions, deletions, substitutions) between two phone sequences, divided by
the length of the longer one. 0 = identical pronunciation, 1 = nothing in common.

- dog `D AO G` vs dig `D IH G` → 1 substitution / 3 = **0.33**
- dog vs puppy `P AH P IY` → **1.00**

**Threshold 0.4.** Only 0.16% of random word pairs fall at or below 0.4; 0.4 is the 0.1th
percentile of random-pair distances (`notebooks/01_inspect_pairs.ipynb`). So ≤ 0.4 means "far more
alike than chance".

## 4. Word maps (`scripts/02_build_pairs.py`)

Built per run, from a sample of word tokens:

- **seed runs (0–4):** 10,000 random tokens from the whole table (~3,300 distinct words);
- **speaker runs (5142, 2412, 6313, 1580, 2277):** all tokens of one speaker (~1,100 tokens).

Then for the distinct words in the sample:

- **Synonyms:** WordNet synonyms that are also in the sample **and sound different**
  (distance > 0.4). Example: poor – miserable (1.00).
- **Near-homophones:** words in the sample that **sound alike** (0 < distance ≤ 0.4) **and are not
  synonyms** (in either direction). Example: broad – bread (0.25).

## 5. The five pair types

Each pair is two word tokens (two separate audio clips).

| Pair type | Built how | Why we need it |
|---|---|---|
| random | shuffle all tokens, pair each with its shuffled partner | baseline: how similar any two words are |
| synonym | each token with every token of its synonyms | same meaning, different sound |
| near-homophone | each token with every token of its near-homophones | similar sound, different meaning |
| same speaker | shuffle within each speaker | how much the voice alone makes words look similar |
| same word | shuffle within each word | upper bound: the same word said twice |

Shuffles use a fixed seed (42), like the authors' code.

### Why the same-speaker data?

The speaker shows up in the experiments in two ways:

1. **Same-speaker pairs (in the seed runs).** Two *different* words from the *same* speaker.
   The model also encodes who is talking (voice, microphone, recording). If two words from the
   same person already look similar, a high similarity for synonyms or near-homophones could
   just be a speaker effect. This pair type tells us how big that effect is, so the
   phonetic/semantic comparison can be read relative to it.
2. **Single-speaker runs (paper Sec. 3.5).** Every word comes from one speaker, so the voice is
   held fixed and speaker differences cannot explain any result. If near-homophones are still
   closer than synonyms here, the finding is about sound vs. meaning, not about who spoke.
   These runs have no same-speaker pair type, since every pair is from the same speaker.

## 6. Caveats

**Choices copied from the authors (kept so our results are comparable):**

1. **True homophones (distance 0) are excluded.** night/knight, two/too, son/sun have identical
   CMUdict phones and are never near-homophone pairs; 48 such groups in seed 0. The authors' code
   requires `0 < distance`, but the paper's text does not explain why. Our reading: words with
   identical canonical pronunciation are, acoustically, the same word, so they would behave like
   same-word pairs rather than "similar-sounding" ones. Some are also just spelling variants
   (honorable/honourable).
2. **Only the first pronunciation is used.** 1,407 of 9,788 words have more than one. E.g. "read"
   is always `R EH D`, so read/red count as identical even when the speaker said "reed".
3. **Pronunciations come from the dictionary, not the audio.** The distance uses canonical
   CMUdict phones, not what the speaker actually said (accent, reduction, fast speech).
4. **Synonyms ignore word sense and context.** All WordNet senses count, so we get odd pairs from
   rare senses: as–a (arsenic / the letter A), one–i (Roman numeral), dog–cad. Single letters
   (a, b, c, …) stay in the table because WordNet has entries for them.
5. **Synonyms that sound alike are dropped from both groups.** gave/give, got/get, carrying/carry
   (730 pairs in seed 0) are too close in sound to be synonyms (> 0.4) and, being synonyms, are
   excluded from near-homophones too, so they are in neither group.
6. **Frequent words dominate token pairs.** Every token is paired with every token of the other
   word, so as–a alone gives 40k synonym token pairs; the top 5 word pairs cover 36% of synonym
   token pairs. The similarity step must therefore average per word pair first, then across word
   pairs (as the authors do), so as–a counts once.
7. **Random and same-speaker pairs can be the same word** (0.4% and 0.7% of pairs). Too few to matter.
8. **Sampling is by token, not by word.** Frequent words appear more often in the 10,000-token
   sample, which shapes which synonym/homophone pairs exist.

**Our own implementation choices:**

9. **Own edit distance instead of `jiwer`.** Same number as the authors' `jiwer.wer` on phone
   strings; one less package.
10. **CSV instead of pickle/parquet**, so files open anywhere. The samples are committed because
    `pandas.sample` can change between pandas versions; the pair files (`tables/pairs_*.csv`,
    73 MB) are gitignored and can be regenerated in ~6 minutes.
11. **Alignment quality.** Word boundaries come from forced alignment, which can be off by a few
    tens of milliseconds; very short words (median 0.26 s) are the most affected when we cut the
    audio in the next step.
