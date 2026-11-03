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

Sample of . trans . txt data in LibriSpeech:
	6241-66616-0000 BUT THIS POWER OF DISCERNMENT WAS DENIED THEM AND ONLY IN AFTER YEARS ...
	The ID 6241-66616-0000 means speaker 6241, chapter 66616, utterance 0000, and it matches the .flac audio file name.
BUT this is not directly used anywhere, we use the Textgrid files from the alignments folder itself to get the words used and their timestamps!

## 2. Word table (`scripts/01_clean_data.py`)

It keeps one row per spoken word from the `words` tier of every TextGrid. 
A word is kept only if:
1. it is in CMUdict, and
2. WordNet gives it at least one synonym.

Phones = the **first** CMUdict pronunciation with stress digits removed (`D AO1 G` → `D AO G`).

Result: tables/librispeech_words.csv where we have words from TextGrid along with their Phones and synonyms 
(and other metadata columns like start, end, file, speaker etc)

Statistics:
70,746 words in this csv file(9,788 distinct words, 80 speakers), 36,281 words were dropped . 
Dropped words include silences, `<unk>` (words the aligner did not know) and words with no WordNet entry (many names, and
function words like "the", "of", "and").

## 3. Phonetic distance Calcuation:

Edit distance (insertions, deletions, substitutions) between two phone sequences, divided by
the length of the longer one. 0 = identical pronunciation, 1 = nothing in common.

- dog `D AO G` vs dig `D IH G` → 1 substitution / 3 = **0.33**
- dog vs puppy `P AH P IY` → **1.00**

**Threshold 0.4.** Only 0.16% of *random word pairs* fall at or below 0.4; 0.4 is the 0.1th
percentile of random-pair distances (`notebooks/01_inspect_pairs.ipynb`). So ≤ 0.4 means "far more
alike than chance".

## 4. Word maps (`scripts/02_build_pairs.py`)
Input: tables/librispeech_words.csv and config file->phone_threshold, seed value, speaker id, sample size 

Workflow:
1) pick sample size worth random rows (seed value) from librispeech_words.csv
Samples picked, gets saved directly as sample_seed0.csv. Now,
2) build_synonym_map: word → its synonyms that are in the sample AND sound different (> 0.4)
3) build_homophone_map: word → sample words that sound alike (0 < d ≤ 0.4) AND aren't synonyms
Mapping done, these are stored as dictionaries: 
synonym_map   = {"one": {"i", "single"}, "ability": {"power", ...}, ...}   # word → set of words
homophone_map = {"broad": {"bread", ...}, "bound": {"drowned", ...}, ...}
then  wordmap_seed0.csv stores one rows of data with (word, other_word) as columns which makes this a one to one mapping.
word_map_seed file is only as a checkpoint, the code uses dictionary directly for any further usecase from here:

4) make token pairs: (A,B are pairs from wordmap_seed0)
	synonym_pairs(sample, synonym_map)       every token of A × every token of each synonym B
	homophone_pairs(sample, homophone_map)   every token of A × every token of each sound-alike B
	random_pairs(sample)                     shuffle all 10,000 tokens, pair each with its partner
	same_speaker_pairs(sample)               shuffle within each speaker, pair each with its partner
	same_word_pairs(sample)                  shuffle within each word, pair each with its partner

all these 5 lists combine repping one row per pair, with a pair_type column and gets saved as: pairs_seed0.csv;
(seed 0 → sample_seed0 → dictionaries for seed 0 → pairs_seed0)

Built per run, from a sample of word tokens:
- **seed runs (0–4):** 10,000 random tokens from the whole table (~3,300 distinct words);
- **speaker runs (5142, 2412, 6313, 1580, 2277):** all tokens of one speaker (~1,100 tokens).

for the distinct words in the sample:
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

## 6. Output tables and how they connect

```
librispeech_words.csv     every usable spoken word (70,746 rows), with phones + synonyms
        │  seed picks 10,000 rows (or: all rows of one speaker)
        ▼
sample_{run}.csv          a subset of librispeech_words.csv, same columns, same row_ids
        │  compare the distinct words in the sample (no randomness here)
        ▼
wordmap_{run}.csv         which WORDS are related:  "ability ~ power"  (synonym)
        │  expand every related word pair into all its audio clips, add the other 3 types
        ▼
pairs_{run}.csv           which CLIPS to compare:   row 45168 ("i") vs row 42608 ("one")
```

`{run}` is `seed0` … `seed4` or `spk5142`, `spk2412`, ….

| Table | Step | One row is | Randomness | Example |
|---|---|---|---|---|
| `librispeech_words.csv` | 01 | one spoken word token (a word at one moment in one recording) | none | "i", speaker 237, 1.89–2.00 s of 237-126133-0022 |
| `sample_{run}.csv` | 02 | a token picked for this run | **the seed** chooses the 10,000 tokens | same columns and `row_id` as the word table |
| `wordmap_{run}.csv` | 02 | two related **words** | none: same sample → same word map | `ability, power, synonym`; `abashed, attached, homophone` |
| `pairs_{run}.csv` | 02 | two **audio clips** to compare (by `row_id`) | shuffles for random / same-speaker / same-word (seed 42) | `synonym, 45168, 42608, i, one` |

- **`row_id`** is the permanent ID of a token in `librispeech_words.csv`; every later table refers
  to tokens by it, so a pair can always be traced back to its audio file and start/end time.
- **Why sample?** Feature extraction runs on 10k clips instead of 70k, and 5 seeds give 5
  independent samples, so results are reported as a mean over seeds rather than one lucky draw.
- **Word map vs. pairs.** The word map only says which *words* are related. A word occurs many
  times in the sample (different sentences and speakers), so one word-map row expands into many
  clip pairs: "i ~ one" becomes every *i* clip × every *one* clip. Hence ~7k synonym word pairs
  but ~236k synonym clip pairs per seed.
- **Pair files in git.** The 10 `pairs_*.csv` (73 MB) are gitignored; a zipped copy,
  `tables/pairs.zip` (14 MB), is committed. Unzip from the project root with
  `unzip tables/pairs.zip`, or regenerate with `python scripts/02_build_pairs.py`. If the pairs
  are ever rebuilt, refresh the zip: `rm tables/pairs.zip && zip -9 tables/pairs.zip tables/pairs_*.csv`.
- **Pair types without a word map** come straight from the sample: random (shuffle all tokens),
  same speaker (shuffle within each speaker), same word (shuffle within each word).

How the next steps use them:

- **Features (step 03):** for every row of `sample_{run}.csv`, cut `audio_path` from `start` to
  `end`, run the model, save one vector per layer under its `row_id`.
- **Similarity (step 04):** for every row of `pairs_{run}.csv`, look up both vectors by `row_id`,
  compute cosine similarity per layer, average per word pair, then per pair type.

## 7. Features (`scripts/03_extract_features.py`, run on Colab)

Model: HuBERT-large (`facebook/hubert-large-ll60k`): CNN encoder (one frame per 20 ms) followed by
24 Transformer layers, 1024-dim. With `output_hidden_states=True` it returns 25 layers
(transformer input + each layer's output).

- **Which tokens:** the union of all sample files (40,333 tokens from 5,239 utterances). Each
  token is extracted once and every seed/speaker run looks its vectors up by `row_id`.
- **Audio slicing:** cut the word's samples (`start`–`end`) out of the recording and run the model
  on that clip alone; average all its frames, per layer.
- **Feature slicing:** run the model on the whole recording once; convert each word's start/end
  time to frame numbers and average those frames (inclusive), per layer.
- **Output:** `feats/<setting>/part_XXX.npz`, one part per 250 utterances, holding `row_ids` and
  `feats` of shape (n, 25, 1024) in float16. Parts are skipped if they exist, so a run resumes
  after a disconnect.
- Audio is loaded with `librosa` at 16 kHz mono (as in the authors' code); the model's own
  feature extractor normalizes each input (each word clip separately for audio slicing).
  One input at a time, no padding.

| Setting folder | Slicing | Frame formula |
|---|---|---|
| `hubert-large-ll60k_audio_mean` | audio | (not used) |
| `hubert-large-ll60k_feature_mean` | feature | fixed: `int(t × 16000) // 320` (50 frames/s) |
| `hubert-large-ll60k_feature_mean_authors` | feature | authors': `int(t / 2 × 16000) // 320` (25 frames/s) |

**Frame formula (`--formula`).** HuBERT produces 50 frames per second, but the authors'
`extract_features.py` divides the time by 2 before converting, so a word at 4.0 s is read from
frames around 2.0 s. This looks like a bug that affects only feature slicing (audio slicing uses
all frames of the clip). We run both: `fixed` as our main version, `authors` to reproduce their
Figure 1 and see whether the "squashing" of feature-sliced representations survives the fix.

**Normalized vs. unnormalized** (paper Fig. 1) is not a feature setting: the same similarities
are plotted raw and with the random-pair similarity of the same layer subtracted.

## 8. Caveats

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
12. **Centroid pooling (Sec. 3.2, not run yet).** We pick the frame with the highest average
    cosine similarity to all frames, as the paper describes. The authors' code divides frames by
    their *squared* norm and then picks the frame closest (Euclidean) to the mean, which is not
    exactly cosine.
13. **Features stored in float16** to halve storage (~2 GB per setting); the model itself runs in
    float32. Precision loss is far below the differences we compare.
