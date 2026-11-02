"""Step 02: word table -> word maps and the 5 pair types, for each seed and each single speaker.
Writes tables/sample_{run}.csv, wordmap_{run}.csv, pairs_{run}.csv  (run = seed0..4, spk5142, ...)
Run: python scripts/02_build_pairs.py              (all seeds + all speakers)
     python scripts/02_build_pairs.py --seed 0     (one seed)
     python scripts/02_build_pairs.py --speaker 5142
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--speaker", type=int, default=None)
args = parser.parse_args()

words = pd.read_csv(config.TABLES_DIR / "librispeech_words.csv", index_col="row_id")
print(f"word table: {len(words)} words")

# which runs to do
runs = []
if args.seed is not None:
    runs.append(("seed", args.seed))
if args.speaker is not None:
    runs.append(("speaker", args.speaker))
if len(runs) == 0:
    runs = [("seed", s) for s in config.SEEDS] + [("speaker", s) for s in config.SPEAKERS]

threshold = config.PHONETIC_THRESHOLD

for kind, value in runs:
    start_time = time.time()
    if kind == "seed":
        name = f"seed{value}"
        sample = utils.sample_words(words, seed=value, n=config.SAMPLE_SIZE)
    else:
        name = f"spk{value}"
        sample = utils.sample_words(words, speaker=value)
    print(f"\n== {name}: {len(sample)} words, {sample.word.nunique()} distinct")

    synonym_map = utils.build_synonym_map(sample, threshold)
    all_synonyms = utils.build_synonym_map(sample, None)  # unfiltered, to exclude synonyms from homophones
    homophone_map = utils.build_homophone_map(sample, all_synonyms, threshold)

    pairs = {
        "random": utils.random_pairs(sample),
        "synonym": utils.synonym_pairs(sample, synonym_map),
        "homophone": utils.homophone_pairs(sample, homophone_map),
        "same_speaker": utils.same_speaker_pairs(sample),
        "same_word": utils.same_word_pairs(sample),
    }
    if kind == "speaker":
        del pairs["same_speaker"]  # every pair is the same speaker here

    # save
    sample.to_csv(config.TABLES_DIR / f"sample_{name}.csv")

    map_rows = []
    for relation, word_map in [("synonym", synonym_map), ("homophone", homophone_map)]:
        for word in sorted(word_map):
            for other in sorted(word_map[word]):
                map_rows.append({"word": word, "other_word": other, "relation": relation})
    pd.DataFrame(map_rows).to_csv(config.TABLES_DIR / f"wordmap_{name}.csv", index=False)

    pair_rows = []
    for pair_type, pair_list in pairs.items():
        for left, right in pair_list:
            pair_rows.append({
                "pair_type": pair_type,
                "left_row": left,
                "right_row": right,
                "left_word": sample.at[left, "word"],
                "right_word": sample.at[right, "word"],
            })
    pd.DataFrame(pair_rows).to_csv(config.TABLES_DIR / f"pairs_{name}.csv", index=False)

    print(f"word pairs: {sum(len(v) for v in synonym_map.values())} synonym, "
          f"{sum(len(v) for v in homophone_map.values())} homophone")
    for pair_type, pair_list in pairs.items():
        print(f"  {pair_type:13s} {len(pair_list):7d} token pairs")
    print(f"  done in {time.time() - start_time:.0f}s")
