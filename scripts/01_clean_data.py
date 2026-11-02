"""Step 01: LibriSpeech TextGrids -> tables/librispeech_words.csv (one row per spoken word).
Keeps words that have a CMUdict pronunciation and at least one WordNet synonym.
Run: python scripts/01_clean_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from textgrids import TextGrid

import config
import utils

cmu = utils.load_cmudict()
synonym_cache = {}
rows = []
skipped = 0

for split in config.LIBRISPEECH_SPLITS:
    paths = sorted((config.ALIGNMENTS_DIR / split).glob("*/*/*.TextGrid"))
    print(f"{split}: {len(paths)} TextGrids")
    for path in paths:
        utt_id = path.stem
        speaker, chapter, _ = utt_id.split("-")
        audio_path = f"LibriSpeech/{split}/{speaker}/{chapter}/{utt_id}.flac"

        for interval in TextGrid(str(path))["words"]:
            word = interval.text.strip()
            if word == "":
                continue  # silence
            if word not in synonym_cache:
                synonym_cache[word] = utils.get_synonyms(word)
            phones = utils.get_phones(word, cmu)
            synonyms = synonym_cache[word]
            if phones is None or len(synonyms) == 0:
                skipped += 1
                continue
            rows.append({
                "split": split,
                "utt_id": utt_id,
                "speaker": int(speaker),
                "word": word,
                "start": interval.xmin,
                "end": interval.xmax,
                "audio_path": audio_path,  # relative to config.DATA_DIR
                "phones": " ".join(phones),
                "synonyms": " ".join(synonyms),
            })

df = pd.DataFrame(rows)
config.TABLES_DIR.mkdir(exist_ok=True)
out = config.TABLES_DIR / "librispeech_words.csv"
df.to_csv(out, index_label="row_id")

print(f"kept {len(df)} words, skipped {skipped} (no pronunciation or no synonym)")
print(f"{df.word.nunique()} distinct words, {df.speaker.nunique()} speakers")
print(f"saved {out}")
