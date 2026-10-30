#!/usr/bin/env bash
# Step 00: download all raw data into data/.
#
# Adapted from the authors' datasets/download_datasets.sh
# (https://github.com/juice500ml/phonetic_semantic_probing), keeping only what
# our proposal needs:
#   kept    : LibriSpeech dev-clean + test-clean, LibriSpeech TextGrid alignments,
#             MASE repo (FSC/Snips splits)
#   added   : WordNet (their code assumes nltk.download('wordnet') was run)
#   dropped : fastText vectors, SemCor, Multilingual Spoken Words, CC-CEDICT
#             (only used for extra sanity checks / the multilingual experiment)
# CMUdict is not downloaded: it ships inside the `cmudict` pip package.
#
# Needs only wget, tar, unzip and git. Safe to re-run: finished items are
# skipped and interrupted downloads resume (wget -c).
#
# Usage:  bash scripts/00_download_data.sh
#         SAL_DATA_DIR=/content/data bash scripts/00_download_data.sh   # e.g. Colab

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${SAL_DATA_DIR:-$ROOT/data}"
DL="$DATA/_downloads"          # archives are kept here so re-runs never re-download
mkdir -p "$DATA" "$DL"

fetch() {  # fetch URL FILE
    echo ">> downloading $2"
    wget -c --tries=20 --retry-connrefused --waitretry=5 --progress=dot:giga -O "$DL/$2" "$1"
}

# 1. LibriSpeech dev-clean + test-clean (flac + transcripts) -> data/LibriSpeech/
for split in dev-clean test-clean; do
    if [ -d "$DATA/LibriSpeech/$split" ]; then
        echo "== LibriSpeech/$split already present, skipping"
    else
        fetch "https://www.openslr.org/resources/12/$split.tar.gz" "$split.tar.gz"
        tar -xzf "$DL/$split.tar.gz" -C "$DATA"
    fi
done

# 2. TextGrid word alignments (Lugosch, Zenodo 2619474) -> data/librispeech_alignments/
#    The zip covers all of LibriSpeech; we only extract dev-clean and test-clean.
if [ -d "$DATA/librispeech_alignments/dev-clean" ] && [ -d "$DATA/librispeech_alignments/test-clean" ]; then
    echo "== librispeech_alignments already present, skipping"
else
    fetch "https://zenodo.org/record/2619474/files/librispeech_alignments.zip" "librispeech_alignments.zip"
    mkdir -p "$DATA/librispeech_alignments"
    unzip -q -o "$DL/librispeech_alignments.zip" '*dev-clean/*' '*test-clean/*' -d "$DATA/librispeech_alignments"
fi

# 3. WordNet for NLTK -> data/nltk_data/corpora/wordnet/
#    Fetched directly (same file nltk.download uses) so no Python package is needed.
if [ -d "$DATA/nltk_data/corpora/wordnet" ]; then
    echo "== WordNet already present, skipping"
else
    fetch "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip" "wordnet.zip"
    mkdir -p "$DATA/nltk_data/corpora"
    unzip -q -o "$DL/wordnet.zip" -d "$DATA/nltk_data/corpora"
fi

# 4. MASE repo: FSC + Snips train/test splits -> data/mase/
if [ -d "$DATA/mase/.git" ]; then
    echo "== mase already cloned, skipping"
else
    git clone --depth 1 https://github.com/maseEval/mase.git "$DATA/mase"
fi

# 5. FSC and Snips audio are gated: they cannot be downloaded by script.
cat <<EOF

==================================================================
Done with the automatic downloads. Manual steps still needed:

 Fluent Speech Commands  -> $DATA/fsc/
   Request access: https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/

 Snips SmartLights       -> $DATA/snips/
   Request access: https://github.com/sonos/spoken-language-understanding-research-datasets
   Then (per MASE README) copy data/mase/slu_splits/snips_close_field/*
   into the extracted snips_slu_data_v1.0/ folder.
==================================================================
EOF
