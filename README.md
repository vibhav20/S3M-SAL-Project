# Self-Supervised Speech Representations are More Phonetic than Semantic — reimplementation

Course project for Speech Analysis and Linguistics (Priyanshu Panchasara, Vibhav Pandey).
Reimplements [Choi et al., Interspeech 2024](https://arxiv.org/abs/2406.08619)
([authors' code](https://github.com/juice500ml/phonetic_semantic_probing)).
The proposal is in [`docs/`](docs/).

## Layout

```
config.py      all paths / seeds / thresholds (paths overridable via SAL_* env vars)
utils.py       shared helper functions (phonetics, synonyms, word maps, pair types)
data/          raw downloads, never edited (gitignored)
scripts/       run in order; each reads the previous step's output
tables/        word table, samples, word maps (committed); pairs_*.csv (gitignored, regenerate)
feats/         cached embeddings (gitignored)
results/       similarity / accuracy CSVs
figs/          plots for slides and report
notebooks/     exploration only; final results come from scripts/
docs/          proposal, report
```

## Setup

**Environment.** Python >= 3.10 with PyTorch plus `requirements.txt`.
- Local (Vibhav): reuses the existing venv `~/envs/scratch` (has torch already).
  `source ~/envs/scratch/bin/activate`, then install missing packages as each step needs them.
- Fresh machine / Colab / Kaggle: `pip install -r requirements.txt` (torch is preinstalled on Colab/Kaggle).

**Data.**
```bash
bash scripts/00_download_data.sh      # ~1.3 GB, resumable, safe to re-run
```
Downloads LibriSpeech dev-clean + test-clean, their TextGrid word alignments,
WordNet and the MASE split repo into `data/`. Fluent Speech Commands and Snips
SmartLights need access requests; the script prints where to get them.

## Steps

| Step | Script | Status |
|---|---|---|
| 00 | `scripts/00_download_data.sh` — raw data | done |
| 01 | `scripts/01_clean_data.py` — TextGrids → `tables/librispeech_words.csv` (~10 s) | done |
| 02 | `scripts/02_build_pairs.py` — word maps + 5 pair types, 5 seeds + 5 speakers (~6 min) | done |
| 03 | extract layerwise features (GPU) | todo |
| 04 | layerwise similarity + plots | todo |
| 05 | intent classification (BoW baseline vs. probes) | todo |

Notebooks `00_inspect_data` and `01_inspect_pairs` sanity-check the data and the pairs
(including the 0.4 phonetic-distance threshold). Methods and known caveats: [`docs/methods.md`](docs/methods.md).
