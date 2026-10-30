# Self-Supervised Speech Representations are More Phonetic than Semantic — reimplementation

Course project for Speech Analysis and Linguistics (Priyanshu Panchasara, Vibhav Pandey).
Reimplements [Choi et al., Interspeech 2024](https://arxiv.org/abs/2406.08619)
([authors' code](https://github.com/juice500ml/phonetic_semantic_probing)).
The proposal is in [`docs/`](docs/).

## Layout

```
config.py      all paths / seeds / thresholds (paths overridable via SAL_* env vars)
data/          raw downloads, never edited (gitignored)
src/           reusable functions imported by the scripts
scripts/       run in order; each reads the previous step's output
tables/        word tables, pair lists (small, committed so everyone uses the same pairs)
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
| 01 | build word table from TextGrids | todo |
| 02 | phonetic-distance threshold check | todo |
| 03 | build the 5 pair types | todo |
| 04 | extract layerwise features (GPU) | todo |
| 05 | layerwise similarity + plots | todo |
| 06 | intent classification (BoW baseline vs. probes) | todo |
