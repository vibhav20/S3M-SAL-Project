"""All paths, model names, seeds and thresholds in one place.

Every directory can be overridden with an environment variable, so the same
code runs locally, on Colab (/content/...) and on Kaggle (/kaggle/working/...).
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _dir(env_var, default):
    return Path(os.environ.get(env_var, ROOT / default))


# ---- paths -------------------------------------------------------------------
DATA_DIR = _dir("SAL_DATA_DIR", "data")
TABLES_DIR = _dir("SAL_TABLES_DIR", "tables")
FEATS_DIR = _dir("SAL_FEATS_DIR", "feats")
RESULTS_DIR = _dir("SAL_RESULTS_DIR", "results")
FIGS_DIR = _dir("SAL_FIGS_DIR", "figs")

LIBRISPEECH_DIR = DATA_DIR / "LibriSpeech"                 # dev-clean/, test-clean/
ALIGNMENTS_DIR = DATA_DIR / "librispeech_alignments"       # dev-clean/, test-clean/ (.TextGrid)
NLTK_DATA_DIR = DATA_DIR / "nltk_data"                     # corpora/wordnet/
MASE_DIR = DATA_DIR / "mase"                               # slu_splits/{fluent_speech_commands,snips_close_field}
FSC_DIR = DATA_DIR / "fsc"
SNIPS_DIR = DATA_DIR / "snips"

LIBRISPEECH_SPLITS = ["dev-clean", "test-clean"]

# Make `from nltk.corpus import wordnet` use our project-local copy.
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))

# ---- experiment settings (filled in as later steps are implemented) -----------
SEEDS = [0, 1, 2, 3, 4]
SAMPLE_SIZE = 10_000          # words sampled per seed (paper Sec. 2.2)
PHONETIC_THRESHOLD = 0.4      # normalized phone edit distance for near-homophones
SPEAKERS = [5142, 2412, 6313, 1580, 2277]   # single-speaker runs (paper Sec. 3.5)
