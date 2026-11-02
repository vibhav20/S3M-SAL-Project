"""Step 03: sampled words -> layer-wise SSL features (tables/features/{run}_model-..._pool-....npy, [n_words, n_layers, dim]).
Run: python scripts/extract_features.py --seed 0 --model microsoft/wavlm-large --device cuda:0
     python scripts/extract_features.py --speaker 5142
     python scripts/extract_features.py            (all seeds + all speakers)

     python scripts/extract_features.py --seed 0 --model microsoft/wavlm-large --pooling mean --slice True --device cuda:0
     python scripts/extract_features.py --speaker 5142 --model facebook/hubert-large-ll60k --device cuda:0
     python scripts/extract_features.py            # all seeds + all speakers
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, WhisperFeatureExtractor, Wav2Vec2FeatureExtractor

import config

SAMPLE_RATE = 16000
FRAMES_PER_SECOND = 50
MIN_SAMPLES = 400
WHISPER_MAX_SAMPLES = 30 * SAMPLE_RATE


def load_model(name, device):
    if "whisper" in name:
        processor = WhisperFeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name).encoder
    else:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
    return processor, model.to(device).eval()


@torch.no_grad()
def get_hidden_states(processor, model, x, name, device):
    """Waveform -> [n_layers + 1, n_frames, dim], or None if Whisper input is too long."""
    if "whisper" in name:
        if len(x) > WHISPER_MAX_SAMPLES:
            return None
        inputs = processor(raw_speech=[x], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    else:
        x = np.pad(x, (0, max(0, MIN_SAMPLES - len(x))))
        inputs = processor(raw_speech=[x], sampling_rate=SAMPLE_RATE, padding=False, return_tensors="pt")
    outputs = model(**{k: t.to(device) for k, t in inputs.items()}, output_hidden_states=True)
    return torch.stack(outputs.hidden_states)[:, 0].float().cpu().numpy()


def pool(feats, start, end, pooling):
    """feats [n_layers, n_frames, dim], frames start..end inclusive -> [n_layers, dim]."""
    seg = feats[:, start:end + 1]
    if pooling == "center":
        return feats[:, (start + end) // 2]
    if pooling == "mean":
        return seg.mean(axis=1)
    ref = seg
    if pooling == "median_cosine":
        ref = seg / np.linalg.norm(seg, axis=-1, keepdims=True).clip(min=1e-8)
    dists = ((ref - ref.mean(axis=1, keepdims=True)) ** 2).sum(-1)
    return seg[np.arange(seg.shape[0]), dists.argmin(axis=1)]


def extract_run(run, args, processor, model):
    stem = f"{run}_model-{args.model.split('/')[-1]}_slice-{args.slice}_pool-{args.pooling}"
    feat_path = config.TABLES_DIR / "features" / f"{stem}.npy"
    if feat_path.exists():
        print(f"{run}: already extracted")
        return
    feat_path.parent.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(config.TABLES_DIR / f"sample_{run}.csv", index_col="row_id")
    print(f"\n== {run}: {len(sample)} words")
    positions = {row_id: i for i, row_id in enumerate(sample.index)}
    out = None

    def store(row_id, vec):
        nonlocal out
        if out is None:
            out = np.lib.format.open_memmap(feat_path, mode="w+", dtype="float32", shape=(len(sample), *vec.shape))
            out[:] = np.nan
        out[positions[row_id]] = vec

    if args.slice:
        for row_id, row in tqdm(sample.iterrows(), total=len(sample)):
            x, _ = librosa.load(config.DATA_DIR / row.audio_path, sr=SAMPLE_RATE, mono=True)
            s = int(np.clip(row.start * SAMPLE_RATE, 0, len(x)))
            e = int(np.clip(row.end * SAMPLE_RATE, 0, len(x)))
            feats = get_hidden_states(processor, model, x[s:e], args.model, args.device)
            if feats is None:
                continue
            n_frames = feats.shape[1]
            if "whisper" in args.model:  # ignore Whisper's 30 s padding
                n_frames = min(n_frames, max(1, int(np.ceil((e - s) / SAMPLE_RATE * FRAMES_PER_SECOND))))
            store(row_id, pool(feats, 0, n_frames - 1, args.pooling))
    else:
        for audio_path, group in tqdm(sample.groupby("audio_path")):
            x, _ = librosa.load(config.DATA_DIR / audio_path, sr=SAMPLE_RATE, mono=True)
            feats = get_hidden_states(processor, model, x, args.model, args.device)
            if feats is None:
                continue
            last = feats.shape[1] - 1
            for row_id, row in group.iterrows():
                s = min(int(row.start * FRAMES_PER_SECOND), last)
                e = min(int(row.end * FRAMES_PER_SECOND), last)
                store(row_id, pool(feats, s, max(s, e), args.pooling))

    out.flush()
    pd.DataFrame({"row_id": sample.index}).to_csv(feat_path.with_suffix(".rows.csv"), index=False)
    print(f"saved {feat_path}  shape={out.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-large")
    parser.add_argument("--pooling", default="mean", choices=["center", "mean", "median_euclidean", "median_cosine"])
    parser.add_argument("--slice", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--speaker", type=int, default=None)
    args = parser.parse_args()

    runs = []
    if args.seed is not None:
        runs.append(f"seed{args.seed}")
    if args.speaker is not None:
        runs.append(f"spk{args.speaker}")
    if len(runs) == 0:
        runs = [f"seed{s}" for s in config.SEEDS] + [f"spk{s}" for s in config.SPEAKERS]

    processor, model = load_model(args.model, args.device)
    for run in runs:
        extract_run(run, args, processor, model)