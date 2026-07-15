#!/usr/bin/env python3
"""Day2: Offline English Paraformer ASR (non-autoregressive, FunASR)."""
from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

PARAFORMER_DIR = (
    "/root/siton-tmp/multimodal/c2/models/modelscope/iic/"
    "speech_paraformer_asr-en-16k-vocab4199-pytorch"
)

_ASR_MODEL = None


def get_paraformer_model(device: str = "cuda:0"):
    global _ASR_MODEL
    if _ASR_MODEL is None:
        from funasr import AutoModel

        if not Path(PARAFORMER_DIR).exists():
            raise FileNotFoundError(f"Paraformer dir not found: {PARAFORMER_DIR}")
        if not Path(PARAFORMER_DIR, "model.pt").exists():
            raise FileNotFoundError(f"model.pt not found under: {PARAFORMER_DIR}")

        _ASR_MODEL = AutoModel(
            model=PARAFORMER_DIR,
            device=device,
            disable_update=True,
        )
    return _ASR_MODEL


def transcribe_array(waveform: np.ndarray, sr: int = 16000, device: str = "cuda:0") -> str:
    model = get_paraformer_model(device=device)
    res = model.generate(input=waveform, cache={})
    if not res:
        return ""
    item = res[0] if isinstance(res, list) else res
    if isinstance(item, dict):
        return (item.get("text") or "").strip()
    return str(item).strip()


def transcribe_wav(wav_path: str | Path, device: str = "cuda:0") -> str:
    y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    return transcribe_array(y, sr=16000, device=device)

def transcribe_segment(
    wav_path: str | Path,
    start_sec: float,
    end_sec: float,
    device: str = "cuda:0",
    sr: int = 16000,
) -> str:
    """Cut [start_sec, end_sec) from wav and run Paraformer."""
    y, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    i0 = max(0, int(start_sec * sr))
    i1 = min(len(y), int(end_sec * sr))
    if i1 <= i0:
        return ""
    clip = y[i0:i1]
    if len(clip) < int(0.1 * sr):  # 太短跳过
        return ""
    return transcribe_array(clip, sr=sr, device=device)

if __name__ == "__main__":
    import json
    import time

    ROOT = Path(__file__).resolve().parents[1]
    VAD_IN = ROOT / "outputs" / "meeting" / "vad_segments.json"
    OUT = ROOT / "outputs" / "meeting" / "asr_segments.json"

    meetings = json.load(open(VAD_IN, encoding="utf-8"))
    results = []

    for mtg in meetings:
        audio = mtg["audio"]
        segs = mtg["segments"]
        print(f"ASR: {mtg['id']} | {audio} | segments={len(segs)}")

        seg_out = []
        for i, seg in enumerate(segs, start=1):
            t0 = time.time()
            text = transcribe_segment(
                audio,
                seg["start_sec"],
                seg["end_sec"],
            )
            seg_out.append({
                **seg,
                "text_en": text,
                "is_final": True,
                "latency_sec": round(time.time() - t0, 3),
            })
            if i % 20 == 0 or i == len(segs):
                print(f"  progress: {i}/{len(segs)}")

        results.append({
            "id": mtg["id"],
            "audio": audio,
            "asr_model": "paraformer_asr-en-16k-vocab4199-pytorch",
            "asr_backend": "funasr",
            "asr_type": "non-autoregressive",
            "note": "Paraformer: CIF predictor + parallel decoder (arXiv:2206.08317)",
            "segments": seg_out,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved: {OUT}")