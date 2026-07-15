#!/usr/bin/env python3
"""Cut VAD segments and run Paraformer on each."""
from __future__ import annotations

import time

import librosa
import numpy as np

from paraformer_asr import transcribe_array


def load_full_audio(audio_path: str, sr: int = 16000):
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def cut_segment(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    s = max(0, int(start_sec * sr))
    e = min(len(y), int(end_sec * sr))
    if e <= s:
        return np.zeros(0, dtype=np.float32)
    return y[s:e]


def run_segment_asr(vad_item: dict, device: str = "cuda:0") -> dict:
    audio_path = vad_item["audio"]
    y, sr = load_full_audio(audio_path)

    out_segments = []
    for seg in vad_item["segments"]:
        t0 = time.time()
        clip = cut_segment(y, sr, seg["start_sec"], seg["end_sec"])

        if len(clip) < int(sr * 0.1):
            text_en = ""
        else:
            text_en = transcribe_array(clip, sr=sr, device=device)

        out_segments.append(
            {
                **seg,
                "text_en": text_en,
                "is_final": True,
                "latency_sec": round(time.time() - t0, 3),
            }
        )

    return {
        "id": vad_item["id"],
        "audio": audio_path,
        "asr_model": "paraformer_asr-en-16k-vocab4199-pytorch",
        "asr_backend": "funasr",
        "asr_type": "non-autoregressive",
        "note": "Paraformer: CIF predictor + parallel decoder (arXiv:2206.08317)",
        "segments": out_segments,
    }