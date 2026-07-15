#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import jiwer
import torch

from metrics_utils import model_size_mb, peak_vram_mb, reset_peak_vram

PARAFORMER_DIR = Path(
    "/root/siton-tmp/multimodal/c2/models/modelscope/iic/"
    "speech_paraformer_asr-en-16k-vocab4199-pytorch"
)

_MODEL = None

def norm(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())

def get_model(device="cuda:0"):
    global _MODEL
    if _MODEL is None:
        from funasr import AutoModel
        reset_peak_vram()
        _MODEL = AutoModel(
            model=str(PARAFORMER_DIR),
            device=device,
            disable_update=True,
        )
    return _MODEL

def transcribe_one(model, wav_path: str) -> str:
    res = model.generate(input=wav_path, batch_size=1)
    if not res:
        return ""
    item = res[0]
    if isinstance(item, dict):
        return (item.get("text") or item.get("text_en") or "").strip()
    return str(item).strip()

def run_asr(dataset_path: str, outdir: str, device="cuda:0") -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(dataset_path, encoding="utf-8"))
    model = get_model(device)

    rows = []
    t0_all = time.time()
    reset_peak_vram()

    for i, row in enumerate(data, 1):
        wav = row.get("processed_audio") or row.get("audio")
        ref = (row.get("reference_text") or "").strip()
        t1 = time.time()
        try:
            pred = transcribe_one(model, wav)
            status, err = "ok", None
        except Exception as e:
            pred, status, err = "", "error", str(e)

        latency = time.time() - t1
        nref, npred = norm(ref), norm(pred)
        wer = cer = None
        if nref:
            wer = float(jiwer.wer(nref, npred))
            cer = float(jiwer.cer(nref, npred))

        rows.append({
            "id": row.get("id"),
            "audio": wav,
            "reference_text": ref,
            "prediction_text": pred,
            "wer": wer,
            "cer": cer,
            "latency_sec": round(latency, 4),
            "model": "paraformer_asr-en-16k-vocab4199-pytorch",
            "backend": "funasr",
            "status": status,
            "error": err,
        })
        if i % 100 == 0:
            print(f"  [{i}/{len(data)}] peak_vram={peak_vram_mb():.0f}MB")

    total_sec = time.time() - t0_all
    wers = [r["wer"] for r in rows if r["wer"] is not None]
    cers = [r["cer"] for r in rows if r["cer"] is not None]
    lats = [r["latency_sec"] for r in rows if r["status"] == "ok"]

    summary = {
        "count": len(rows),
        "backend": "funasr",
        "model": "paraformer_en",
        "model_path": str(PARAFORMER_DIR),
        "model_size_mb": round(model_size_mb(PARAFORMER_DIR), 2),
        "total_sec": round(total_sec, 2),
        "avg_latency_sec": round(sum(lats) / len(lats), 4) if lats else None,
        "avg_wer": round(sum(wers) / len(wers), 6) if wers else None,
        "avg_cer": round(sum(cers) / len(cers), 6) if cers else None,
        "peak_vram_mb": round(peak_vram_mb() or 0, 2),
        "num_wer_scored": len(wers),
    }

    json.dump(rows, open(outdir / "asr_predictions.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(summary, open(outdir / "c2_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("summary:", summary)
    return summary
