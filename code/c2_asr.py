#!/usr/bin/env python3
import argparse
import json
import re
import time
import os
from pathlib import Path

import jiwer
import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from metrics_utils import model_size_mb, peak_vram_mb, reset_peak_vram

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

BATCH_SIZE = 64

def norm(t):
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())

class WhisperASR:
    def __init__(self, model_path, device="cuda"):
        self.processor = WhisperProcessor.from_pretrained(model_path)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        self.model.to(device).eval()
        self.device = device
        self.forced = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

    def transcribe_one(self, wav_path):
        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features
        if self.device == "cuda":
            feat = feat.to(device=self.device, dtype=torch.float16)
        else:
            feat = feat.to(self.device)
        with torch.inference_mode():
            ids = self.model.generate(
                feat, forced_decoder_ids=self.forced, max_new_tokens=128
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def transcribe_batch(self, wav_paths):
        audios = []
        for p in wav_paths:
            audio, _ = librosa.load(p, sr=16000, mono=True)
            audios.append(audio)
        inputs = self.processor(
            audios, sampling_rate=16000, return_tensors="pt", padding=True
        )
        feat = inputs.input_features
        if self.device == "cuda":
            feat = feat.to(device=self.device, dtype=torch.float16)
        else:
            feat = feat.to(self.device)
        gen_kwargs = {
            "forced_decoder_ids": self.forced,
            "max_new_tokens": 128,
        }
        with torch.inference_mode():
            ids = self.model.generate(feat, **gen_kwargs)
        return [
            t.strip()
            for t in self.processor.batch_decode(ids, skip_special_tokens=True)
        ]

def run_asr(dataset, model, outdir, device="cuda", batch_size=BATCH_SIZE):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(dataset, encoding="utf-8"))

    reset_peak_vram()
    t0_all = time.time()

    asr = WhisperASR(model, device=device)
    results = []

    for start in range(0, len(data), batch_size):
        batch = data[start : start + batch_size]
        paths = [x["processed_audio"] for x in batch]
        t0 = time.time()
        hyps, statuses, errors = [], [], []

        try:
            hyps = asr.transcribe_batch(paths)
            statuses = ["ok"] * len(batch)
            errors = [None] * len(batch)
        except Exception:
            hyps = []
            statuses = []
            errors = []
            for item in batch:
                try:
                    hyps.append(asr.transcribe_one(item["processed_audio"]))
                    statuses.append("ok")
                    errors.append(None)
                except Exception as e2:
                    hyps.append("")
                    statuses.append("error")
                    errors.append(str(e2))

        batch_time = time.time() - t0
        per_item_lat = batch_time / len(batch)

        for j, item in enumerate(batch):
            ref = item.get("reference_text") or ""
            hyp = hyps[j] if j < len(hyps) else ""
            status = statuses[j] if j < len(statuses) else "error"
            err = errors[j] if j < len(errors) else "batch decode mismatch"
            wer = cer = None
            if ref and status == "ok":
                wer = jiwer.wer(norm(ref), norm(hyp))
                cer = jiwer.cer(norm(ref), norm(hyp))
            results.append({
                "id": item["id"],
                "audio": item["processed_audio"],
                "reference_text": ref,
                "prediction_text": hyp,
                "wer": wer,
                "cer": cer,
                "latency_sec": round(per_item_lat, 3),
                "model": Path(model).name,
                "backend": "transformers",
                "batch_size": batch_size,
                "status": status,
                "error": err,
            })

        done = min(start + batch_size, len(data))
        vram = peak_vram_mb()
        vram_s = f", peak_vram={vram:.0f}MB" if vram is not None else ""
        print(f"  [{done}/{len(data)}] batch ok, batch_time={batch_time:.2f}s{vram_s}")

    total_sec = time.time() - t0_all
    ok = [r for r in results if r["status"] == "ok" and r["wer"] is not None]
    lats = [r["latency_sec"] for r in results if r["status"] == "ok"]

    summary = {
        "count": len(results),
        "backend": "transformers",
        "model": Path(model).name,
        "model_path": str(model),
        "batch_size": batch_size,
        "avg_wer": sum(r["wer"] for r in ok) / len(ok) if ok else None,
        "avg_cer": sum(r["cer"] for r in ok) / len(ok) if ok else None,
        "avg_latency_sec": round(sum(lats) / len(lats), 4) if lats else None,
        "model_size_mb": round(model_size_mb(model), 2),
        "total_sec": round(total_sec, 2),
        "peak_vram_mb": round(peak_vram_mb() or 0, 2),
        "num_wer_scored": len(ok),
    }

    json.dump(
        results,
        open(outdir / "asr_predictions.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    json.dump(
        summary,
        open(outdir / "c2_summary.json", "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="../../common_data/dataset.json")
    p.add_argument("--model", required=True)
    p.add_argument("--outdir", default="../outputs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()
    s = run_asr(args.dataset, args.model, args.outdir, args.device, args.batch_size)
    print("summary:", s)

if __name__ == "__main__":
    main()