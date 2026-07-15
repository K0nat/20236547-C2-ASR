#!/usr/bin/env python3
import json
from pathlib import Path

from c2_asr import run_asr

ROOT = Path(__file__).resolve().parents[1]
C1_ROOT = Path("/root/siton-tmp/multimodal/c1/advanced_outputs")
OUT_ROOT = ROOT / "outputs" / "robustness"
MODEL = ROOT / "models" / "whisper-large-v3-turbo"

VERSIONS = [
    "clean_16k",
    "sr_8k", "sr_24k",
    "speed_0_9", "speed_1_1",
    "volume_0_5", "volume_1_5",
    "noise_snr_20", "noise_snr_10", "noise_snr_5",
    "vad_speech_only",
]


def load_merged_dataset(version: str) -> list:
    tess_path = C1_ROOT / "TESS" / version / "dataset.json"
    crema_path = C1_ROOT / "CREMA-D" / version / "dataset.json"
    tess = json.load(open(tess_path, encoding="utf-8"))
    crema = json.load(open(crema_path, encoding="utf-8"))
    merged = tess + crema
    print(f"[{version}] TESS={len(tess)} CREMA={len(crema)} total={len(merged)}")
    return merged


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for version in VERSIONS:
        print(f"\n===== VERSION: {version} =====")
        outdir = OUT_ROOT / version
        outdir.mkdir(parents=True, exist_ok=True)

        # 可选：已有结果就跳过（断点续跑）
        if (outdir / "c2_summary.json").exists():
            print(f"skip (already done): {outdir / 'c2_summary.json'}")
            continue

        merged = load_merged_dataset(version)

        # 必须写文件，run_asr 读路径
        dataset_path = outdir / "dataset_merged.json"
        json.dump(merged, open(dataset_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        summary = run_asr(str(dataset_path), str(MODEL), str(outdir))
        print(f"done {version}: avg_wer={summary.get('avg_wer')} total_sec={summary.get('total_sec')}")

    print("\nALL ROBUSTNESS VERSIONS DONE.")


if __name__ == "__main__":
    main()