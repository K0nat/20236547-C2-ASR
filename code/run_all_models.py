from pathlib import Path
import json, time

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "common_data" / "dataset.json"

MODELS = {
    "small": {
        "path": ROOT / "models" / "whisper-small",
        "backend": "transformers",
    },
    "turbo": {
        "path": ROOT / "models" / "whisper-large-v3-turbo",
        "backend": "transformers",
    },
    "large_v3": {
        "path": ROOT / "models" / "whisper-large-v3",
        "backend": "transformers",
    },
    "paraformer_en": {
        "path": ROOT / "models/modelscope/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch",
        "backend": "funasr",
    },
}

def main():
    from c2_asr import run_asr as run_whisper
    from paraformer_batch import run_asr as run_paraformer

    timing = []
    for name, cfg in MODELS.items():
        outdir = ROOT / "outputs" / name
        print(f"\n===== MODEL: {name} =====")
        t0 = time.time()
        if cfg["backend"] == "funasr":
            summary = run_paraformer(str(DATASET), str(outdir))
        else:
            summary = run_whisper(str(DATASET), str(cfg["path"]), str(outdir))
        wall = time.time() - t0
        summary["wall_sec"] = round(wall, 2)
        json.dump(summary, open(outdir / "c2_summary.json", "w", encoding="utf-8"), indent=2)
        timing.append({"model": name, **summary})
        print(f"done {name}, wall={wall:.1f}s, peak_vram={summary.get('peak_vram_mb')}MB")

    print("ALL MODELS DONE ONCE. EXIT.")

    compare_dir = ROOT / "outputs" / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": str(DATASET), "models": timing},
                open(compare_dir / "timing_report.json", "w", encoding="utf-8"), indent=2)

if __name__ == "__main__":
    main()