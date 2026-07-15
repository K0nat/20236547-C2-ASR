#!/usr/bin/env python3
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["small", "turbo", "large_v3", "paraformer_en"]
COMPARE = ROOT / "outputs" / "compare"
LATEST = ROOT / "outputs" / "latest"

def load_summary(name):
    p = ROOT / "outputs" / name / "c2_summary.json"
    if not p.exists():
        return None
    s = json.load(open(p, encoding="utf-8"))
    s["name"] = name
    return s

def md_table(rows):
    hdr = "| Model | Size(MB) | Peak VRAM(MB) | Total(s) | Avg latency(s) | Avg WER | Avg CER | Scored |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r.get('model_size_mb','-')} | {r.get('peak_vram_mb','-')} | "
            f"{r.get('total_sec', r.get('wall_sec','-'))} | {r.get('avg_latency_sec','-')} | "
            f"{r.get('avg_wer','-')} | {r.get('avg_cer','-')} | {r.get('num_wer_scored','-')} |"
        )
    return "\n".join(lines)

def pick_best(rows):
    # 有 wer 的里选最低；并列时选 total_sec 更短
    scored = [r for r in rows if r.get("avg_wer") is not None]
    if not scored:
        return "large_v3"
    return min(scored, key=lambda r: (r["avg_wer"], r.get("total_sec", 1e9)))["name"]

def main():
    rows = [load_summary(n) for n in NAMES]
    rows = [r for r in rows if r]
    if not rows:
        raise SystemExit("no summaries found")

    best = pick_best(rows)
    out = {"best_model": best, "models": rows}
    COMPARE.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(COMPARE / "model_comparison.json", "w", encoding="utf-8"), indent=2)

    md = [
        "# ASR Model Comparison",
        "",
        md_table(rows),
        "",
        f"**Best (lowest WER, then speed): `{best}`**",
        "",
        "## Trade-off notes",
        "- **small**: smallest / fastest baseline",
        "- **turbo**: balanced speed vs accuracy",
        "- **large_v3**: highest Whisper quality, more VRAM/time",
        "- **paraformer_en**: non-autoregressive (FunASR), good for segment/long-audio pipeline",
        "",
    ]
    open(COMPARE / "model_comparison.md", "w", encoding="utf-8").write("\n".join(md))

    # publish best to C3（若你固定用 large_v3，可改成 best = "large_v3"）
    src = ROOT / "outputs" / best / "asr_predictions.json"
    LATEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, LATEST / "asr_predictions.json")
    json.dump({"best_model": best, "source": str(src)}, open(LATEST / "meta.json", "w"), indent=2)

    print("wrote", COMPARE / "model_comparison.md")
    print("best_model =", best)
    print(md_table(rows))

if __name__ == "__main__":
    main()
