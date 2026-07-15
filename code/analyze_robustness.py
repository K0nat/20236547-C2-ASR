#!/usr/bin/env python3
"""Read 11 robustness summaries and write comparison JSON + markdown."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROB_DIR = ROOT / "outputs" / "robustness"
OUT_DIR = ROB_DIR

BASELINE = "clean_16k"

VERSIONS = [
    "clean_16k",
    "sr_8k", "sr_24k",
    "speed_0_9", "speed_1_1",
    "volume_0_5", "volume_1_5",
    "noise_snr_20", "noise_snr_10", "noise_snr_5",
    "vad_speech_only",
]

GROUPS = {
    "sample_rate": ["clean_16k", "sr_8k", "sr_24k"],
    "speed": ["clean_16k", "speed_0_9", "speed_1_1"],
    "volume": ["clean_16k", "volume_0_5", "volume_1_5"],
    "noise": ["clean_16k", "noise_snr_20", "noise_snr_10", "noise_snr_5"],
    "vad": ["clean_16k", "vad_speech_only"],
}


def load_results():
    rows = []
    for v in VERSIONS:
        p = ROB_DIR / v / "c2_summary.json"
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
        s = json.load(open(p, encoding="utf-8"))
        rows.append({
            "version": v,
            "count": s.get("count"),
            "avg_wer": s.get("avg_wer"),
            "avg_cer": s.get("avg_cer"),
            "total_sec": s.get("total_sec"),
            "peak_vram_mb": s.get("peak_vram_mb"),
            "avg_latency_sec": s.get("avg_latency_sec"),
        })
    return rows


def main():
    rows = load_results()
    by_ver = {r["version"]: r for r in rows}
    base_wer = by_ver[BASELINE]["avg_wer"]
    base_cer = by_ver[BASELINE]["avg_cer"]

    for r in rows:
        r["wer_delta_vs_baseline"] = round(r["avg_wer"] - base_wer, 6)
        r["cer_delta_vs_baseline"] = round(r["avg_cer"] - base_cer, 6)
        if base_wer > 0:
            r["wer_relative_increase_pct"] = round(
                100.0 * (r["avg_wer"] - base_wer) / base_wer, 2
            )
        else:
            r["wer_relative_increase_pct"] = None

    # 1) 总表
    summary = {
        "task": "c2_robustness_eval",
        "model": "whisper-large-v3-turbo",
        "baseline": BASELINE,
        "baseline_wer": base_wer,
        "baseline_cer": base_cer,
        "num_versions": len(rows),
        "results": rows,
    }
    p1 = OUT_DIR / "robustness_summary.json"
    json.dump(summary, open(p1, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 2) 分组分析
    group_analysis = {}
    for gname, vers in GROUPS.items():
        items = [by_ver[v] for v in vers]
        worst = max(items, key=lambda x: x["avg_wer"])
        best = min(items, key=lambda x: x["avg_wer"])
        group_analysis[gname] = {
            "versions": vers,
            "wer_by_version": {v: by_ver[v]["avg_wer"] for v in vers},
            "best_version": best["version"],
            "worst_version": worst["version"],
            "max_wer_delta_vs_baseline": round(worst["avg_wer"] - base_wer, 6),
        }
    analysis = {
        "baseline": BASELINE,
        "groups": group_analysis,
    }
    p2 = OUT_DIR / "robustness_analysis.json"
    json.dump(analysis, open(p2, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 3) meta
    meta = {
        "model": "whisper-large-v3-turbo",
        "batch_eval_root": str(ROB_DIR),
        "c1_data_root": "/root/siton-tmp/multimodal/c1/advanced_outputs",
        "versions": VERSIONS,
    }
    p3 = OUT_DIR / "robustness_meta.json"
    json.dump(meta, open(p3, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 4) Markdown 报告表
    md_lines = [
        "# C2 Robustness (Whisper turbo)",
        "",
        f"Baseline: `{BASELINE}` WER={base_wer:.6f} CER={base_cer:.6f}",
        "",
        "| Version | WER | CER | ΔWER vs baseline | Rel. increase % | Total(s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['version']} | {r['avg_wer']:.6f} | {r['avg_cer']:.6f} | "
            f"{r['wer_delta_vs_baseline']:+.6f} | {r['wer_relative_increase_pct']} | {r['total_sec']} |"
        )
    md_lines.append("")
    md_lines.append("## By group")
    for gname, info in group_analysis.items():
        md_lines.append(f"- **{gname}**: worst=`{info['worst_version']}` "
                        f"(ΔWER={info['max_wer_delta_vs_baseline']:+.6f})")
    p4 = OUT_DIR / "robustness_report.md"
    p4.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"saved: {p1}")
    print(f"saved: {p2}")
    print(f"saved: {p3}")
    print(f"saved: {p4}")
    print("\n" + p4.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()