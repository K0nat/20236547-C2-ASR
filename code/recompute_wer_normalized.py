# code/recompute_wer_normalized.py
import json
from pathlib import Path

import jiwer

from text_normalize import normalize_text

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "common_data" / "dataset.json"   # 若不对，改成 ROOT / "common_data" / "dataset.json"
OUTPUTS = ROOT / "outputs"

MODELS = ["small", "turbo", "large_v3", "paraformer_en"]


def load_refs(dataset_path: Path) -> dict[str, str]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    refs = {}
    for item in data:
        rid = item["id"]
        ref = item.get("reference_text") or item.get("text") or ""
        refs[rid] = ref
    return refs


def compute_wer(pairs: list[tuple[str, str]]) -> tuple[float | None, int]:
    """pairs: [(ref, hyp), ...]，空 ref 跳过"""
    refs, hyps = [], []
    for ref, hyp in pairs:
        if not ref or not str(ref).strip():
            continue
        refs.append(ref)
        hyps.append(hyp or "")
    if not refs:
        return None, 0
    return float(jiwer.wer(refs, hyps)), len(refs)


def eval_model(model_name: str, refs_by_id: dict[str, str]) -> dict:
    pred_path = OUTPUTS / model_name / "asr_predictions.json"
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    pred_map = {p["id"]: p.get("prediction_text", "") for p in preds}

    raw_pairs = []
    norm_pairs = []
    digit_mismatch = 0

    for rid, ref in refs_by_id.items():
        hyp = pred_map.get(rid, "")
        raw_pairs.append((ref, hyp))
        norm_pairs.append((normalize_text(ref), normalize_text(hyp)))

        # 统计：归一化前不同、归一化后相同（典型 eleven/11）
        r0, h0 = ref.lower().strip(), (hyp or "").lower().strip()
        r1, h1 = normalize_text(ref), normalize_text(hyp)
        if r0 != h0 and r1 == h1:
            digit_mismatch += 1

    raw_wer, scored = compute_wer(raw_pairs)
    norm_wer, _ = compute_wer(norm_pairs)

    return {
        "model": model_name,
        "scored": scored,
        "raw_wer": raw_wer,
        "norm_wer": norm_wer,
        "norm_fixed_count": digit_mismatch,
    }


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"dataset not found: {DATASET}")

    refs_by_id = load_refs(DATASET)
    rows = [eval_model(m, refs_by_id) for m in MODELS]

    out_dir = OUTPUTS / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "wer_raw_vs_normalized.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown 表
    md_lines = [
        "| Model | Scored | Raw WER | Normalized WER | Fixed by norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['scored']} | {r['raw_wer']:.6f} | {r['norm_wer']:.6f} | {r['norm_fixed_count']} |"
        )
    md_path = out_dir / "wer_raw_vs_normalized.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("\n".join(md_lines))
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()