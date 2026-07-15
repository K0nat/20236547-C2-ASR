#!/usr/bin/env python3
"""Day1: FSMN-VAD via FunASR -> speech segments in seconds."""
from __future__ import annotations

from pathlib import Path
from typing import Any

# 本地模型目录（不要用 damo/... 在线 ID，否则会连 modelscope.cn）
MODEL_DIR = "/root/siton-tmp/multimodal/c2/models/modelscope/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 全局只加载一次模型
_VAD_MODEL = None


def get_vad_model(device: str = "cuda:0"):
    global _VAD_MODEL
    if _VAD_MODEL is None:
        from funasr import AutoModel

        if not Path(MODEL_DIR).exists():
            raise FileNotFoundError(f"VAD model dir not found: {MODEL_DIR}")

        _VAD_MODEL = AutoModel(
            model=MODEL_DIR,
            device=device,
            disable_update=True,
        )
    return _VAD_MODEL


def _parse_funasr_vad_result(res: Any) -> list[tuple[float, float]]:
    """Parse FunASR VAD output -> [(start_sec, end_sec), ...]."""
    segments: list[tuple[float, float]] = []

    def add_ms_pair(start_ms, end_ms):
        start_sec = float(start_ms) / 1000.0
        end_sec = float(end_ms) / 1000.0
        if end_sec > start_sec:
            segments.append((start_sec, end_sec))

    if isinstance(res, list):
        for item in res:
            if not isinstance(item, dict):
                continue
            val = item.get("value") or item.get("text") or item
            if isinstance(val, list):
                for seg in val:
                    if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                        add_ms_pair(seg[0], seg[1])
    elif isinstance(res, dict):
        val = res.get("value") or res.get("text")
        if isinstance(val, list):
            for seg in val:
                if isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    add_ms_pair(seg[0], seg[1])

    return segments


def run_vad(
    audio_path: str | Path,
    device: str = "cuda:0",
    min_duration_sec: float = 0.3,
) -> list[dict]:
    """
    Returns:
        [{"seg_id", "start_sec", "end_sec", "duration_sec"}, ...]
    """
    audio_path = str(Path(audio_path).resolve())
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    model = get_vad_model(device=device)
    res = model.generate(input=audio_path)

    raw = _parse_funasr_vad_result(res)
    out = []
    for i, (start, end) in enumerate(raw, start=1):
        dur = end - start
        if dur < min_duration_sec:
            continue
        out.append(
            {
                "seg_id": f"s{i:04d}",
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(dur, 3),
            }
        )
    return out

if __name__ == "__main__":
    import json

    ROOT = Path(__file__).resolve().parents[1]
    DATASET = ROOT / "testdata" / "meeting_dataset.json"
    OUT = ROOT / "outputs" / "meeting" / "vad_segments.json"

    data = json.load(open(DATASET, encoding="utf-8"))
    results = []

    for item in data:
        audio = item["audio"]
        print(f"VAD: {item['id']} -> {audio}")
        segments = run_vad(audio)
        results.append({
            "id": item["id"],
            "audio": audio,
            "vad_backend": "fsmn-vad",
            "num_segments": len(segments),
            "segments": segments,
        })
        print(f"  segments: {len(segments)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"saved: {OUT}")