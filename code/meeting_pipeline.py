#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vad import run_vad
from asr_segment import run_segment_asr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vad", "asr"], required=True)
    ap.add_argument("--dataset", help="for --stage vad")
    ap.add_argument("--vad", help="vad_segments.json for --stage asr")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.stage == "vad":
        if not args.dataset:
            raise SystemExit("--dataset required for vad stage")
        rows = json.load(open(args.dataset, encoding="utf-8"))
        results = []
        for item in rows:
            segs = run_vad(item["audio"], device=args.device)
            results.append(
                {
                    "id": item["id"],
                    "audio": item["audio"],
                    "vad_backend": "fsmn-vad",
                    "num_segments": len(segs),
                    "segments": segs,
                }
            )
            print(f"[{item['id']}] vad segments={len(segs)}")

    elif args.stage == "asr":
        if not args.vad:
            raise SystemExit("--vad required for asr stage")
        rows = json.load(open(args.vad, encoding="utf-8"))
        results = []
        for item in rows:
            one = run_segment_asr(item, device=args.device)
            results.append(one)
            print(f"[{item['id']}] asr segments={len(one['segments'])}")
            for s in one["segments"]:
                preview = (s.get("text_en") or "")[:80]
                print(f"  {s['seg_id']}: {preview}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved:", out.resolve())


if __name__ == "__main__":
    main()