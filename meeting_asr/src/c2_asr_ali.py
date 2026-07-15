from __future__ import annotations

import argparse

from tqdm import tqdm

from .utils.api_clients import AliOpenAIClient, MissingConfiguration
from .utils.audio_utils import audio_to_data_url
from .utils.io_utils import load_config, read_json, write_json
from .utils.logger import setup_logging


def run(config_path: str = "config/config.yaml", limit: int | None = None) -> list[dict]:
    cfg = load_config(config_path)
    logger = setup_logging()
    segments = read_json(cfg["paths"]["segments_json"])
    if limit:
        segments = segments[:limit]
    model = cfg["models"]["asr_model"]
    max_mb = cfg["api"]["max_data_url_mb"]
    results = []

    try:
        client = AliOpenAIClient(cfg)
        ready, preflight_error = client.preflight()
        if not ready:
            logger.error(preflight_error)
            client = None
    except MissingConfiguration as exc:
        logger.error(str(exc))
        client = None
        preflight_error = str(exc)

    for seg in tqdm(segments, desc="C2 ASR"):
        item = {
            "id": seg["id"],
            "audio_path": seg["audio_path"],
            "start": seg["start"],
            "end": seg["end"],
            "asr_text": "",
            "asr_model": model,
            "latency": 0.0,
            "success": False,
            "fallback": False,
            "error": None,
        }
        if client is None:
            item["error"] = preflight_error or "API 未就绪，跳过真实 ASR 调用"
            results.append(item)
            continue
        try:
            data_url = audio_to_data_url(seg["audio_path"], max_mb=max_mb)
            text, latency, fallback = client.asr(
                model=model,
                audio_path=seg["audio_path"],
                data_url=data_url,
                prompt="Please transcribe this English meeting speech segment. Output English text only.",
            )
            item.update({"asr_text": text, "latency": latency, "success": bool(text), "fallback": fallback})
            if not text:
                item["error"] = "ASR 返回为空"
        except Exception as exc:
            logger.error("ASR failed for %s: %s", seg["id"], exc)
            item["error"] = str(exc)
        results.append(item)

    write_json(cfg["paths"]["asr_json"], results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.limit)


if __name__ == "__main__":
    main()
