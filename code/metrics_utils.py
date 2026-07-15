from __future__ import annotations
import os
from pathlib import Path

import torch

def model_size_mb(model_path: str | Path) -> float:
    p = Path(model_path)
    if p.is_file():
        return p.stat().st_size / (1024 ** 2)
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 ** 2)

def reset_peak_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def peak_vram_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)
