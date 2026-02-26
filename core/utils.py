from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_csv_list(text: str, cast: type = str) -> list[Any]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if cast is str:
        return values
    return [cast(v) for v in values]
