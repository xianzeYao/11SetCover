from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SetRecord:
    index: int
    cost: float
    items: tuple[int, ...]


@dataclass(frozen=True)
class SetCoverInstance:
    dataset_id: str
    class_id: str
    sample_id: str
    path: str
    n_items: int
    n_sets: int
    density: float
    pattern: str
    seed: int
    sets: tuple[SetRecord, ...]

    @property
    def instance_id(self) -> str:
        return f"{self.class_id}/{self.sample_id}"


@dataclass(frozen=True)
class SolveResult:
    objective: float
    runtime_sec: float
    is_feasible: bool
    selected_sets: tuple[int, ...]
    convergence_curve: tuple[float, ...]
    meta: dict[str, Any]
