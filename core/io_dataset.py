from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from core.types import SetCoverInstance, SetRecord


META_PATTERN = re.compile(
    r"^# class=(?P<class_id>\S+) sample=(?P<sample_id>\S+) n=(?P<n>\d+) m=(?P<m>\d+) "
    r"density=(?P<density>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?) pattern=(?P<pattern>\S+) seed=(?P<seed>-?\d+)$"
)


def parse_meta_line(line: str) -> dict[str, str]:
    match = META_PATTERN.match(line.strip())
    if match is None:
        raise ValueError(
            "首行元信息格式错误，要求: "
            "# class=<id> sample=<id> n=<items> m=<sets> density=<d> pattern=<p> seed=<s>"
        )
    return match.groupdict()


def read_instance(path: str | Path, dataset_id: str | None = None) -> SetCoverInstance:
    p = Path(path)
    lines = [line.rstrip("\n") for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"实例文件内容不足: {p}")

    meta = parse_meta_line(lines[0])

    header_parts = lines[1].split()
    if len(header_parts) != 2:
        raise ValueError(f"第二行头部应为 '<n_items> <n_sets>': {p}")

    n_items = int(header_parts[0])
    n_sets = int(header_parts[1])
    if n_items != int(meta["n"]) or n_sets != int(meta["m"]):
        raise ValueError(f"首行与第二行规模不一致: {p}")

    set_lines = lines[2:]
    if len(set_lines) != n_sets:
        raise ValueError(f"集合行数不匹配，声明={n_sets}，实际={len(set_lines)}: {p}")

    records: list[SetRecord] = []
    for idx, raw in enumerate(set_lines):
        parts = raw.split()
        if not parts:
            raise ValueError(f"集合行为空，index={idx}: {p}")
        cost = float(parts[0])
        items = tuple(int(x) for x in parts[1:])
        for item in items:
            if item < 0 or item >= n_items:
                raise ValueError(f"元素越界，set={idx}, item={item}, n_items={n_items}: {p}")
        records.append(SetRecord(index=idx, cost=cost, items=items))

    covered = set()
    for rec in records:
        covered.update(rec.items)
    if len(covered) < n_items:
        raise ValueError(f"实例不可行，未覆盖元素数={n_items - len(covered)}: {p}")

    nonzeros = sum(len(rec.items) for rec in records)
    density = nonzeros / float(n_items * n_sets) if n_items > 0 and n_sets > 0 else 0.0

    ds_id = dataset_id if dataset_id is not None else p.parent.parent.name

    return SetCoverInstance(
        dataset_id=ds_id,
        class_id=meta["class_id"],
        sample_id=meta["sample_id"],
        path=str(p),
        n_items=n_items,
        n_sets=n_sets,
        density=density,
        pattern=meta["pattern"],
        seed=int(meta["seed"]),
        sets=tuple(records),
    )


def iter_instances(
    dataset_root: str | Path,
    class_filter: Iterable[str] | None = None,
    file_prefix: str = "sc_",
) -> Iterable[SetCoverInstance]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    allowed = set(class_filter) if class_filter else None
    for class_dir in sorted(x for x in root.iterdir() if x.is_dir()):
        if allowed is not None and class_dir.name not in allowed:
            continue
        for path in sorted(x for x in class_dir.iterdir() if x.is_file() and x.name.startswith(file_prefix)):
            yield read_instance(path, dataset_id=root.name)


def read_all_instances(
    dataset_root: str | Path,
    class_filter: Iterable[str] | None = None,
    file_prefix: str = "sc_",
) -> list[SetCoverInstance]:
    return list(iter_instances(dataset_root=dataset_root, class_filter=class_filter, file_prefix=file_prefix))


def write_instance_file(
    path: str | Path,
    class_id: str,
    sample_id: str,
    n_items: int,
    n_sets: int,
    density: float,
    pattern: str,
    seed: int,
    costs: list[int],
    set_items: list[list[int]],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    first = (
        f"# class={class_id} sample={sample_id} n={n_items} m={n_sets} "
        f"density={density:.6f} pattern={pattern} seed={seed}"
    )
    lines = [first, f"{n_items} {n_sets}"]
    for cost, items in zip(costs, set_items):
        item_text = " ".join(str(i) for i in items)
        cost_text = str(int(cost))
        if item_text:
            lines.append(f"{cost_text} {item_text}")
        else:
            lines.append(cost_text)

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
