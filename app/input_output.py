from __future__ import annotations

import csv
from pathlib import Path


def read_input(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        n = int(first[0].strip())
        for _ in range(n):
            width, height, name = next(reader)[:3]
            rows.append({"width": int(width), "height": int(height), "name": name.strip()})
    if len(rows) != n:
        raise ValueError("input.csv の画像数と行数が一致しません")
    return rows


def write_output(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([len(rows), ""])
        for r in rows:
            w.writerow([r["width"], r["height"], r["output"], r["level"], r["p"], r["w"], f'{r["ratio"]:.2f}'])
