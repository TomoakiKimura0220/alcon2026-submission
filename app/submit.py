from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models import load_model, logits_of
from data import letterbox, tensor_image
from input_output import read_input, write_output

LUT = np.asarray([[0, 0, 0], [128, 128, 128], [255, 255, 255]], dtype=np.uint8)


def ratio_counts(mask: np.ndarray) -> tuple[int, int, float]:
    counts = np.bincount(mask.reshape(-1), minlength=3)
    p, w = int(counts[1]), int(counts[2])
    return p, w, (100.0 * w / (p + w)) if p + w else 0.0


def predict(model, image: Image.Image, size: int, device: str) -> np.ndarray:
    boxed, _, (left, top, nw, nh) = letterbox(image, None, size)
    x = tensor_image(boxed)[None].to(device)
    with torch.inference_mode():
        logits = logits_of(model, x)
        logits = logits[..., top:top + nh, left:left + nw]
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return np.asarray(Image.fromarray(pred).resize(image.size, Image.Resampling.NEAREST))


def level(ratio: float) -> int:
    # 暫定共通閾値: 3.5%, 7.5%, 25%
    return int(np.searchsorted(np.asarray([3.5, 7.5, 25.0]), ratio, side="right"))


def process_one(index, row, root, model, mode, size, device):
    started = time.perf_counter()
    rel = Path(row["name"].replace("\\", os.sep))
    image_path = root / rel
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    pred = predict(model, image, size, device)
    p, w, ratio = ratio_counts(pred)
    output_rel = rel.with_name(rel.stem + "-output.JPG")
    output_path = root / output_rel
    Image.fromarray(LUT[pred]).save(output_path, format="JPEG", quality=95)
    return index, {"width": image.width, "height": image.height, "output": str(output_rel).replace(os.sep, "\\"),
                   "level": level(ratio), "p": p, "w": w, "ratio": ratio,
                   "seconds": time.perf_counter() - started}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B", "C"], default="A")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    root = Path.cwd()
    rows = read_input(root / "input.csv")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDAが利用できません")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.mode == "B": ckpt = root / "weights" / "student_distilled.pt"
    else: ckpt = root / "weights" / "teacher1024.pt"
    model, _ = load_model(ckpt, args.device)
    model.eval()
    # Cの前処理は提出版の初期実装ではAと同じにし、出力仕様を先に安定させる。
    # 追加前処理は検証後にCへ実装する。
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = [pool.submit(process_one, i, r, root, model, args.mode, args.size, args.device) for i, r in enumerate(rows)]
        result = [j.result() for j in jobs]
    result.sort(key=lambda x: x[0])
    write_output(root / "output.csv", [r for _, r in result])
    with (root / "execution_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "image", "seconds"])
        for i, r in result: w.writerow([i, rows[i]["name"], f'{r["seconds"]:.6f}'])


if __name__ == "__main__": main()
