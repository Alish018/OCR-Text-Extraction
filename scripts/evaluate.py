# scripts/evaluate.py
"""
Evaluate extraction on a folder of images with a ground-truth CSV.
Usage:
  python scripts/evaluate.py --images_folder images --gt tests/ground_truth.csv
"""

import argparse
import pandas as pd
import os
from src.text_extraction import extract_target_line_from_image
from rapidfuzz.distance import Levenshtein

def evaluate(images_folder, gt_csv):
    df = pd.read_csv(gt_csv)
    rows = []
    for _, r in df.iterrows():
        fname = str(r['filename'])
        expected = str(r['expected']).strip()
        path = os.path.join(images_folder, fname)
        if not os.path.exists(path):
            rows.append({
                "filename": fname, "expected": expected, "extracted": "",
                "match": False, "char_acc": 0.0, "method": "file_missing"
            })
            continue
        extracted, info = extract_target_line_from_image(path)
        match = (extracted == expected)
        dist = Levenshtein.distance(extracted, expected)
        maxlen = max(1, len(expected))
        char_acc = 1 - (dist / maxlen)
        rows.append({
            "filename": fname, "expected": expected, "extracted": extracted,
            "match": match, "char_acc": round(char_acc, 3), "method": info.get("method","")
        })
    out = pd.DataFrame(rows)
    out.to_csv("results/eval_results.csv", index=False)
    exact = out['match'].mean()
    char_mean = out['char_acc'].mean()
    print(f"Exact match accuracy: {exact:.3f}")
    print(f"Mean char-level accuracy: {char_mean:.3f}")
    print("Saved: results/eval_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_folder", required=True, help="Path to folder with images")
    parser.add_argument("--gt", default="tests/ground_truth.csv", help="Ground truth CSV")
    args = parser.parse_args()
    evaluate(args.images_folder, args.gt)
