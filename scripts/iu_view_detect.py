"""IU X-Ray view detection and symmetry computation.

Computes left-right symmetry score for each image via Pearson correlation
between the image and its horizontal flip. Generates images.csv index and
prints distribution statistics for 2-image and 1-image studies.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image


def compute_symmetry(img_or_arr: Union[Image.Image, np.ndarray]) -> float:
    """Compute left-right symmetry score = Pearson correlation with horizontal flip.

    Args:
        img_or_arr: PIL Image or 2D NumPy array. If PIL Image, downscaled to 128x128
                    grayscale. If NumPy array of shape other than 128x128, converted
                    and resized.

    Returns:
        float: Pearson correlation in range [-1.0, 1.0]. Returns 0.0 for uniform images.
    """
    if isinstance(img_or_arr, Image.Image):
        gray = img_or_arr.convert("L").resize((128, 128), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.float64)
    elif isinstance(img_or_arr, np.ndarray):
        arr = img_or_arr.astype(np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape}")
        if arr.shape != (128, 128):
            im = Image.fromarray(arr)
            arr = np.array(im.resize((128, 128), Image.Resampling.BILINEAR), dtype=np.float64)
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_arr)}")

    flipped = np.fliplr(arr)
    a = arr.ravel()
    b = flipped.ravel()

    std_a = np.std(a)
    std_b = np.std(b)
    if std_a == 0.0 or std_b == 0.0:
        return 0.0

    corr = np.corrcoef(a, b)[0, 1]
    return float(corr)


def process_single_image(path: Path) -> Dict[str, Any]:
    """Process a single PNG file: get dimensions and symmetry score."""
    with Image.open(path) as img:
        width, height = img.size
        sym = compute_symmetry(img)
    image_id = path.stem
    report_id = image_id.split("_")[0]
    return {
        "image_id": image_id,
        "report_id": report_id,
        "width": width,
        "height": height,
        "symmetry": sym,
    }


def analyze_and_report(df_images: pd.DataFrame) -> None:
    """Compute and print metrics for 2-image studies, suffix distribution, and 1-image studies."""
    # Group by report_id
    grouped = df_images.groupby("report_id")

    # 3a. 2-image studies
    two_img_studies = [group for _, group in grouped if len(group) == 2]
    higher_scores: List[float] = []
    lower_scores: List[float] = []
    gaps: List[float] = []

    # 3b. Suffix tracking
    suffix_higher_counts: Dict[str, int] = {}
    suffix_total_counts: Dict[str, int] = {}

    for group in two_img_studies:
        records = group.to_dict("records")
        r1, r2 = records[0], records[1]
        s1, s2 = r1["symmetry"], r2["symmetry"]

        if s1 >= s2:
            higher, lower = s1, s2
            winner_suffix = r1["image_id"].split("-")[-1]
        else:
            higher, lower = s2, s1
            winner_suffix = r2["image_id"].split("-")[-1]

        higher_scores.append(higher)
        lower_scores.append(lower)
        gaps.append(higher - lower)

        suffix1 = r1["image_id"].split("-")[-1]
        suffix2 = r2["image_id"].split("-")[-1]
        suffix_total_counts[suffix1] = suffix_total_counts.get(suffix1, 0) + 1
        suffix_total_counts[suffix2] = suffix_total_counts.get(suffix2, 0) + 1
        suffix_higher_counts[winner_suffix] = suffix_higher_counts.get(winner_suffix, 0) + 1

    print("\n" + "=" * 70)
    print(f"3a. 2-IMAGE STUDIES DISTRIBUTION (N = {len(two_img_studies)} studies)")
    print("=" * 70)

    def print_distribution(name: str, values: List[float]) -> None:
        s = pd.Series(values)
        q = s.quantile([0.0, 0.25, 0.50, 0.75, 1.0])
        print(f"  {name:15s} | Min: {q[0.0]:.4f} | Q1: {q[0.25]:.4f} | Median: {q[0.50]:.4f} | Q3: {q[0.75]:.4f} | Max: {q[1.0]:.4f} | Mean: {s.mean():.4f}")

    print_distribution("Higher Score", higher_scores)
    print_distribution("Lower Score", lower_scores)
    print_distribution("Score Gap", gaps)

    print("\n" + "=" * 70)
    print("3b. FILE-NAME SUFFIX DISTRIBUTION IN 2-IMAGE STUDIES")
    print("=" * 70)
    print(f"  {'Suffix':<12} | {'Higher (Won)':<14} | {'Total in 2-Img':<16} | {'Higher Win Rate':<16}")
    print("  " + "-" * 62)

    # Sort suffixes by total appearances descending
    sorted_suffixes = sorted(suffix_total_counts.keys(), key=lambda x: suffix_total_counts[x], reverse=True)
    for sfx in sorted_suffixes:
        won = suffix_higher_counts.get(sfx, 0)
        tot = suffix_total_counts[sfx]
        pct = (won / tot) * 100.0 if tot > 0 else 0.0
        print(f"  {sfx:<12} | {won:<14d} | {tot:<16d} | {pct:>6.2f}%")

    # 3c. Single-image studies
    single_img_studies = [group for _, group in grouped if len(group) == 1]
    single_scores = [group.iloc[0]["symmetry"] for group in single_img_studies]

    print("\n" + "=" * 70)
    print(f"3c. SINGLE-IMAGE STUDIES DISTRIBUTION (N = {len(single_img_studies)} studies)")
    print("=" * 70)
    print_distribution("Symmetry", single_scores)
    print("=" * 70 + "\n")


def build_images_index(
    images_dir: Union[str, Path] = "data/iu_xray/images",
    output_csv: Union[str, Path] = "data/iu_xray/index/images.csv",
    num_workers: int = 8,
) -> pd.DataFrame:
    """Compute symmetry scores for all PNGs and save to images.csv."""
    images_dir = Path(images_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    png_files = sorted(images_dir.glob("*.png"))
    print(f"Processing {len(png_files)} images from {images_dir} with {num_workers} threads...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        records = list(executor.map(process_single_image, png_files))

    df_images = pd.DataFrame(records)
    # Sort for deterministic output
    df_images = df_images.sort_values("image_id").reset_index(drop=True)
    df_images.to_csv(output_csv, index=False)
    print(f"Saved {len(df_images)} image records to {output_csv}")
    return df_images


if __name__ == "__main__":
    images_dir = Path("data/iu_xray/images")
    output_csv = Path("data/iu_xray/index/images.csv")
    df_images = build_images_index(images_dir, output_csv)
    analyze_and_report(df_images)
