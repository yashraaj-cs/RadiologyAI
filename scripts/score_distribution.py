"""Score distribution analysis script for chest X-ray image cohorts.

Runs the vision and rules pipeline over every image in a directory (default: data/sample/),
writes per-finding model scores and categorical statuses to results/scores_<weights>.csv,
and prints cohort summary statistics per group ("normal", "pneu", "other").
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure project root is in sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Console encoding protection for Windows shells
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.config import MODEL_WEIGHTS
from app.rules import findings_from_scores
from app.vision import VALID_IMAGE_EXTENSIONS, load_model, predict


def find_images(folder: Path) -> list[Path]:
    """Find all PNG and JPEG images within the given folder.

    Args:
        folder: Path to directory to scan.

    Returns:
        Sorted list of matching image Path objects.
    """
    if not folder.is_dir():
        return []
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    )


def classify_group(filename: str) -> str:
    """Classify image into cohort group based on filename prefix.

    - "normal" if filename starts with "normal_"
    - "pneu" if filename starts with "pneu_"
    - "other" for any other prefix or pattern

    Args:
        filename: Name of the image file.

    Returns:
        Group classification string ("normal", "pneu", or "other").
    """
    lower = filename.lower()
    if lower.startswith("normal_"):
        return "normal"
    elif lower.startswith("pneu_"):
        return "pneu"
    else:
        return "other"


def main() -> None:
    """Run pipeline over image cohort, save CSV, and print distribution summary per group."""
    parser = argparse.ArgumentParser(
        description="Compute pathology score distribution over chest X-ray image cohort."
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=Path,
        default=Path("data/sample"),
        help="Folder containing chest X-ray images (default: data/sample)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=MODEL_WEIGHTS,
        help=f"Pretrained weights name (default: {MODEL_WEIGHTS})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Destination CSV path (default: results/scores_<weights>[_pad].csv)",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        default=False,
        help="Pad images to square instead of center-cropping",
    )
    args = parser.parse_args()

    # Determine output file path
    pad_suffix = "_pad" if args.pad else ""
    default_filename = f"scores_{args.weights}{pad_suffix}.csv"
    output_path = args.output or (Path("results") / default_filename)

    # Step 1: Scan for images
    images = find_images(args.folder)
    if not images:
        sys.stderr.write(
            f"Error: No chest X-ray images ({', '.join(sorted(VALID_IMAGE_EXTENSIONS))}) found in '{args.folder}'.\n"
        )
        sys.exit(1)

    pad_msg = " with pad_to_square" if args.pad else " with center_crop"
    print(f"Found {len(images)} image(s) in '{args.folder}'. Loading vision model with weights '{args.weights}'{pad_msg}...")
    model = load_model(weights=args.weights)

    # Step 2: Prepare CSV output directory and writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[str]] = []

    # Per-group statistics tracking
    group_images: dict[str, list[Path]] = defaultdict(list)
    group_present_counts: dict[str, list[int]] = defaultdict(list)
    group_pathology_counters: dict[str, Counter[str]] = defaultdict(Counter)

    print("Processing images and computing score distributions...")

    for img_path in images:
        group = classify_group(img_path.name)
        group_images[group].append(img_path)

        try:
            scores = predict(img_path, model=model, pad_to_square=args.pad)
            findings = findings_from_scores(scores, weights=args.weights)

            image_present_count = 0
            for f in findings:
                rows.append([img_path.name, group, f.name, f"{f.probability:.4f}", f.status])
                if f.status == "present":
                    image_present_count += 1
                    group_pathology_counters[group][f.name] += 1

            group_present_counts[group].append(image_present_count)
        except Exception as exc:
            print(f"Error processing '{img_path.name}': {exc}", file=sys.stderr)

    # Step 3: Write CSV results
    with open(output_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image", "group", "pathology", "score", "status"])
        writer.writerows(rows)

    print(f"\nSaved detailed score rows to '{output_path}'.\n")

    # Step 4: Compute and print per-group summary metrics
    mode_tag = " (pad_to_square)" if args.pad else " (center_crop)"
    print("=" * 60)
    print(f"SCORE DISTRIBUTION SUMMARY (Weights: {args.weights}{mode_tag})")
    print("=" * 60)

    for group_name in sorted(group_images.keys()):
        imgs_in_group = len(group_images[group_name])
        counts = group_present_counts[group_name]
        avg_present = sum(counts) / imgs_in_group if imgs_in_group > 0 else 0.0
        counter = group_pathology_counters[group_name]

        print(f"\nGroup: {group_name}")
        print(f"  Images processed: {imgs_in_group}")
        print(f"  Average number of present findings: {avg_present:.2f}")
        print("  5 pathologies most often present:")

        top5_present = counter.most_common(5)
        if top5_present:
            for rank, (pathology, count) in enumerate(top5_present, start=1):
                pct = (count / imgs_in_group) * 100 if imgs_in_group > 0 else 0.0
                print(f"    {rank}. {pathology:<25} {count} ({pct:.1f}% of group images)")
        else:
            print("    (No findings met the present threshold)")

    # Overall summary line
    total_imgs = sum(len(imgs) for imgs in group_images.values())
    all_counts = [c for counts in group_present_counts.values() for c in counts]
    overall_avg = sum(all_counts) / total_imgs if total_imgs > 0 else 0.0
    print("-" * 60)
    print(f"Total cohort: {total_imgs} images, average {overall_avg:.2f} present findings per image.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
