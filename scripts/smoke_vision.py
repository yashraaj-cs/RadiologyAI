"""Smoke test script for running TorchXRayVision DenseNet inference on chest X-rays.

Refactored to delegate model loading and image inference to app.vision.
Takes a folder of chest X-ray images (default: data/sample/), runs prediction,
and prints the top 5 predicted pathology model scores per image.
"""

import argparse
import sys
from pathlib import Path

# Ensure Windows console encoding handles progress bar glyphs (e.g. \u2588) without charmap error
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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


def main() -> None:
    """Parse arguments, validate image directory, and run vision inference."""
    parser = argparse.ArgumentParser(
        description="Run TorchXRayVision DenseNet smoke test on chest X-ray images."
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
        default=None,
        help="TorchXRayVision DenseNet weights (default: from app.config.MODEL_WEIGHTS)",
    )
    args = parser.parse_args()

    # Step 1: Scan for images before initializing model weights
    folder: Path = args.folder
    images = find_images(folder)

    if not images:
        sys.stderr.write(
            f"Error: No chest X-ray images ({', '.join(sorted(VALID_IMAGE_EXTENSIONS))}) found in '{folder}'.\n"
            f"Please place PNG or JPEG sample chest X-rays into '{folder}' and run again.\n"
        )
        sys.exit(1)

    print(f"Found {len(images)} image(s) in '{folder}'. Initializing vision model...")

    # Step 2: Load model once (cached via app.vision)
    model = load_model(weights=args.weights)

    # Step 3: Run inference on each image and display top 5 model scores
    weights_name = args.weights or getattr(model, "weights", None) or "default"
    print(f"Running inference using '{weights_name}'...\n")

    for img_path in images:
        try:
            scores = predict(img_path, model=model)
            top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

            print(f"Image: {img_path.name}")
            print("-" * 45)
            for rank, (pathology, score) in enumerate(top5, start=1):
                print(f"  {rank}. {pathology:<25} {score:.4f} ({score * 100:.1f}%)")
            print()
        except Exception as exc:
            print(f"Error processing '{img_path.name}': {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
