"""Contact sheet preview script for image preprocessing.

For every chest X-ray image in a directory (default: data/sample/), creates a
visual contact sheet at results/preview.png displaying each original image
side-by-side with its preprocessed counterpart (rescaled to 0-255 for display)
and labeled with filename and resolution.

Uses PIL and numpy only (no new dependencies).
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path for direct execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows console encoding protection
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.vision import VALID_IMAGE_EXTENSIONS, get_input_resolution, preprocess_image


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


def tensor_to_display_image(tensor_2d: np.ndarray) -> Image.Image:
    """Rescale a [-1024, 1024] preprocessed float array back to [0, 255] uint8 PIL Image.

    Args:
        tensor_2d: 2D float NumPy array with pixel intensities roughly in [-1024, 1024].

    Returns:
        Grayscale PIL Image ('L' mode) scaled to 0-255.
    """
    # Inverse of xrv.datasets.normalize: ((x / 1024.0) + 1.0) / 2.0 * 255.0
    rescaled = ((tensor_2d / 1024.0) + 1.0) * 127.5
    clipped = np.clip(rescaled, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="L")


def fit_image_in_box(img: Image.Image, box_size: int = 256) -> Image.Image:
    """Scale an image to fit inside a square box while preserving aspect ratio.

    The image is centered on a black background square.

    Args:
        img: Input PIL Image.
        box_size: Width and height of bounding square.

    Returns:
        PIL Image of exactly (box_size, box_size).
    """
    w, h = img.size
    scale = min(box_size / w, box_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (box_size, box_size), color=0)
    offset_x = (box_size - new_w) // 2
    offset_y = (box_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def create_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    target_res: int | None = None,
    pad_to_square: bool = False,
) -> None:
    """Generate and save contact sheet comparing original and preprocessed images.

    Args:
        image_paths: List of chest X-ray image paths.
        output_path: Destination path for results/preview[_pad].png.
        target_res: Optional target square resolution for preprocessing.
        pad_to_square: If True, pad to square instead of center crop.
    """
    if not image_paths:
        print("No images to preview.", file=sys.stderr)
        return

    # Display geometry constants
    box_size = 256
    padding_x = 24
    padding_y = 20
    panel_gap = 20
    card_gap = 24

    # Per-card layout
    header_height = 42
    caption_height = 28
    card_content_height = header_height + box_size + caption_height
    card_total_height = card_content_height + (padding_y * 2)

    card_width = (padding_x * 2) + (box_size * 2) + panel_gap
    canvas_width = card_width + 40
    canvas_height = 70 + (len(image_paths) * (card_total_height + card_gap))

    # Canvas initialization (dark slate background)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(20, 24, 32))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Sheet title
    draw.text((20, 20), "PREPROCESSING VERIFICATION CONTACT SHEET", fill=(255, 255, 255), font=font)
    mode_label = "pad_to_square" if pad_to_square else "XRayCenterCrop"
    res_label = f"Target Resolution: {target_res or get_input_resolution()}x{target_res or get_input_resolution()} | Pipeline: normalize -> single_ch -> {mode_label} -> XRayResizer"
    draw.text((20, 40), res_label, fill=(140, 160, 190), font=font)

    current_y = 70

    for idx, path in enumerate(image_paths, start=1):
        # 1. Load original image
        orig_pil = Image.open(path).convert("L")
        orig_w, orig_h = orig_pil.size
        orig_display = fit_image_in_box(orig_pil, box_size=box_size)

        # 2. Preprocess with pipeline
        tensor = preprocess_image(path, input_size=target_res, pad_to_square=pad_to_square)
        tensor_np = tensor[0, 0].cpu().numpy()
        prep_h, prep_w = tensor_np.shape
        prep_display_raw = tensor_to_display_image(tensor_np)
        prep_display = fit_image_in_box(prep_display_raw, box_size=box_size)

        # Card container bounds
        card_x0 = 20
        card_y0 = current_y
        card_x1 = card_x0 + card_width
        card_y1 = card_y0 + card_total_height

        # Card background
        draw.rectangle([card_x0, card_y0, card_x1, card_y1], fill=(30, 36, 48), outline=(50, 60, 80))

        # Card header text
        text_x = card_x0 + padding_x
        text_y = card_y0 + padding_y
        header_text = f"[{idx}/{len(image_paths)}] File: {path.name}"
        meta_text = f"Original: {orig_w}x{orig_h} | Preprocessed: {prep_w}x{prep_h} | Min: {tensor.min().item():.1f}, Max: {tensor.max().item():.1f}"
        draw.text((text_x, text_y), header_text, fill=(255, 255, 255), font=font)
        draw.text((text_x, text_y + 16), meta_text, fill=(120, 200, 150), font=font)

        # Image panels placement
        img_y = text_y + header_height
        orig_x = text_x
        prep_x = orig_x + box_size + panel_gap

        canvas.paste(orig_display.convert("RGB"), (orig_x, img_y))
        canvas.paste(prep_display.convert("RGB"), (prep_x, img_y))

        # Panel captions
        draw.text((orig_x, img_y + box_size + 8), f"Original ({orig_w}x{orig_h})", fill=(180, 190, 205), font=font)
        draw.text((prep_x, img_y + box_size + 8), f"Preprocessed ({prep_w}x{prep_h})", fill=(180, 190, 205), font=font)

        current_y += card_total_height + card_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Contact sheet successfully saved to '{output_path}'.")


def main() -> None:
    """Parse arguments and create preprocessing contact sheet."""
    parser = argparse.ArgumentParser(
        description="Create side-by-side contact sheet for chest X-ray image preprocessing verification."
    )
    parser.add_argument(
        "--folder",
        "-f",
        type=Path,
        default=Path("data/sample"),
        help="Directory containing images to preview (default: data/sample)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path for saved contact sheet (default: results/preview[_pad].png)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=None,
        help="Target square resolution (default: derived from model_urls metadata)",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        default=False,
        help="Pad images to square instead of center-cropping",
    )
    args = parser.parse_args()

    # Determine default output file name
    pad_suffix = "_pad" if args.pad else ""
    default_output = Path("results") / f"preview{pad_suffix}.png"
    output_path = args.output or default_output

    images = find_images(args.folder)
    if not images:
        print(f"No chest X-ray images ({', '.join(sorted(VALID_IMAGE_EXTENSIONS))}) found in '{args.folder}'.", file=sys.stderr)
        sys.exit(1)

    mode_text = "with pad_to_square" if args.pad else "with center_crop"
    print(f"Found {len(images)} image(s) in '{args.folder}'. Generating contact sheet {mode_text}...")
    create_contact_sheet(images, output_path, target_res=args.size, pad_to_square=args.pad)


if __name__ == "__main__":
    main()
