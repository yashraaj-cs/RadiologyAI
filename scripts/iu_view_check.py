"""IU X-Ray view detector contact sheet generator.

Picks 40 random 2-image studies (seed 0) and generates two contact sheets
(results/iu_view_check_1.png and results/iu_view_check_2.png) of 20 studies each.
Each image is downscaled to about 256 px and labeled with image_id, symmetry score,
and detector guess ('frontal' for higher score, 'lateral' for lower).
Uses PIL only.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def create_contact_sheet(
    studies: List[Tuple[str, List[dict]]],
    images_dir: Path,
    output_path: Path,
    title: str = "IU X-Ray Frontal vs Lateral View Check",
) -> None:
    """Generate and save a single contact sheet for up to 20 studies using PIL only."""
    num_studies = len(studies)
    width = 720
    row_height = 310
    header_height = 50
    total_height = header_height + num_studies * row_height + 20

    canvas = Image.new("RGB", (width, total_height), color=(24, 24, 28))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # Draw sheet header
    draw.text((24, 18), title, fill=(255, 255, 255), font=font)
    draw.line([(24, 42), (width - 24, 42)], fill=(60, 60, 68), width=1)

    for idx, (rep_id, records) in enumerate(studies):
        row_y = header_height + idx * row_height

        # Draw row separator line if not first
        if idx > 0:
            draw.line([(24, row_y - 8), (width - 24, row_y - 8)], fill=(40, 40, 48), width=1)

        # Study label on left
        draw.text((24, row_y + 12), f"#{idx + 1:02d}", fill=(140, 140, 150), font=font)
        draw.text((24, row_y + 30), rep_id, fill=(220, 220, 220), font=font)

        # Sort images by image_id for consistent ordering
        records = sorted(records, key=lambda x: x["image_id"])
        s0, s1 = records[0]["symmetry"], records[1]["symmetry"]
        guess0 = "frontal" if s0 >= s1 else "lateral"
        guess1 = "lateral" if s0 >= s1 else "frontal"

        slots = [(records[0], guess0, 120), (records[1], guess1, 410)]

        for rec, guess, slot_x in slots:
            iid = rec["image_id"]
            sym = rec["symmetry"]

            # Label text above image
            draw.text((slot_x, row_y + 6), iid, fill=(230, 230, 230), font=font)
            guess_color = (74, 222, 128) if guess == "frontal" else (251, 146, 60)
            status_text = f"sym: {sym:.4f}  [{guess.upper()}]"
            draw.text((slot_x, row_y + 22), status_text, fill=guess_color, font=font)

            # Image box outline
            box_y = row_y + 40
            draw.rectangle(
                [slot_x - 1, box_y - 1, slot_x + 256, box_y + 256],
                outline=(60, 60, 68),
                fill=(12, 12, 14),
            )

            # Load and downscale image to ~256 px
            img_path = images_dir / f"{iid}.png"
            if img_path.exists():
                with Image.open(img_path) as orig:
                    im = orig.convert("RGB")
                    im.thumbnail((256, 256), Image.Resampling.BILINEAR)
                    w, h = im.size
                    # Center in 256x256 slot
                    offset_x = slot_x + (256 - w) // 2
                    offset_y = box_y + (256 - h) // 2
                    canvas.paste(im, (offset_x, offset_y))
            else:
                draw.text((slot_x + 10, box_y + 120), "FILE NOT FOUND", fill=(255, 100, 100), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    print(f"Saved contact sheet to {output_path} ({canvas.size[0]}x{canvas.size[1]} px)")


def generate_view_checks(
    images_csv: Path = Path("data/iu_xray/index/images.csv"),
    images_dir: Path = Path("data/iu_xray/images"),
    results_dir: Path = Path("results"),
    seed: int = 0,
) -> None:
    """Select 40 random 2-image studies and create two 20-study contact sheets."""
    df_images = pd.read_csv(images_csv)

    # Group by report_id and find 2-image studies
    grouped = df_images.groupby("report_id")
    two_img_studies = {rep_id: group.to_dict("records") for rep_id, group in grouped if len(group) == 2}

    print(f"Total 2-image studies available: {len(two_img_studies)}")

    sorted_rep_ids = sorted(two_img_studies.keys())
    rng = random.Random(seed)
    selected_rep_ids = rng.sample(sorted_rep_ids, 40)

    sheet_1_studies = [(rep_id, two_img_studies[rep_id]) for rep_id in selected_rep_ids[:20]]
    sheet_2_studies = [(rep_id, two_img_studies[rep_id]) for rep_id in selected_rep_ids[20:40]]

    out1 = results_dir / "iu_view_check_1.png"
    out2 = results_dir / "iu_view_check_2.png"

    create_contact_sheet(sheet_1_studies, images_dir, out1, title="IU X-Ray View Check 1/2 (Seed 0, Studies 1-20)")
    create_contact_sheet(sheet_2_studies, images_dir, out2, title="IU X-Ray View Check 2/2 (Seed 0, Studies 21-40)")


if __name__ == "__main__":
    generate_view_checks()
