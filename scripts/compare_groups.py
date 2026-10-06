"""Group comparison script for chest X-ray model score cohorts.

For every results/scores_*.csv file:
1. Filters out 'other' group rows (evaluates 'normal' vs 'pneu').
2. Counts unique images per group.
3. Computes per-pathology:
   - Mean score for normal group
   - Mean score for pneu group
   - Difference (pneu - normal)
   - ROC AUC (distinguishing pneu from normal using sklearn.metrics.roc_auc_score)
4. Displays formatted tables sorted by AUC descending for each file.
5. Summarizes and identifies the file with the highest mean AUC across pathologies.

Uses pandas and scikit-learn only.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Console encoding protection for Windows shells
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def analyze_scores_file(csv_path: Path) -> dict:
    """Analyze a single scores CSV file for normal vs pneu separation.

    Args:
        csv_path: Path to scores_*.csv file.

    Returns:
        Dictionary containing file metadata, per-group image counts,
        per-pathology metrics DataFrame, and overall mean AUC.
    """
    df = pd.read_csv(csv_path)

    # Filter out rows with group 'other'
    df_eval = df[df["group"].isin(["normal", "pneu"])].copy()
    if df_eval.empty:
        return {
            "file": csv_path.name,
            "path": csv_path,
            "error": "No 'normal' or 'pneu' rows found.",
            "metrics": None,
            "mean_auc": None,
        }

    # Count unique images per group
    normal_images = df_eval[df_eval["group"] == "normal"]["image"].nunique()
    pneu_images = df_eval[df_eval["group"] == "pneu"]["image"].nunique()
    total_images = df_eval["image"].nunique()

    rows = []
    aucs = []

    for pathology, group_df in df_eval.groupby("pathology"):
        normal_scores = group_df[group_df["group"] == "normal"]["score"].dropna()
        pneu_scores = group_df[group_df["group"] == "pneu"]["score"].dropna()

        mean_normal = float(normal_scores.mean()) if not normal_scores.empty else float("nan")
        mean_pneu = float(pneu_scores.mean()) if not pneu_scores.empty else float("nan")
        diff = mean_pneu - mean_normal if (not np.isnan(mean_pneu) and not np.isnan(mean_normal)) else float("nan")

        # Binary ground truth: 1 for pneu (positive class), 0 for normal (negative class)
        valid_df = group_df.dropna(subset=["score"])
        y_true = (valid_df["group"] == "pneu").astype(int)
        y_scores = valid_df["score"]

        if len(y_true.unique()) >= 2:
            try:
                auc = float(roc_auc_score(y_true, y_scores))
                aucs.append(auc)
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        rows.append({
            "pathology": pathology,
            "mean_normal": mean_normal,
            "mean_pneu": mean_pneu,
            "diff": diff,
            "auc": auc,
        })

    metrics_df = pd.DataFrame(rows)
    # Sort by AUC descending, then diff descending
    metrics_df = metrics_df.sort_values(by=["auc", "diff"], ascending=[False, False]).reset_index(drop=True)
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")

    return {
        "file": csv_path.name,
        "path": csv_path,
        "normal_images": normal_images,
        "pneu_images": pneu_images,
        "total_images": total_images,
        "metrics": metrics_df,
        "mean_auc": mean_auc,
    }


def print_table(analysis: dict) -> None:
    """Print a formatted ASCII table of pathology separation metrics."""
    filename = analysis["file"]
    print("=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    if analysis.get("error"):
        print(f"Error: {analysis['error']}\n")
        return

    n_normal = analysis["normal_images"]
    n_pneu = analysis["pneu_images"]
    n_total = analysis["total_images"]
    print(f"Cohort: {n_total} images ({n_normal} normal, {n_pneu} pneu)")
    print("-" * 80)

    header = f"{'Pathology':<28} {'Mean Normal':>12} {'Mean Pneu':>12} {'Difference':>12} {'ROC AUC':>10}"
    print(header)
    print("-" * 80)

    metrics_df = analysis["metrics"]
    for _, row in metrics_df.iterrows():
        p_name = row["pathology"]
        m_norm = f"{row['mean_normal']:.4f}" if not np.isnan(row["mean_normal"]) else "N/A"
        m_pneu = f"{row['mean_pneu']:.4f}" if not np.isnan(row["mean_pneu"]) else "N/A"
        m_diff = f"{row['diff']:+.4f}" if not np.isnan(row["diff"]) else "N/A"
        m_auc = f"{row['auc']:.4f}" if not np.isnan(row["auc"]) else "N/A"

        print(f"{p_name:<28} {m_norm:>12} {m_pneu:>12} {m_diff:>12} {m_auc:>10}")

    print("-" * 80)
    mean_auc = analysis["mean_auc"]
    if not np.isnan(mean_auc):
        print(f"Mean AUC across all pathologies: {mean_auc:.4f}")
    else:
        print("Mean AUC across all pathologies: N/A")
    print()


def main() -> None:
    """Find all scores_*.csv files, compute metrics, display tables, and summarize best."""
    parser = argparse.ArgumentParser(
        description="Compare normal vs pneu score distributions and compute ROC AUC per pathology."
    )
    parser.add_argument(
        "--results-dir",
        "-r",
        type=Path,
        default=Path("results"),
        help="Directory containing scores_*.csv files (default: results)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    csv_files = sorted(results_dir.glob("scores_*.csv"))

    if not csv_files:
        sys.stderr.write(f"No scores_*.csv files found in '{results_dir}'.\n")
        sys.exit(1)

    all_analyses = []
    for csv_file in csv_files:
        analysis = analyze_scores_file(csv_file)
        all_analyses.append(analysis)
        print_table(analysis)

    # Identify file(s) with the highest mean AUC
    valid_analyses = [a for a in all_analyses if a["mean_auc"] is not None and not np.isnan(a["mean_auc"])]
    if not valid_analyses:
        print("No valid AUC calculations could be made.")
        return

    # Sort runs by mean AUC descending
    ranked = sorted(valid_analyses, key=lambda a: a["mean_auc"], reverse=True)
    best_mean_auc = ranked[0]["mean_auc"]
    # Account for potential ties
    best_files = [a["file"] for a in ranked if np.isclose(a["mean_auc"], best_mean_auc, atol=1e-5)]

    print("=" * 80)
    print("HIGHEST MEAN AUC SUMMARY")
    print("=" * 80)
    if len(best_files) == 1:
        print(f"File with highest mean AUC: {best_files[0]} (Mean AUC: {best_mean_auc:.4f})")
    else:
        print(f"Files tied with highest mean AUC ({best_mean_auc:.4f}):")
        for bf in best_files:
            print(f"  - {bf}")

    print("\nComplete Ranking (Mean AUC across pathologies):")
    for rank, a in enumerate(ranked, start=1):
        print(f"  {rank}. {a['file']:<40} Mean AUC = {a['mean_auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
