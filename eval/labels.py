"""IU X-Ray per-finding label derivation from index terms.

Maps model pathologies to keyword stems, matches them case-insensitively against
the part of each term before the first '/', applies specialized overrides for
Pleural_Thickening and Effusion, evaluates ambiguous patterns against full term text,
and assigns one of 5 discrete statuses following strict priority:
  1. Any non-ambiguous major match -> 'pos'
  2. Else any ambiguous match (major or auto) -> 'ambiguous'
  3. Else any auto match -> 'pos_auto_only'
  4. Else major terms exactly ['normal'] -> 'neg_clean'
  5. Else -> 'neg_assumed'

For 'pos' and 'pos_auto_only' rows, evaluates 'negated_text' and 'hedged_text'
flags from the cleaned findings and impression sections without altering the status.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

# Legacy stub compatibility
DATASET_TO_MODEL_PATHOLOGIES: Dict[str, Dict[str, str]] = {
    "chexpert": {},
    "nih": {},
    "mimic": {},
}


def map_dataset_label(dataset_name: str, label: str) -> Optional[str]:
    """Map a raw dataset label to the corresponding model pathology key (stub)."""
    mapping = DATASET_TO_MODEL_PATHOLOGIES.get(dataset_name.lower(), {})
    return mapping.get(label)


# Standard 18 pathologies from TorchXRayVision
ALL_PATHOLOGIES: List[str] = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
]

# Mapping from model pathology to keyword stems matched before first '/'
PATHOLOGY_STEMS: Dict[str, List[str]] = {
    "Atelectasis": ["atelect"],
    "Consolidation": ["consolidat"],
    "Infiltration": ["infiltrat"],
    "Pneumothorax": ["pneumothorax", "pneumothorac"],
    "Edema": ["edema"],
    "Emphysema": ["emphysema"],
    "Fibrosis": ["fibros", "fibrot"],
    "Effusion": ["pleural effusion"],  # auto terms also match 'effusion' without 'pericardial'
    "Pneumonia": ["pneumonia"],
    "Pleural_Thickening": ["pleural thickening"],  # major also matches prefix 'thickening' with 'pleura' in full
    "Cardiomegaly": ["cardiomegaly"],
    "Nodule": ["nodul"],
    "Mass": ["mass"],
    "Hernia": ["hernia"],
    "Lung Lesion": [],  # unmapped
    "Fracture": ["fracture"],
    "Lung Opacity": ["opacity", "opacities", "airspace disease"],
    "Enlarged Cardiomediastinum": [],  # unmapped
}

MAPPED_PATHOLOGIES: List[str] = [p for p, stems in PATHOLOGY_STEMS.items() if stems]

# Ambiguous patterns tested against the FULL term text
AMBIGUOUS_PATTERNS: Dict[str, List[str]] = {
    "Emphysema": ["subcutaneous"],
    "Mass": ["mediastin", "thyroid", "paratracheal", "lesion"],
    "Cardiomegaly": ["borderline"],
    "Fibrosis": ["mediastinal"],
    "Fracture": ["healed"],
}

# Negation and hedge cues for text flags
NEGATION_PREFIXES: List[str] = [
    r"no\s+evidence\s+of",
    r"negative\s+for",
    r"free\s+of",
    r"absence\s+of",
    r"no",
    r"without",
]

NEGATION_SUFFIXES: List[str] = [
    r"not\s+seen",
    r"resolved",
]

NEGATION_EXCEPTIONS: List[str] = [
    r"without\s+significant\s+interval\s+change",
    r"not\s+seen\s+on\s+prior",
    r"no\s+significant\s+interval\s+change",
    r"no\s+interval\s+change",
    r"no\s+significant\s+change",
    r"no\s+change",
    r"unchanged",
    r"no\s+other",
]

NEGATION_EXCEPTION_RE = re.compile(
    rf"\b(?:{'|'.join(NEGATION_EXCEPTIONS)})\b",
    re.IGNORECASE,
)

HEDGE_CUES: List[str] = [
    r"may\s+represent",
    r"cannot\s+exclude",
    r"possible",
    r"possibly",
    r"probable",
    r"probably",
    r"likely",
    r"versus",
    r"suspect",
]

HEDGE_RE = re.compile(rf"\b(?:{'|'.join(HEDGE_CUES)})\b", re.IGNORECASE)


def derive_eval_role(
    status: str,
    negated_text: bool = False,
    hedged_text: bool = False,
    redacted_near_stem: bool = False,
) -> str:
    """Derive evaluation role for a pathology label row:
      - 'positive': status pos AND negated_text False AND hedged_text False AND redacted_near_stem False
      - 'excluded': status ambiguous or pos_auto_only, or pos with either flag True, or redacted_near_stem True
      - 'negative_clean': status neg_clean (and not redacted_near_stem)
      - 'negative_assumed': status neg_assumed (and not redacted_near_stem)
    """
    if redacted_near_stem:
        return "excluded"
    if status == "pos":
        if not negated_text and not hedged_text:
            return "positive"
        return "excluded"
    if status in ("ambiguous", "pos_auto_only"):
        return "excluded"
    if status == "neg_clean":
        return "negative_clean"
    if status == "neg_assumed":
        return "negative_assumed"
    raise ValueError(f"Unknown status: {status}")


def match_term_for_pathology(pathology: str, term: str, is_major: bool = True) -> bool:
    """Check if a term matches a pathology, respecting major/auto overrides.

    Overrides:
      - Pleural_Thickening: also match major terms whose prefix is 'thickening'
        and whose full text contains 'pleura'.
      - Effusion: for auto terms only, also match generic 'effusion', but never
        when the term contains 'pericardial'.
    """
    if not term:
        return False
    prefix = term.split("/")[0].strip().lower()
    full = term.lower()

    if pathology == "Pleural_Thickening":
        if "pleural thickening" in prefix:
            return True
        if is_major and "thickening" in prefix and "pleura" in full:
            return True
        return False

    if pathology == "Effusion":
        if "pericardial" in full:
            return False
        if "pleural effusion" in prefix:
            return True
        if not is_major and "effusion" in prefix:
            return True
        return False

    stems = PATHOLOGY_STEMS.get(pathology, [])
    return any(s in prefix for s in stems)


def match_term(term: str, stems: List[str]) -> bool:
    """Generic check if any keyword stem appears case-insensitively before the first '/'."""
    if not stems or not term:
        return False
    prefix = term.split("/")[0].strip().lower()
    return any(s.lower() in prefix for s in stems)


def is_term_ambiguous(pathology: str, term: str) -> bool:
    """Check if a term contains any ambiguous pattern in its full text."""
    patterns = AMBIGUOUS_PATTERNS.get(pathology, [])
    if not patterns:
        return False
    full = term.lower()
    return any(pat in full for pat in patterns)


def is_normal_clean(major_terms: List[str]) -> bool:
    """Check if major terms are exactly ['normal'] (case-insensitive)."""
    cleaned = [m.strip().lower() for m in major_terms if m.strip()]
    return cleaned == ["normal"]


def derive_pathology_status(
    major_terms: List[str],
    auto_terms: List[str],
    pathology: str,
) -> str:
    """Derive discrete status for a single pathology following priority:
      1. Any non-ambiguous major match -> 'pos'
      2. Else any ambiguous match (major or auto) -> 'ambiguous'
      3. Else any auto match -> 'pos_auto_only'
      4. Else major terms exactly ['normal'] -> 'neg_clean'
      5. Else -> 'neg_assumed'
    """
    matching_major = [t for t in major_terms if match_term_for_pathology(pathology, t, is_major=True)]
    matching_auto = [t for t in auto_terms if match_term_for_pathology(pathology, t, is_major=False)]

    # 1. Any non-ambiguous major match -> 'pos'
    if any(not is_term_ambiguous(pathology, t) for t in matching_major):
        return "pos"

    # 2. Any ambiguous match (major or auto) -> 'ambiguous'
    if any(is_term_ambiguous(pathology, t) for t in matching_major) or any(
        is_term_ambiguous(pathology, t) for t in matching_auto
    ):
        return "ambiguous"

    # 3. Any auto match (which must be non-ambiguous here) -> 'pos_auto_only'
    if matching_auto:
        return "pos_auto_only"

    # 4. Major terms exactly ['normal'] -> 'neg_clean'
    if is_normal_clean(major_terms):
        return "neg_clean"

    # 5. Anything else -> 'neg_assumed'
    return "neg_assumed"


def derive_study_labels(
    major_terms: List[str],
    auto_terms: List[str],
    pathologies: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Derive status dict for all requested pathologies for a single study."""
    if pathologies is None:
        pathologies = MAPPED_PATHOLOGIES
    return {p: derive_pathology_status(major_terms, auto_terms, p) for p in pathologies}


def get_sentences(text: Optional[str]) -> List[str]:
    """Split text into sentences while preserving numbered lists."""
    if not text or not isinstance(text, str):
        return []
    return [s.strip() for s in re.split(r"(?:(?<=[a-zA-Z0-9\]\)\'\"][.!?])\s+|\n+)", text) if s.strip()]


def is_stem_negated_in_sentence(sentence: str, stem: str) -> bool:
    """Check if the stem is negated in the sentence.

    A sentence must NOT count as negated when it contains 'no change',
    'no interval change', 'no significant change', 'unchanged', or 'no other'.
    """
    if not sentence or not stem:
        return False
    if NEGATION_EXCEPTION_RE.search(sentence):
        return False
    # 1. Negation cue before stem within clause
    pref_pat = rf"\b(?:{'|'.join(NEGATION_PREFIXES)})\b[^.!?:]*?\b{stem}"
    if re.search(pref_pat, sentence, re.IGNORECASE):
        return True
    # 2. Stem followed by negation suffix
    suff_pat = rf"\b{stem}[^.!?:]*?\b(?:{'|'.join(NEGATION_SUFFIXES)})\b"
    if re.search(suff_pat, sentence, re.IGNORECASE):
        return True
    return False


# Backward-compatible alias
is_sentence_negated = is_stem_negated_in_sentence


def check_text_flags(
    findings_clean: Optional[str],
    impression_clean: Optional[str],
    pathology: str,
) -> Tuple[bool, bool, List[str], List[str]]:
    """Evaluate negated_text and hedged_text flags from findings and impression.

    Returns:
        (has_negated_text, has_hedged_text, negated_sentences, hedged_sentences)
    """
    f_str = "" if not findings_clean or pd.isna(findings_clean) else str(findings_clean).strip()
    imp_str = "" if not impression_clean or pd.isna(impression_clean) else str(impression_clean).strip()
    all_text = " ".join([s for s in [f_str, imp_str] if s])
    sentences = get_sentences(all_text)

    stems = list(PATHOLOGY_STEMS.get(pathology, []))
    if pathology == "Effusion" and "effusion" not in stems:
        stems.append("effusion")
    if pathology == "Pleural_Thickening" and "thickening" not in stems:
        stems.append("thickening")

    has_neg = False
    has_hedge = False
    neg_matches: List[str] = []
    hedge_matches: List[str] = []

    for sent in sentences:
        s_lower = sent.lower()
        if not any(st in s_lower for st in stems):
            continue

        # Check negation of stem
        for st in stems:
            if is_stem_negated_in_sentence(sent, st):
                has_neg = True
                neg_matches.append(sent)
                break

        # Check hedge cues in the same sentence containing the stem
        if HEDGE_RE.search(sent):
            has_hedge = True
            hedge_matches.append(sent)

    return has_neg, has_hedge, neg_matches, hedge_matches


def check_redacted_near_stem(
    findings_clean: Optional[str],
    impression_clean: Optional[str],
    pathology: str,
) -> Tuple[bool, List[str]]:
    """Check if the sentence containing a pathology stem also contains [REDACTED].

    Returns:
        (has_redacted_near_stem, matching_sentences)
    """
    f_str = "" if not findings_clean or pd.isna(findings_clean) else str(findings_clean).strip()
    imp_str = "" if not impression_clean or pd.isna(impression_clean) else str(impression_clean).strip()
    all_text = " ".join([s for s in [f_str, imp_str] if s])
    sentences = get_sentences(all_text)

    stems = list(PATHOLOGY_STEMS.get(pathology, []))
    if pathology == "Effusion" and "effusion" not in stems:
        stems.append("effusion")
    if pathology == "Pleural_Thickening" and "thickening" not in stems:
        stems.append("thickening")

    redacted_matches: List[str] = []
    for sent in sentences:
        s_lower = sent.lower()
        if any(st in s_lower for st in stems):
            if "[REDACTED]" in sent:
                redacted_matches.append(sent)

    return len(redacted_matches) > 0, redacted_matches


def derive_dataset_labels(
    reports_df: pd.DataFrame,
    pathologies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate long table of labels per study and pathology with text flags and eval_role.

    Columns: report_id, pathology, status, has_image, negated_text, hedged_text, redacted_near_stem, eval_role.
    """
    if pathologies is None:
        pathologies = MAPPED_PATHOLOGIES

    rows: List[Dict[str, Any]] = []
    for _, row in reports_df.iterrows():
        rep_id = row["report_id"]
        has_img = bool(row.get("has_image", False))
        m_terms = row["major_terms"] if isinstance(row.get("major_terms"), list) else []
        a_terms = row["auto_terms"] if isinstance(row.get("auto_terms"), list) else []
        f_clean = row.get("findings_clean")
        imp_clean = row.get("impression_clean")

        for p in pathologies:
            st = derive_pathology_status(m_terms, a_terms, p)

            # Flags only computed for pos and pos_auto_only rows
            neg_flag = False
            hedge_flag = False
            if st in ("pos", "pos_auto_only"):
                neg_flag, hedge_flag, _, _ = check_text_flags(f_clean, imp_clean, p)

            red_flag, _ = check_redacted_near_stem(f_clean, imp_clean, p)
            role = derive_eval_role(st, neg_flag, hedge_flag, red_flag)

            rows.append({
                "report_id": rep_id,
                "pathology": p,
                "status": st,
                "has_image": has_img,
                "negated_text": neg_flag,
                "hedged_text": hedge_flag,
                "redacted_near_stem": red_flag,
                "eval_role": role,
            })

    return pd.DataFrame(rows)


def print_status_table(labels_df: pd.DataFrame) -> None:
    """Print status counts per pathology for studies with images."""
    subset = labels_df[labels_df["has_image"]]
    ct = pd.crosstab(subset["pathology"], subset["status"])
    cols = [c for c in ["pos", "ambiguous", "pos_auto_only", "neg_clean", "neg_assumed"] if c in ct.columns]
    ct = ct.reindex(columns=cols, fill_value=0)
    ct["total"] = ct.sum(axis=1)

    print("\n" + "=" * 95)
    print(f"STATUS COUNTS PER PATHOLOGY (Restricted to studies with image, N = {len(subset['report_id'].unique())})")
    print("=" * 95)
    print(ct.to_string())
    print("=" * 95)


def print_eval_role_table(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Print counts per pathology and eval_role for studies with an image, marking >= 60 positive."""
    subset = labels_df[labels_df["has_image"]]
    ct = pd.crosstab(subset["pathology"], subset["eval_role"])
    cols = [c for c in ["positive", "excluded", "negative_clean", "negative_assumed"] if c in ct.columns]
    ct = ct.reindex(columns=cols, fill_value=0)
    ct["total"] = ct.sum(axis=1)
    ct[">= 60 pos"] = ct["positive"].apply(lambda x: "YES (>=60)" if x >= 60 else "no")

    print("\n" + "=" * 95)
    print(f"EVALUATION ROLE COUNTS PER PATHOLOGY (Restricted to studies with image, N = {len(subset['report_id'].unique())})")
    print("=" * 95)
    print(ct.to_string())
    print("=" * 95)
    return ct


def print_all_remaining_negated_rows(
    reports_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Print ALL remaining negated rows (study, pathology, sentence, status)."""
    rep_lookup = {r["report_id"]: r for _, r in reports_df.iterrows()}

    neg_df = labels_df[labels_df["negated_text"]]
    rows_to_print: List[Dict[str, Any]] = []

    for _, row in neg_df.iterrows():
        rid = row["report_id"]
        p = row["pathology"]
        st = row["status"]
        has_img = row["has_image"]
        rep = rep_lookup.get(rid, {})
        f = rep.get("findings_clean")
        imp = rep.get("impression_clean")

        _, _, neg_sents, _ = check_text_flags(f, imp, p)
        sent = neg_sents[0] if neg_sents else "(No matching sentence extracted)"
        rows_to_print.append({
            "study": rid,
            "pathology": p,
            "sentence": sent,
            "status": st,
            "has_image": has_img,
        })

    print("\n" + "=" * 95)
    print(f"ALL REMAINING NEGATED ROWS (Total: {len(rows_to_print)}, Studies with Image: {len([r for r in rows_to_print if r['has_image']])})")
    print("=" * 95)
    for idx, r in enumerate(rows_to_print):
        print(f"[{idx + 1:02d}] Study: {r['study']} | Pathology: {r['pathology']} | Status: {r['status']}")
        print(f"     Sentence: \"{r['sentence']}\"")
    print("=" * 95)
    return rows_to_print


def print_random_redacted_rows(
    reports_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n: int = 15,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Print n random rows with [REDACTED] near a stem."""
    subset = labels_df[(labels_df["has_image"]) & (labels_df["redacted_near_stem"])]
    records = subset.to_dict("records")
    rng = random.Random(seed)
    sampled = rng.sample(records, min(n, len(records)))

    rep_lookup = {r["report_id"]: r for _, r in reports_df.iterrows()}
    results = []
    print("\n" + "=" * 95)
    print(f"15 RANDOM ROWS WITH [REDACTED] NEAR STEM (Studies with image, seed={seed})")
    print("=" * 95)
    for idx, r in enumerate(sampled):
        rid = r["report_id"]
        p = r["pathology"]
        st = r["status"]
        role = r["eval_role"]
        rep = rep_lookup.get(rid, {})
        f = rep.get("findings_clean")
        imp = rep.get("impression_clean")
        _, red_sents = check_redacted_near_stem(f, imp, p)
        sent = red_sents[0] if red_sents else "(No matching sentence)"
        print(f"[{idx+1:02d}] Study: {rid} | Pathology: {p:<18} | Status: {st:<12} | Role: {role}")
        print(f"     Sentence: \"{sent}\"")
        results.append({"study": rid, "pathology": p, "status": st, "eval_role": role, "sentence": sent})
    print("=" * 95)
    return results


def print_transitions_prev_to_new(
    prev_labels_path: Path,
    new_labels_df: pd.DataFrame,
) -> None:
    """Print status and eval_role transitions between labels_v3_prev.csv and new labels.csv."""
    if not prev_labels_path.exists():
        print(f"Previous labels file {prev_labels_path} does not exist.")
        return

    prev_df = pd.read_csv(prev_labels_path)
    subset_prev = prev_df[prev_df["has_image"]]
    subset_new = new_labels_df[new_labels_df["has_image"]]

    merged = pd.merge(
        subset_prev[["report_id", "pathology", "status", "eval_role"]].rename(
            columns={"status": "old_status", "eval_role": "old_role"}
        ),
        subset_new[["report_id", "pathology", "status", "eval_role"]].rename(
            columns={"status": "new_status", "eval_role": "new_role"}
        ),
        on=["report_id", "pathology"],
    )

    # 1. Status transitions
    diff_status = merged[merged["old_status"] != merged["new_status"]]
    print("\n" + "=" * 95)
    print("STATUS TRANSITIONS (labels_v3_prev.csv -> new labels.csv, studies with image)")
    print("=" * 95)
    if diff_status.empty:
        print("Empty DataFrame (0 status transitions; all statuses identical between prev and new file)")
        empty_table = pd.DataFrame(columns=["pathology", "old_status", "new_status", "count"])
        print(empty_table.to_string())
    else:
        table_st = diff_status.groupby(["pathology", "old_status", "new_status"]).size().reset_index(name="count")
        print(table_st.to_string())

    # 2. Eval_role transitions
    diff_role = merged[merged["old_role"] != merged["new_role"]]
    print("\n" + "=" * 95)
    print("EVAL_ROLE TRANSITIONS (labels_v3_prev.csv -> new labels.csv, studies with image)")
    print("=" * 95)
    table_role = diff_role.groupby(["pathology", "old_role", "new_role"]).size().reset_index(name="count")
    print(table_role.to_string())

    # Transitions with count < 5
    under_5 = table_role[table_role["count"] < 5]
    print("\n" + "-" * 95)
    print("TRANSITIONS WITH COUNT UNDER 5 (STUDY IDs LISTED)")
    print("-" * 95)
    for _, r in under_5.iterrows():
        p = r["pathology"]
        o = r["old_role"]
        n = r["new_role"]
        matching_ids = diff_role[
            (diff_role["pathology"] == p) & (diff_role["old_role"] == o) & (diff_role["new_role"] == n)
        ]["report_id"].tolist()
        print(f"  {p:<18} ({o} -> {n}, count={r['count']}): {matching_ids}")
    print("=" * 95)


def print_random_pos_sentences(
    reports_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    n: int = 20,
    seed: int = 0,
) -> None:
    """Print n random 'pos' studies showing sentences containing the pathology stem."""
    pos_rows = labels_df[(labels_df["has_image"]) & (labels_df["status"] == "pos")].to_dict("records")
    rng = random.Random(seed)
    sampled = rng.sample(pos_rows, min(n, len(pos_rows)))

    rep_lookup = {r["report_id"]: r for _, r in reports_df.iterrows()}

    print("\n" + "=" * 95)
    print(f"20 RANDOM 'POS' STUDIES SHOWING STEM SENTENCES (Seed = {seed})")
    print("=" * 95)

    for idx, row in enumerate(sampled):
        rid = row["report_id"]
        path = row["pathology"]
        rep = rep_lookup.get(rid, {})

        f = rep.get("findings_clean")
        imp = rep.get("impression_clean")
        f_str = "" if not f or pd.isna(f) else str(f).strip()
        imp_str = "" if not imp or pd.isna(imp) else str(imp).strip()
        all_text = " ".join([s for s in [f_str, imp_str] if s])
        sentences = get_sentences(all_text)

        stems = list(PATHOLOGY_STEMS.get(path, []))
        if path == "Effusion" and "effusion" not in stems:
            stems.append("effusion")
        if path == "Pleural_Thickening" and "thickening" not in stems:
            stems.append("thickening")

        matching_sents = [s for s in sentences if any(st in s.lower() for st in stems)]

        print(f"\n[{idx + 1:02d}] Study: {rid} | Pathology: {path}")
        print(f"     Major terms: {rep.get('major_terms')}")
        if matching_sents:
            for s in matching_sents[:2]:
                print(f"     Stem Sentence: \"{s}\"")
        else:
            print(f"     (No explicit stem match found in findings/impression; stems={stems})")
            if f_str:
                print(f"     Findings: \"{f_str[:120]}...\"")
            if imp_str:
                print(f"     Impression: \"{imp_str[:120]}...\"")


def run_pipeline() -> None:
    """Execute complete label derivation, save labels.csv, and print all reports."""
    from eval.iu_loader import load_reports_csv

    reports_path = Path("data/iu_xray/index/reports.csv")
    labels_path = Path("data/iu_xray/index/labels.csv")
    labels_prev_path = Path("data/iu_xray/index/labels_v3_prev.csv")

    print(f"Loading reports from {reports_path}...")
    df_reports = load_reports_csv(reports_path)

    print(f"Deriving dataset labels with redacted_near_stem and updated eval_role...")
    df_labels = derive_dataset_labels(df_reports)
    df_labels.to_csv(labels_path, index=False)
    print(f"Saved {len(df_labels)} records to {labels_path}")

    # 1. Print all remaining negated rows
    print_all_remaining_negated_rows(df_reports, df_labels)

    # 2. Print evaluation role table
    print_eval_role_table(df_labels)

    # 3. Print 15 random rows with [REDACTED] near a stem
    print_random_redacted_rows(df_reports, df_labels, n=15, seed=42)

    # 4. Print status and eval_role transitions from labels_v3_prev.csv
    print_transitions_prev_to_new(labels_prev_path, df_labels)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    run_pipeline()


