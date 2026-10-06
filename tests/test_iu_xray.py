"""Unit tests for IU X-Ray report parser and view symmetry detection.

CRITICAL: Tests must NOT read from data/ directory.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from eval.iu_loader import clean_token_xxxx, parse_report_xml
from eval.labels import (
    ALL_PATHOLOGIES,
    MAPPED_PATHOLOGIES,
    PATHOLOGY_STEMS,
    check_redacted_near_stem,
    check_text_flags,
    derive_eval_role,
    derive_pathology_status,
    derive_study_labels,
    is_sentence_negated,
    match_term,
    match_term_for_pathology,
)
from scripts.iu_view_detect import compute_symmetry


def test_parse_report_xml_synthetic_with_trailing_whitespace():
    """Verify XML parser extracts sections, strips trailing whitespace from MeSH terms, and cleans XXXX."""
    synthetic_xml = """<?xml version="1.0" encoding="utf-8"?>
<eCitation>
   <uId id="CXR_TEST_42"/>
   <MedlineCitation>
      <Article>
         <Abstract>
            <AbstractText Label="COMPARISON">Chest radiographs dated XXXX.</AbstractText>
            <AbstractText Label="INDICATION">Cough and fever XXXX</AbstractText>
            <AbstractText Label="FINDINGS">The lungs are clear. No XXXX effusion or pneumothorax.</AbstractText>
            <AbstractText Label="IMPRESSION">Normal chest x-XXXX.</AbstractText>
         </Abstract>
      </Article>
   </MedlineCitation>
   <MeSH>
      <major>Calcified Granuloma/lung/upper lobe   \t \n</major>
      <major>Cardiomegaly   </major>
      <automatic>cardiomegaly  \t</automatic>
   </MeSH>
   <parentImage id="CXR_TEST_42_IM-0001-1001"/>
   <parentImage id="CXR_TEST_42_IM-0001-2001"/>
</eCitation>
"""
    result = parse_report_xml(synthetic_xml)

    # Report ID
    assert result["report_id"] == "CXR_TEST_42"

    # Raw text preserved
    assert result["indication"] == "Cough and fever XXXX"
    assert result["comparison"] == "Chest radiographs dated XXXX."
    assert result["findings"] == "The lungs are clear. No XXXX effusion or pneumothorax."
    assert result["impression"] == "Normal chest x-XXXX."

    # Cleaned text (XXXX replaced with [REDACTED])
    assert result["indication_clean"] == "Cough and fever [REDACTED]"
    assert result["comparison_clean"] == "Chest radiographs dated [REDACTED]."
    assert result["findings_clean"] == "The lungs are clear. No [REDACTED] effusion or pneumothorax."
    assert result["impression_clean"] == "Normal chest x-[REDACTED]."
    assert result["clean_findings"] == result["findings_clean"]
    assert result["clean_impression"] == result["impression_clean"]

    # MeSH terms with trailing whitespace stripped
    assert result["major_terms"] == ["Calcified Granuloma/lung/upper lobe", "Cardiomegaly"]
    assert result["auto_terms"] == ["cardiomegaly"]

    # Image IDs
    assert result["image_ids"] == ["CXR_TEST_42_IM-0001-1001", "CXR_TEST_42_IM-0001-2001"]

    # Flags
    assert result["has_image"] is True
    assert result["has_findings"] is True
    assert result["has_impression"] is True


def test_parse_report_xml_empty_sections():
    """Verify XML parser handles empty/missing sections gracefully."""
    synthetic_xml = """<?xml version="1.0" encoding="utf-8"?>
<eCitation>
   <uId id="CXR_EMPTY_01"/>
   <MedlineCitation>
      <Article>
         <Abstract>
            <AbstractText Label="COMPARISON"/>
            <AbstractText Label="FINDINGS"/>
            <AbstractText Label="IMPRESSION"/>
         </Abstract>
      </Article>
   </MedlineCitation>
</eCitation>
"""
    result = parse_report_xml(synthetic_xml)

    assert result["report_id"] == "CXR_EMPTY_01"
    assert result["indication"] == ""
    assert result["comparison"] == ""
    assert result["findings"] == ""
    assert result["impression"] == ""
    assert result["findings_clean"] == ""
    assert result["impression_clean"] == ""
    assert result["major_terms"] == []
    assert result["auto_terms"] == []
    assert result["image_ids"] == []
    assert result["has_image"] is False
    assert result["has_findings"] is False
    assert result["has_impression"] is False


def test_clean_token_xxxx():
    """Verify [REDACTED] marker substitution and punctuation spacing cleanup."""
    assert clean_token_xxxx("") == ""
    assert clean_token_xxxx(None) == ""
    assert clean_token_xxxx("Normal chest x-XXXX.") == "Normal chest x-[REDACTED]."
    assert clean_token_xxxx("No XXXX of pneumothorax.") == "No [REDACTED] of pneumothorax."
    assert clean_token_xxxx("Dated XXXX.") == "Dated [REDACTED]."
    assert clean_token_xxxx("XXXX") == "[REDACTED]"
    assert clean_token_xxxx("XXXX XXXX") == "[REDACTED] [REDACTED]"


def test_missing_sections_empty_string_never_nan():
    """Verify missing indication and comparison sections give '' and never 'nan'."""
    xml_missing_ind_comp = """<?xml version="1.0" encoding="utf-8"?>
<eCitation>
   <uId id="CXR_TEST_MISSING_01"/>
   <MedlineCitation>
      <Article>
         <Abstract>
            <AbstractText Label="FINDINGS">Heart size normal.</AbstractText>
            <AbstractText Label="IMPRESSION">No acute findings.</AbstractText>
         </Abstract>
      </Article>
   </MedlineCitation>
</eCitation>
"""
    result = parse_report_xml(xml_missing_ind_comp)
    assert result["indication"] == ""
    assert result["indication_clean"] == ""
    assert result["comparison"] == ""
    assert result["comparison_clean"] == ""
    assert "nan" not in result["indication"]
    assert "nan" not in result["indication_clean"]
    assert "nan" not in result["comparison"]
    assert "nan" not in result["comparison_clean"]


def test_symmetry_score_mirror_symmetric():
    """Verify symmetry function returns about 1 on a mirror-symmetric synthetic image."""
    rng = np.random.RandomState(42)
    left_half = rng.uniform(0.1, 1.0, size=(128, 64))
    right_half = np.fliplr(left_half)
    symmetric_arr = np.hstack([left_half, right_half])

    score = compute_symmetry(symmetric_arr)
    assert abs(score - 1.0) < 1e-4, f"Expected symmetry ~ 1.0, got {score}"

    # Also test via PIL Image
    pil_img = Image.fromarray((symmetric_arr * 255).astype(np.uint8))
    pil_score = compute_symmetry(pil_img)
    assert abs(pil_score - 1.0) < 1e-3, f"Expected PIL symmetry ~ 1.0, got {pil_score}"


def test_symmetry_score_one_sided():
    """Verify symmetry function returns clearly lower score on a one-sided synthetic image."""
    # Flat zero on right, bright block on left
    one_sided_arr = np.zeros((128, 128), dtype=np.float64)
    one_sided_arr[:, :64] = 200.0

    score = compute_symmetry(one_sided_arr)
    # Perfectly inverted across midline => correlation is -1.0
    assert score < 0.0, f"Expected negative score for one-sided image, got {score}"

    # Partial one-sided feature
    asym = np.zeros((128, 128), dtype=np.uint8)
    asym[20:100, 10:50] = 220
    asym_pil = Image.fromarray(asym)
    asym_score = compute_symmetry(asym_pil)
    assert asym_score < 0.1, f"Expected low score for asymmetric image, got {asym_score}"


def test_pathology_stems_completeness():
    """Verify that all 18 model pathologies are present, with 16 mapped and 2 unmapped."""
    assert len(ALL_PATHOLOGIES) == 18
    assert len(PATHOLOGY_STEMS) == 18
    assert len(MAPPED_PATHOLOGIES) == 16
    assert PATHOLOGY_STEMS["Lung Lesion"] == []
    assert PATHOLOGY_STEMS["Enlarged Cardiomediastinum"] == []

    # Check key mappings from specification
    assert "atelect" in PATHOLOGY_STEMS["Atelectasis"]
    assert "cardiomegaly" in PATHOLOGY_STEMS["Cardiomegaly"]
    assert "pleural effusion" in PATHOLOGY_STEMS["Effusion"]
    assert "consolidat" in PATHOLOGY_STEMS["Consolidation"]
    assert "pneumonia" in PATHOLOGY_STEMS["Pneumonia"]
    assert "opacity" in PATHOLOGY_STEMS["Lung Opacity"]
    assert "airspace disease" in PATHOLOGY_STEMS["Lung Opacity"]


def test_match_term():
    """Verify term matching logic before the first '/'."""
    # Prefix before '/' matches
    assert match_term("Pulmonary Atelectasis/middle lobe/right", ["atelect"]) is True
    assert match_term("Cardiomegaly/mild", ["cardiomegaly"]) is True
    assert match_term("Opacity/lung/base/left", ["opacity", "airspace disease"]) is True

    # Stem only present after '/' does NOT match
    assert match_term("Thickening/pleura", ["pleural thickening"]) is False
    assert match_term("Quality/atelectasis", ["atelect"]) is False

    # Empty inputs
    assert match_term("", ["atelect"]) is False
    assert match_term("Atelectasis", []) is False


def test_derive_pathology_status():
    """Verify discrete status derivation using priority rules."""
    # 1. Matched in major terms => pos
    assert derive_pathology_status(
        major_terms=["Pulmonary Atelectasis/base/left"],
        auto_terms=["Atelectasis"],
        pathology="Atelectasis",
    ) == "pos"

    # 2. Matched only in auto terms => pos_auto_only
    assert derive_pathology_status(
        major_terms=["Cardiomegaly"],
        auto_terms=["Atelectasis"],
        pathology="Atelectasis",
    ) == "pos_auto_only"

    # 3. Not matched, but major terms are exactly ['normal'] => neg_clean
    assert derive_pathology_status(
        major_terms=["normal"],
        auto_terms=[],
        pathology="Atelectasis",
    ) == "neg_clean"
    # Case insensitivity for normal
    assert derive_pathology_status(
        major_terms=["Normal"],
        auto_terms=[],
        pathology="Atelectasis",
    ) == "neg_clean"

    # 4. Not matched, but major terms contain other non-normal findings => neg_assumed
    assert derive_pathology_status(
        major_terms=["Aorta/tortuous"],
        auto_terms=[],
        pathology="Atelectasis",
    ) == "neg_assumed"

    # 5. Empty major and auto terms => neg_assumed
    assert derive_pathology_status(
        major_terms=[],
        auto_terms=[],
        pathology="Atelectasis",
    ) == "neg_assumed"


def test_subcutaneous_emphysema_ambiguous():
    """Verify Subcutaneous Emphysema maps to ambiguous for Emphysema."""
    assert derive_pathology_status(["Subcutaneous Emphysema/thorax/right"], [], "Emphysema") == "ambiguous"
    assert derive_pathology_status([], ["subcutaneous emphysema"], "Emphysema") == "ambiguous"


def test_thickening_pleura_pleural_thickening():
    """Verify Thickening/pleura matches Pleural_Thickening in major terms."""
    assert derive_pathology_status(["Thickening/pleura/right"], [], "Pleural_Thickening") == "pos"
    assert derive_pathology_status(["Thickening/pleura"], [], "Pleural_Thickening") == "pos"
    # Unrelated thickening does not map to Pleural_Thickening
    assert derive_pathology_status(["Thickening/heart ventricles"], [], "Pleural_Thickening") == "neg_assumed"


def test_pericardial_effusion_never_maps_to_effusion():
    """Verify Pericardial Effusion never maps to Effusion in major or auto terms."""
    assert derive_pathology_status(["Pericardial Effusion"], [], "Effusion") == "neg_assumed"
    assert derive_pathology_status([], ["pericardial effusion"], "Effusion") == "neg_assumed"
    assert derive_pathology_status(["normal"], ["pericardial effusion"], "Effusion") == "neg_clean"


def test_cardiomegaly_borderline_ambiguous():
    """Verify Cardiomegaly/borderline maps to ambiguous for Cardiomegaly."""
    assert derive_pathology_status(["Cardiomegaly/borderline"], [], "Cardiomegaly") == "ambiguous"


def test_priority_order():
    """Verify status priority order:
    1. non-ambiguous major -> pos
    2. ambiguous match (major or auto) -> ambiguous
    3. auto match -> pos_auto_only
    4. normal major -> neg_clean
    5. other -> neg_assumed
    """
    # 1. Non-ambiguous major + ambiguous major -> pos (non-ambiguous major wins)
    assert derive_pathology_status(
        major_terms=["Cardiomegaly/mild", "Cardiomegaly/borderline"],
        auto_terms=[],
        pathology="Cardiomegaly",
    ) == "pos"

    # 2. Ambiguous major only -> ambiguous
    assert derive_pathology_status(
        major_terms=["Cardiomegaly/borderline"],
        auto_terms=[],
        pathology="Cardiomegaly",
    ) == "ambiguous"

    # 3. Ambiguous auto only -> ambiguous
    assert derive_pathology_status(
        major_terms=[],
        auto_terms=["mass lesion"],
        pathology="Mass",
    ) == "ambiguous"

    # 4. Non-ambiguous auto match -> pos_auto_only
    assert derive_pathology_status(
        major_terms=[],
        auto_terms=["pneumonia"],
        pathology="Pneumonia",
    ) == "pos_auto_only"
    # Generic auto effusion without pericardial -> pos_auto_only
    assert derive_pathology_status(
        major_terms=[],
        auto_terms=["effusion"],
        pathology="Effusion",
    ) == "pos_auto_only"

    # 5. Major terms exactly ['normal'] -> neg_clean
    assert derive_pathology_status(
        major_terms=["normal"],
        auto_terms=[],
        pathology="Atelectasis",
    ) == "neg_clean"

    # 6. Other unmapped terms -> neg_assumed
    assert derive_pathology_status(
        major_terms=["Aorta/tortuous"],
        auto_terms=[],
        pathology="Atelectasis",
    ) == "neg_assumed"


def test_negation_and_hedge_flags():
    """Verify negated_text and hedged_text flags on synthetic sentences."""
    # 1. Negation example
    f_neg = "There is no focal consolidation or pneumothorax."
    has_neg, has_hdg, neg_sents, _ = check_text_flags(f_neg, "", "Consolidation")
    assert has_neg is True
    assert has_hdg is False
    assert len(neg_sents) == 1

    # 2. Hedge example
    f_hdg = "Bandlike opacity in the left base may represent atelectasis."
    has_neg, has_hdg, _, hdg_sents = check_text_flags(f_hdg, "", "Atelectasis")
    assert has_neg is False
    assert has_hdg is True
    assert len(hdg_sents) == 1


def test_negation_exceptions():
    """Verify that sentences containing exceptions do NOT count as negated:
    - 'no change'
    - 'no interval change'
    - 'no significant change'
    - 'unchanged'
    - 'no other'
    - 'without significant interval change'
    - 'not seen on prior'
    """
    # 1. 'no change'
    sent_no_change = "No change in the small calcified nodule in the right lower lobe."
    assert is_sentence_negated(sent_no_change, "nodul") is False

    # 2. 'no interval change'
    sent_no_interval = "No interval change in the left pleural effusion."
    assert is_sentence_negated(sent_no_interval, "effusion") is False

    # 3. 'no significant change'
    sent_no_sig = "No significant change in cardiomegaly."
    assert is_sentence_negated(sent_no_sig, "cardiomegaly") is False

    # 4. 'unchanged'
    sent_unchanged = "The small right apical pneumothorax is unchanged."
    assert is_sentence_negated(sent_unchanged, "pneumothorax") is False

    # 5. 'no other'
    sent_no_other = "No other focal consolidation is identified."
    assert is_sentence_negated(sent_no_other, "consolidat") is False

    # 6. 'without significant interval change'
    sent_without_change = "There is redemonstration without significant interval change of mild subsegmental atelectasis of the left base."
    assert is_sentence_negated(sent_without_change, "atelect") is False

    # 7. 'not seen on prior'
    sent_not_seen_prior = "There is a 9 mm right lower lobe pulmonary nodule, not seen on prior exams."
    assert is_sentence_negated(sent_not_seen_prior, "nodul") is False


def test_negation_true_negative_still_counts():
    """Verify genuine negation cases still evaluate as negated."""
    # Prefix cue
    assert is_sentence_negated("There is no focal consolidation.", "consolidat") is True
    assert is_sentence_negated("No evidence of pneumothorax.", "pneumothorax") is True
    assert is_sentence_negated("Negative for pleural effusion.", "pleural effusion") is True
    assert is_sentence_negated("The lungs are free of focal airspace disease.", "airspace disease") is True
    assert is_sentence_negated("Without acute fracture.", "fracture") is True

    # Suffix cue
    assert is_sentence_negated("Focal infiltration is not seen.", "infiltrat") is True
    assert is_sentence_negated("Prior pulmonary edema is resolved.", "edema") is True


def test_eval_role_cases():
    """Verify every eval_role assignment case:
    - 'positive': status pos AND negated_text False AND hedged_text False AND redacted_near_stem False
    - 'excluded': status ambiguous or pos_auto_only, or pos with either flag True, or redacted_near_stem True
    - 'negative_clean': status neg_clean (and not redacted_near_stem)
    - 'negative_assumed': status neg_assumed (and not redacted_near_stem)
    """
    # 1. positive: pos + not negated + not hedged + not redacted
    assert derive_eval_role("pos", negated_text=False, hedged_text=False, redacted_near_stem=False) == "positive"

    # 2. excluded:
    # 2a. pos with negated_text=True
    assert derive_eval_role("pos", negated_text=True, hedged_text=False, redacted_near_stem=False) == "excluded"
    # 2b. pos with hedged_text=True
    assert derive_eval_role("pos", negated_text=False, hedged_text=True, redacted_near_stem=False) == "excluded"
    # 2c. pos with both flags True
    assert derive_eval_role("pos", negated_text=True, hedged_text=True, redacted_near_stem=False) == "excluded"
    # 2d. pos with redacted_near_stem=True
    assert derive_eval_role("pos", negated_text=False, hedged_text=False, redacted_near_stem=True) == "excluded"
    # 2e. ambiguous (regardless of text flags)
    assert derive_eval_role("ambiguous", negated_text=False, hedged_text=False) == "excluded"
    assert derive_eval_role("ambiguous", negated_text=True, hedged_text=True, redacted_near_stem=True) == "excluded"
    # 2f. pos_auto_only (regardless of text flags)
    assert derive_eval_role("pos_auto_only", negated_text=False, hedged_text=False) == "excluded"
    assert derive_eval_role("pos_auto_only", negated_text=True, hedged_text=True, redacted_near_stem=True) == "excluded"
    # 2g. negative_clean with redacted_near_stem=True gets excluded
    assert derive_eval_role("neg_clean", redacted_near_stem=True) == "excluded"
    # 2h. negative_assumed with redacted_near_stem=True gets excluded
    assert derive_eval_role("neg_assumed", redacted_near_stem=True) == "excluded"

    # 3. negative_clean: status neg_clean and not redacted
    assert derive_eval_role("neg_clean", redacted_near_stem=False) == "negative_clean"

    # 4. negative_assumed: status neg_assumed and not redacted
    assert derive_eval_role("neg_assumed", redacted_near_stem=False) == "negative_assumed"

    # Unknown status raises ValueError
    with pytest.raises(ValueError):
        derive_eval_role("unknown_status")


def test_redacted_near_stem_detection():
    """Verify check_redacted_near_stem detects [REDACTED] in the sentence containing a stem."""
    # Redacted in same sentence as stem
    f_red = "There is [REDACTED] opacity in the right lung base."
    has_red, matches = check_redacted_near_stem(f_red, "", "Lung Opacity")
    assert has_red is True
    assert len(matches) == 1

    # Redacted in different sentence than stem
    f_diff = "Radiograph dated [REDACTED]. The lungs demonstrate mild bibasilar atelectasis."
    has_red, matches = check_redacted_near_stem(f_diff, "", "Atelectasis")
    assert has_red is False
    assert len(matches) == 0

    # No redaction present
    f_clean = "There is mild pulmonary edema."
    has_red, matches = check_redacted_near_stem(f_clean, "", "Edema")
    assert has_red is False
    assert len(matches) == 0




