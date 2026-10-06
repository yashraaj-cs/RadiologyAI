"""Tests for rule engine in app/rules.py: threshold boundaries and consistency rules."""

import pytest
from app.config import get_thresholds
from app.rules import (
    findings_from_scores,
    rule_allow_no_acute_findings,
    rule_format_present_finding,
    rule_format_uncertain_finding,
    rule_nodule_mass_lung_lesion_consistency,
    rule_threshold_status,
    rule_uncertain_is_never_present,
)
from app.schemas import Finding

DEFAULT_WEIGHTS = "densenet121-res224-all"
PRESENT_THRESHOLD, ABSENT_THRESHOLD = get_thresholds(DEFAULT_WEIGHTS)


# ---------------------------------------------------------------------------
# Threshold boundary tests (parameterized for both weights)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("weights", ["densenet121-res224-all", "resnet50-res512-all"])
def test_threshold_boundary_present_exactly_at_threshold(weights: str) -> None:
    """Score exactly at present_threshold must be categorized as 'present'."""
    p_th, _ = get_thresholds(weights)
    status = rule_threshold_status(p_th, weights=weights)
    assert status == "present"


@pytest.mark.parametrize("weights", ["densenet121-res224-all", "resnet50-res512-all"])
def test_threshold_boundary_present_above_threshold(weights: str) -> None:
    """Score strictly greater than present_threshold must be 'present'."""
    p_th, _ = get_thresholds(weights)
    above_score = min(1.0, p_th + 0.1)
    assert rule_threshold_status(above_score, weights=weights) == "present"
    assert rule_threshold_status(1.0, weights=weights) == "present"


@pytest.mark.parametrize("weights", ["densenet121-res224-all", "resnet50-res512-all"])
def test_threshold_boundary_uncertain_exactly_at_absent_threshold(weights: str) -> None:
    """Score exactly at absent_threshold must be 'uncertain'."""
    _, a_th = get_thresholds(weights)
    status = rule_threshold_status(a_th, weights=weights)
    assert status == "uncertain"


@pytest.mark.parametrize("weights", ["densenet121-res224-all", "resnet50-res512-all"])
def test_threshold_boundary_uncertain_just_below_present(weights: str) -> None:
    """Score just below present_threshold must be 'uncertain'."""
    p_th, _ = get_thresholds(weights)
    status = rule_threshold_status(p_th - 0.0001, weights=weights)
    assert status == "uncertain"


@pytest.mark.parametrize("weights", ["densenet121-res224-all", "resnet50-res512-all"])
def test_threshold_boundary_absent_below_absent_threshold(weights: str) -> None:
    """Score strictly below absent_threshold must be 'absent'."""
    _, a_th = get_thresholds(weights)
    below_score = max(0.0, a_th - 0.01)
    assert rule_threshold_status(below_score, weights=weights) == "absent"
    assert rule_threshold_status(0.0, weights=weights) == "absent"


def test_constant_half_score_for_supported_pathology_gives_uncertain() -> None:
    """A constant 0.5 score for a supported pathology (under DenseNet weights) yields 'uncertain'."""
    p_th, a_th = get_thresholds("densenet121-res224-all")
    assert a_th <= 0.5 < p_th
    status = rule_threshold_status(0.5, weights="densenet121-res224-all")
    assert status == "uncertain"

    findings = findings_from_scores({"Cardiomegaly": 0.5}, weights="densenet121-res224-all")
    assert len(findings) == 1
    assert findings[0].status == "uncertain"


# ---------------------------------------------------------------------------
# Consistency rule tests
# ---------------------------------------------------------------------------

def test_consistency_rule_no_acute_findings_disallowed_when_present() -> None:
    """Never permit 'No acute findings' if any finding is present."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.1)
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.1)
    findings = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Effusion", probability=absent_score, status="absent"),
    ]
    assert rule_allow_no_acute_findings(findings) is False


def test_consistency_rule_no_acute_findings_disallowed_when_uncertain() -> None:
    """Do not permit unqualified 'No acute findings' if an uncertain finding is active."""
    uncertain_score = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.1)
    findings = [
        Finding(name="Pneumothorax", probability=uncertain_score, status="uncertain"),
        Finding(name="Nodule", probability=absent_score, status="absent"),
    ]
    assert rule_allow_no_acute_findings(findings) is False


def test_consistency_rule_no_acute_findings_allowed_when_all_absent() -> None:
    """Permit 'No acute findings' when every finding is absent."""
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.1)
    findings = [
        Finding(name="Cardiomegaly", probability=absent_score, status="absent"),
        Finding(name="Pneumonia", probability=absent_score, status="absent"),
        Finding(name="Effusion", probability=absent_score, status="absent"),
    ]
    assert rule_allow_no_acute_findings(findings) is True


def test_consistency_rule_uncertain_never_present_validation() -> None:
    """An uncertain finding must never be labeled or escalated to 'present'."""
    uncertain_score = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    valid_uncertain = Finding(name="Effusion", probability=uncertain_score, status="uncertain")
    # Passing valid uncertain through validator passes cleanly
    assert rule_uncertain_is_never_present(valid_uncertain, weights=DEFAULT_WEIGHTS).status == "uncertain"

    # Attempting to assign 'present' to a score in the uncertain range (< PRESENT_THRESHOLD) must fail
    with pytest.raises(ValueError, match="cannot be labeled 'present'"):
        invalid_finding = Finding(name="Effusion", probability=uncertain_score, status="present")
        rule_uncertain_is_never_present(invalid_finding, weights=DEFAULT_WEIGHTS)


def test_consistency_rule_uncertain_formatting_hedged() -> None:
    """Uncertain pathologies must be worded as uncertain, never present."""
    hedged = rule_format_uncertain_finding("Pleural_Thickening")
    assert "Possible" in hedged
    assert "pleural thickening" in hedged

    present = rule_format_present_finding("Pleural_Thickening")
    assert present == "Pleural Thickening"


# ---------------------------------------------------------------------------
# findings_from_scores end-to-end mapping test
# ---------------------------------------------------------------------------

def test_findings_from_scores_mapping() -> None:
    """Ensure dictionary of scores maps into correctly categorized Finding objects."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.15)
    uncertain_score = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.15)

    scores = {
        "Cardiomegaly": present_score,
        "Pneumonia": uncertain_score,
        "Pneumothorax": absent_score,
    }
    findings = findings_from_scores(scores, weights=DEFAULT_WEIGHTS)
    lookup = {f.name: f for f in findings}

    assert lookup["Cardiomegaly"].status == "present"
    assert lookup["Cardiomegaly"].probability == present_score

    assert lookup["Pneumonia"].status == "uncertain"
    assert lookup["Pneumonia"].probability == uncertain_score

    assert lookup["Pneumothorax"].status == "absent"
    assert lookup["Pneumothorax"].probability == absent_score


# ---------------------------------------------------------------------------
# Nodule/Mass vs Lung Lesion consistency rule tests
# ---------------------------------------------------------------------------

def test_nodule_present_converts_lung_lesion_absent_to_uncertain() -> None:
    """If Nodule is present and Lung Lesion is absent, Lung Lesion becomes uncertain."""
    findings = [
        Finding(name="Nodule", probability=PRESENT_THRESHOLD + 0.1, status="present"),
        Finding(name="Lung Lesion", probability=ABSENT_THRESHOLD - 0.1, status="absent"),
    ]
    updated = rule_nodule_mass_lung_lesion_consistency(findings)
    lookup = {f.name: f for f in updated}
    assert lookup["Nodule"].status == "present"
    assert lookup["Lung Lesion"].status == "uncertain"


def test_mass_present_converts_lung_lesion_absent_to_uncertain() -> None:
    """If Mass is present and Lung Lesion is absent, Lung Lesion becomes uncertain."""
    findings = [
        Finding(name="Mass", probability=PRESENT_THRESHOLD + 0.1, status="present"),
        Finding(name="Lung Lesion", probability=ABSENT_THRESHOLD - 0.1, status="absent"),
    ]
    updated = rule_nodule_mass_lung_lesion_consistency(findings)
    lookup = {f.name: f for f in updated}
    assert lookup["Mass"].status == "present"
    assert lookup["Lung Lesion"].status == "uncertain"


def test_neither_nodule_nor_mass_present_leaves_lung_lesion_absent() -> None:
    """If neither Nodule nor Mass is present, Lung Lesion remains absent."""
    findings = [
        Finding(name="Nodule", probability=ABSENT_THRESHOLD - 0.1, status="absent"),
        Finding(name="Mass", probability=ABSENT_THRESHOLD - 0.1, status="absent"),
        Finding(name="Lung Lesion", probability=ABSENT_THRESHOLD - 0.1, status="absent"),
    ]
    updated = rule_nodule_mass_lung_lesion_consistency(findings)
    lookup = {f.name: f for f in updated}
    assert lookup["Lung Lesion"].status == "absent"


def test_nodule_mass_missing_from_list_no_error() -> None:
    """When Nodule and Mass are absent from the list, rule handles gracefully without error."""
    findings = [
        Finding(name="Cardiomegaly", probability=PRESENT_THRESHOLD + 0.1, status="present"),
        Finding(name="Lung Lesion", probability=ABSENT_THRESHOLD - 0.1, status="absent"),
    ]
    updated = rule_nodule_mass_lung_lesion_consistency(findings)
    lookup = {f.name: f for f in updated}
    assert lookup["Lung Lesion"].status == "absent"


def test_lung_lesion_missing_from_list_no_error() -> None:
    """When Lung Lesion is absent from the list, rule runs without error."""
    findings = [
        Finding(name="Nodule", probability=PRESENT_THRESHOLD + 0.1, status="present"),
        Finding(name="Cardiomegaly", probability=PRESENT_THRESHOLD + 0.1, status="present"),
    ]
    updated = rule_nodule_mass_lung_lesion_consistency(findings)
    lookup = {f.name: f for f in updated}
    assert lookup["Nodule"].status == "present"


def test_findings_from_scores_applies_nodule_lung_lesion_rule() -> None:
    """End-to-end: findings_from_scores adjusts Lung Lesion if Nodule is present."""
    scores = {
        "Nodule": PRESENT_THRESHOLD + 0.1,       # present
        "Lung Lesion": ABSENT_THRESHOLD - 0.1,   # score in absent range
        "Cardiomegaly": ABSENT_THRESHOLD - 0.1,  # absent
    }
    findings = findings_from_scores(scores, weights=DEFAULT_WEIGHTS)
    lookup = {f.name: f for f in findings}
    assert lookup["Nodule"].status == "present"
    assert lookup["Lung Lesion"].status == "uncertain"
    assert lookup["Cardiomegaly"].status == "absent"


# ---------------------------------------------------------------------------
# NaN score safety tests
# ---------------------------------------------------------------------------

def test_nan_score_creates_no_finding_and_does_not_crash(caplog: pytest.LogCaptureFixture) -> None:
    """A NaN score (e.g., from resnet50 op_threshs) creates no Finding, doesn't crash, and logs a warning."""
    import logging

    scores = {
        "Cardiomegaly": PRESENT_THRESHOLD + 0.1,
        "Lung Lesion": float("nan"),
        "Enlarged Cardiomediastinum": float("nan"),
        "Effusion": ABSENT_THRESHOLD - 0.1,
    }

    with caplog.at_level(logging.WARNING):
        findings = findings_from_scores(scores, weights=DEFAULT_WEIGHTS)

    # Must create exactly 2 findings for the valid numerical scores
    assert len(findings) == 2
    finding_names = [f.name for f in findings]
    assert "Cardiomegaly" in finding_names
    assert "Effusion" in finding_names

    # The NaN pathologies must have NO finding created
    assert "Lung Lesion" not in finding_names
    assert "Enlarged Cardiomediastinum" not in finding_names

    # Verify that skipped pathologies were explicitly logged
    log_text = caplog.text
    assert "Skipping pathology 'Lung Lesion' with NaN model score." in log_text
    assert "Skipping pathology 'Enlarged Cardiomediastinum' with NaN model score." in log_text


def test_rule_threshold_status_raises_on_nan() -> None:
    """rule_threshold_status raises ValueError when given NaN."""
    with pytest.raises(ValueError, match="Cannot determine threshold status for NaN score"):
        rule_threshold_status(float("nan"), weights=DEFAULT_WEIGHTS)


def test_resnet50_unsupported_pathologies_never_appear_in_findings_or_report() -> None:
    """Test 4(a): For resnet50-res512-all, unsupported pathologies never appear in findings or report text."""
    from pathlib import Path
    import torchxrayvision as xrv
    from app.generate import template_report
    from app.vision import load_model, predict

    weights_name = "resnet50-res512-all"
    sample_img = Path("data/sample/00000001_000.png")

    if not sample_img.is_file():
        pytest.skip(f"Sample image '{sample_img}' is missing.")

    weights_url = xrv.models.model_urls.get(weights_name, {}).get("weights_url", "")
    weights_filename = Path(weights_url).name if weights_url else ""
    weights_path = Path(xrv.utils.get_cache_dir()) / weights_filename if weights_filename else None

    if weights_path is None or not weights_path.is_file():
        pytest.skip(f"Pretrained weights file '{weights_path}' is missing for '{weights_name}'.")

    model = load_model(weights_name)
    assert "Lung Lesion" in model.unsupported_pathologies
    assert "Enlarged Cardiomediastinum" in model.unsupported_pathologies

    # Predict on sample image
    scores = predict(str(sample_img), model=model)
    assert "Lung Lesion" not in scores
    assert "Enlarged Cardiomediastinum" not in scores

    # Generate findings from scores
    findings = findings_from_scores(scores, weights=weights_name)
    finding_names = [f.name for f in findings]
    assert "Lung Lesion" not in finding_names
    assert "Enlarged Cardiomediastinum" not in finding_names

    # Generate template report
    report = template_report(findings)
    findings_lower = report.findings_text.lower()
    impression_lower = report.impression_text.lower()
    cited_lower = [c.lower() for c in report.cited_findings]

    for unsupported in ["lung lesion", "enlarged cardiomediastinum"]:
        assert unsupported not in findings_lower
        assert unsupported not in impression_lower
        assert unsupported not in cited_lower


def test_findings_from_scores_score_055_densenet_uncertain_resnet_present() -> None:
    """findings_from_scores on a score of 0.55 gives 'uncertain' under densenet121-res224-all and 'present' under resnet50-res512-all."""
    scores = {"Cardiomegaly": 0.55}

    dense_findings = findings_from_scores(scores, weights="densenet121-res224-all")
    assert len(dense_findings) == 1
    assert dense_findings[0].name == "Cardiomegaly"
    assert dense_findings[0].status == "uncertain"

    resnet_findings = findings_from_scores(scores, weights="resnet50-res512-all")
    assert len(resnet_findings) == 1
    assert resnet_findings[0].name == "Cardiomegaly"
    assert resnet_findings[0].status == "present"


def test_rules_functions_require_weights() -> None:
    """Verify that rule_threshold_status, rule_uncertain_is_never_present, and findings_from_scores require weights."""
    with pytest.raises(TypeError):
        rule_threshold_status(0.5)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        finding = Finding(name="Cardiomegaly", probability=0.5, status="uncertain")
        rule_uncertain_is_never_present(finding)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        findings_from_scores({"Cardiomegaly": 0.5})  # type: ignore[call-arg]


def test_get_thresholds_unknown_weights_raises_value_error() -> None:
    """get_thresholds('unknown-weights') must raise ValueError listing valid options."""
    with pytest.raises(ValueError, match="Unknown weights 'unknown-weights'"):
        get_thresholds("unknown-weights")


