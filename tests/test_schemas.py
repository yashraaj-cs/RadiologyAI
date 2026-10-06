"""Unit tests for Pydantic data schemas in app/schemas.py."""

import pytest
from pydantic import ValidationError

from app.config import get_thresholds
from app.schemas import Finding, PatientContext, Report

PRESENT_THRESHOLD, ABSENT_THRESHOLD = get_thresholds("densenet121-res224-all")


def test_valid_finding_passes() -> None:
    """Ensure a valid Finding instance with probability in [0, 1] and valid status passes."""
    present_prob = min(1.0, PRESENT_THRESHOLD + 0.1)
    finding = Finding(name="Cardiomegaly", probability=present_prob, status="present")
    assert finding.name == "Cardiomegaly"
    assert finding.probability == present_prob
    assert finding.status == "present"

    uncertain_prob = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    uncertain_finding = Finding(name="Effusion", probability=uncertain_prob, status="uncertain")
    assert uncertain_finding.status == "uncertain"

    absent_prob = max(0.0, ABSENT_THRESHOLD - 0.1)
    absent_finding = Finding(name="Pneumothorax", probability=absent_prob, status="absent")
    assert absent_finding.status == "absent"


def test_finding_probability_greater_than_one_fails() -> None:
    """Ensure a Finding with probability > 1 raises a ValidationError."""
    with pytest.raises(ValidationError):
        Finding(name="Cardiomegaly", probability=1.05, status="present")


def test_finding_probability_less_than_zero_fails() -> None:
    """Ensure a Finding with probability < 0 raises a ValidationError."""
    with pytest.raises(ValidationError):
        Finding(name="Cardiomegaly", probability=-0.1, status="absent")


def test_finding_invalid_status_fails() -> None:
    """Ensure a Finding with an unrecognized status literal raises a ValidationError."""
    with pytest.raises(ValidationError):
        Finding(name="Cardiomegaly", probability=0.85, status="positive")  # type: ignore[arg-type]


def test_valid_report_passes() -> None:
    """Ensure valid Report instances pass validation."""
    report = Report(
        findings_text="The cardiomediastinal silhouette is enlarged. Lungs are clear.",
        impression_text="Cardiomegaly without acute pulmonary consolidation.",
        cited_findings=["Cardiomegaly"],
        sources=["standard_template_v1"],
        generated_by="template",
    )
    assert report.generated_by == "template"
    assert "Cardiomegaly" in report.cited_findings

    llm_report = Report(
        findings_text="No findings detected by the model among the 18 pathologies it evaluates.",
        impression_text="No findings detected by the model among the 18 pathologies it evaluates. This does not exclude abnormalities outside the model's scope.",
        cited_findings=[],
        sources=["guideline_cxr_normal"],
        generated_by="llm",
    )
    assert llm_report.generated_by == "llm"


def test_report_invalid_generated_by_fails() -> None:
    """Ensure a Report with an invalid generated_by field raises a ValidationError."""
    with pytest.raises(ValidationError):
        Report(
            findings_text="Normal findings.",
            impression_text="Normal impression.",
            cited_findings=[],
            sources=[],
            generated_by="human",  # type: ignore[arg-type]
        )


def test_patient_context_passes() -> None:
    """Ensure PatientContext accepts valid synthetic fields or defaults to None."""
    ctx = PatientContext(age=55, sex="Female", clinical_note="Synthetic note: shortness of breath")
    assert ctx.age == 55
    assert ctx.sex == "Female"

    empty_ctx = PatientContext()
    assert empty_ctx.age is None
    assert empty_ctx.sex is None
    assert empty_ctx.clinical_note is None
