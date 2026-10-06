"""Tests for deterministic template report generator in app/generate.py."""

import re
from app.config import get_thresholds
from app.generate import template_report
from app.rules import findings_from_scores
from app.schemas import Finding, PatientContext

DEFAULT_WEIGHTS = "densenet121-res224-all"
PRESENT_THRESHOLD, ABSENT_THRESHOLD = get_thresholds(DEFAULT_WEIGHTS)


ALL_18_PATHOLOGIES = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
    "Emphysema", "Fibrosis", "Effusion", "Pneumonia", "Pleural_Thickening",
    "Cardiomegaly", "Nodule", "Mass", "Hernia", "Lung Lesion",
    "Fracture", "Lung Opacity", "Enlarged Cardiomediastinum",
]


def test_all_absent_case() -> None:
    """When all findings are absent, output clean negative findings and impression."""
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.1)
    findings = [
        Finding(name="Cardiomegaly", probability=absent_score, status="absent"),
        Finding(name="Pneumothorax", probability=absent_score, status="absent"),
        Finding(name="Effusion", probability=absent_score, status="absent"),
    ]
    report = template_report(findings)

    assert report.generated_by == "template"
    assert report.cited_findings == []
    expected_imp = (
        "No findings detected by the model among the 3 pathologies it evaluates. "
        "This does not exclude abnormalities outside the model's scope."
    )
    assert expected_imp in report.impression_text
    assert "No acute cardiopulmonary abnormality" not in report.impression_text
    assert "not detected by the model:" in report.findings_text.lower()
    assert "cardiomegaly" in report.findings_text.lower()
    assert "pneumothorax" in report.findings_text.lower()
    assert "effusion" in report.findings_text.lower()


def test_multi_finding_case() -> None:
    """When multiple findings are present, all appear in findings, impression, and cited list."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.15)
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.15)
    findings = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Effusion", probability=present_score, status="present"),
        Finding(name="Pneumothorax", probability=absent_score, status="absent"),
    ]
    report = template_report(findings)

    assert report.generated_by == "template"
    # Both present findings cited
    assert "Cardiomegaly" in report.cited_findings
    assert "Effusion" in report.cited_findings
    assert "Pneumothorax" not in report.cited_findings

    # Findings text describes both
    assert "Cardiomegaly: Present" in report.findings_text
    assert "Effusion: Present" in report.findings_text

    # Impression lists both plainly
    impression_lines = [line.strip() for line in report.impression_text.splitlines() if line.strip()]
    assert "Cardiomegaly." in impression_lines
    assert "Effusion." in impression_lines

    # Consistency rule: must NOT say 'No acute findings'
    assert "No acute cardiopulmonary abnormality" not in report.impression_text
    assert "No acute findings" not in report.findings_text


def test_uncertain_never_rendered_as_present() -> None:
    """An uncertain finding must be explicitly hedged and never worded as present."""
    uncertain_score = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.1)
    findings = [
        Finding(name="Pneumonia", probability=uncertain_score, status="uncertain"),
        Finding(name="Pneumothorax", probability=absent_score, status="absent"),
    ]
    report = template_report(findings)

    # Hedged in findings text
    assert "Possible pneumonia" in report.findings_text
    assert "Pneumonia: Present" not in report.findings_text

    # Hedged in impression text plainly as 'Possible Pneumonia.'
    impression_lines = [l.strip() for l in report.impression_text.splitlines() if l.strip()]
    assert "Possible Pneumonia." in impression_lines
    # Must NOT appear as plain present finding without 'Possible'
    assert "Pneumonia." not in impression_lines

    # Cited findings should include the uncertain finding
    assert "Pneumonia" in report.cited_findings


def test_mixed_present_and_uncertain_findings() -> None:
    """Report with both present and uncertain findings renders each strictly according to status."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.15)
    uncertain_score = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.15)
    findings = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Infiltration", probability=uncertain_score, status="uncertain"),
        Finding(name="Pneumothorax", probability=absent_score, status="absent"),
    ]
    report = template_report(findings)

    # Present is affirmative
    assert "Cardiomegaly: Present" in report.findings_text
    assert "Cardiomegaly." in report.impression_text

    # Uncertain is hedged, never present
    assert "Possible infiltration" in report.findings_text
    assert "Possible Infiltration." in report.impression_text
    assert "Infiltration: Present" not in report.findings_text

    # Never output 'No acute findings' when finding is present
    assert "No acute cardiopulmonary abnormality" not in report.impression_text
    assert "No acute findings" not in report.impression_text


def test_patient_context_integrated_when_provided() -> None:
    """Clinical indication from PatientContext is included in the findings narrative."""
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.1)
    findings = [
        Finding(name="Cardiomegaly", probability=absent_score, status="absent"),
    ]
    patient = PatientContext(age=68, sex="Male", clinical_note="Shortness of breath on exertion")
    report = template_report(findings, patient=patient)

    assert "INDICATION: Shortness of breath on exertion" in report.findings_text
    expected_imp = (
        "No findings detected by the model among the 1 pathologies it evaluates. "
        "This does not exclude abnormalities outside the model's scope."
    )
    assert expected_imp in report.impression_text
    assert "No acute cardiopulmonary abnormality" not in report.impression_text


# ---------------------------------------------------------------------------
# Specific New Tests: 5(a), 5(b), 5(c) and Cap Tests
# ---------------------------------------------------------------------------

def test_report_never_states_absence_unless_status_is_absent() -> None:
    """Test 5(a): A report never states absence of a pathology unless its status is 'absent'."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.15)
    uncertain_score = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.15)

    findings = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Pneumonia", probability=uncertain_score, status="uncertain"),
        Finding(name="Effusion", probability=absent_score, status="absent"),
    ]
    report = template_report(findings)

    # Locate any negative statement in findings_text
    neg_match = re.search(r"not detected by the model:? ([^\.\n]+)", report.findings_text, re.IGNORECASE)
    assert neg_match is not None, "Negative sentence expected when absent findings exist"
    absent_section = neg_match.group(1).lower()

    # The negative section must state Effusion (which is absent)
    assert "effusion" in absent_section

    # The negative section must NEVER state Cardiomegaly (present) or Pneumonia (uncertain)
    assert "cardiomegaly" not in absent_section
    assert "pneumonia" not in absent_section

    # When NO findings are absent, verify no negative sentence is generated
    findings_no_absent = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Pneumonia", probability=uncertain_score, status="uncertain"),
    ]
    report_no_absent = template_report(findings_no_absent)
    assert "not detected by the model" not in report_no_absent.findings_text.lower()


def test_every_pathology_named_in_report_exists_in_findings_list() -> None:
    """Test 5(b): Every pathology named anywhere in report text exists in the findings list."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.15)
    absent_score = max(0.0, ABSENT_THRESHOLD - 0.15)

    # Only 2 pathologies provided in input findings
    findings = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Effusion", probability=absent_score, status="absent"),
    ]
    report = template_report(findings)
    full_report_text = f"{report.findings_text}\n{report.impression_text}".lower()

    input_pathology_names = {f.name.lower().replace("_", " ") for f in findings}

    # Verify that no unlisted pathology from the 18 benchmark pathologies is named
    for pathology in ALL_18_PATHOLOGIES:
        clean_pathology = pathology.lower().replace("_", " ")
        if clean_pathology in input_pathology_names:
            assert clean_pathology in full_report_text
        else:
            assert clean_pathology not in full_report_text, (
                f"Pathology '{clean_pathology}' appeared in report text but was not in the input findings list!"
            )


def test_impression_contains_at_most_two_uncertain_items() -> None:
    """Test 5(c): The Impression contains at most 2 uncertain items, selected by highest score."""
    present_score = min(1.0, PRESENT_THRESHOLD + 0.15)
    base_uncertain = (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2

    # 1 present finding and 5 uncertain findings with distinct scores
    findings = [
        Finding(name="Cardiomegaly", probability=present_score, status="present"),
        Finding(name="Infiltration", probability=base_uncertain + 0.04, status="uncertain"),  # Rank 1 uncertain
        Finding(name="Atelectasis", probability=base_uncertain + 0.03, status="uncertain"),   # Rank 2 uncertain
        Finding(name="Effusion", probability=base_uncertain + 0.02, status="uncertain"),      # Rank 3 uncertain
        Finding(name="Pneumonia", probability=base_uncertain + 0.01, status="uncertain"),     # Rank 4 uncertain
        Finding(name="Nodule", probability=base_uncertain, status="uncertain"),               # Rank 5 uncertain
    ]
    report = template_report(findings)

    impression_lines = [line.strip() for line in report.impression_text.splitlines() if line.strip()]
    uncertain_lines = [line for line in impression_lines if line.startswith("Possible")]

    # Requirement: at most 2 uncertain items in Impression
    assert len(uncertain_lines) <= 2
    assert len(uncertain_lines) == 2

    # Specifically, the two highest-scoring uncertain findings must be included
    assert "Possible Infiltration." in uncertain_lines
    assert "Possible Atelectasis." in uncertain_lines

    # The lower-ranked uncertain findings must NOT appear in the Impression
    assert "Possible Effusion." not in impression_lines
    assert "Possible Pneumonia." not in impression_lines
    assert "Possible Nodule." not in impression_lines

    # Verify that in Findings text, the remaining uncertain findings are grouped in one sentence
    assert "Additional equivocal findings that cannot be excluded: effusion, pneumonia, nodule." in report.findings_text


def test_impression_capped_at_five_present_findings() -> None:
    """Test that Impression is capped at 5 present findings and notes additional findings are listed above."""
    # Create 7 present findings and 3 uncertain findings
    present_names = [
        "Cardiomegaly", "Infiltration", "Effusion", "Atelectasis",
        "Pneumothorax", "Fibrosis", "Emphysema",
    ]
    findings = [
        Finding(name=name, probability=0.95 - (i * 0.03), status="present")
        for i, name in enumerate(present_names)
    ]
    # Add 2 uncertain findings
    findings.append(Finding(name="Pneumonia", probability=0.55, status="uncertain"))
    findings.append(Finding(name="Mass", probability=0.52, status="uncertain"))

    report = template_report(findings)
    impression_lines = [line.strip() for line in report.impression_text.splitlines() if line.strip()]

    # Extract present lines (plain names ending in '.', excluding 'Possible' and 'Additional findings...')
    present_lines_in_imp = [
        line for line in impression_lines
        if not line.startswith("Possible") and not line.startswith("Additional findings")
    ]

    # Impression must be capped at exactly 5 present findings
    assert len(present_lines_in_imp) == 5

    # Top 5 highest scoring present findings must be in Impression
    for name in present_names[:5]:
        assert f"{name}." in present_lines_in_imp

    # The 6th and 7th present findings must NOT be in Impression
    assert "Fibrosis." not in present_lines_in_imp
    assert "Emphysema." not in present_lines_in_imp

    # Sentence saying additional findings are listed above must be present
    assert "Additional findings are listed above." in impression_lines

    # In Findings text, all 7 present findings must be listed
    for name in present_names:
        assert f"- {name}: Present" in report.findings_text


def test_report_never_mentions_skipped_nan_pathology() -> None:
    """Pathologies skipped due to NaN scores must never be mentioned in report text or citations."""
    scores = {
        "Cardiomegaly": PRESENT_THRESHOLD + 0.15,     # Present
        "Pneumonia": (PRESENT_THRESHOLD + ABSENT_THRESHOLD) / 2,  # Uncertain
        "Effusion": ABSENT_THRESHOLD - 0.15,          # Absent
        "Lung Lesion": float("nan"),                  # Skipped
        "Enlarged Cardiomediastinum": float("nan"),  # Skipped
    }
    findings = findings_from_scores(scores, weights=DEFAULT_WEIGHTS)
    report = template_report(findings)

    # Valid findings are appropriately reported
    assert "Cardiomegaly" in report.findings_text
    assert "Cardiomegaly." in report.impression_text
    assert "Cardiomegaly" in report.cited_findings
    assert "possible pneumonia" in report.findings_text.lower()
    assert "not detected by the model: effusion." in report.findings_text.lower()

    # The skipped NaN pathologies must NOT appear anywhere in findings_text, impression_text, or citations
    findings_lower = report.findings_text.lower()
    impression_lower = report.impression_text.lower()
    cited_lower = [c.lower() for c in report.cited_findings]

    for skipped in ["lung lesion", "enlarged cardiomediastinum"]:
        assert skipped not in findings_lower
        assert skipped not in impression_lower
        assert skipped not in cited_lower

