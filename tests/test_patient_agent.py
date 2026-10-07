"""Unit tests for the patient explanation agent."""

import pytest
from app.patient_agent import (
    PATHOLOGY_CLINICAL_KNOWLEDGE,
    generate_offline_patient_explanation,
    call_gemini_patient_agent,
)
from app.schemas import Finding


def test_pathology_clinical_knowledge_keys():
    """Verify essential clinical pathologies exist in the knowledge dictionary."""
    essential_keys = ["Cardiomegaly", "Effusion", "Pneumonia", "Pneumothorax", "Consolidation", "Atelectasis"]
    for key in essential_keys:
        assert key in PATHOLOGY_CLINICAL_KNOWLEDGE
        assert "layman_name" in PATHOLOGY_CLINICAL_KNOWLEDGE[key]
        assert "plain_summary" in PATHOLOGY_CLINICAL_KNOWLEDGE[key]
        assert "doctor_questions" in PATHOLOGY_CLINICAL_KNOWLEDGE[key]


def test_generate_offline_explanation_cardiomegaly():
    """Verify offline explanation answers a specific pathology query."""
    findings = [
        Finding(name="Cardiomegaly", probability=0.82, status="present"),
        Finding(name="Pneumothorax", probability=0.04, status="absent"),
    ]
    report_text = "FINDINGS:\n- Cardiomegaly: Present.\nIMPRESSION:\nCardiomegaly."
    resp = generate_offline_patient_explanation(report_text, findings, "What does cardiomegaly mean?")

    assert "Cardiomegaly" in resp or "Enlarged Heart" in resp
    assert "Detected on your scan" in resp
    assert "questions to ask your doctor" in resp.lower()


def test_generate_offline_explanation_urgency():
    """Verify offline explanation evaluates urgency and provides reassuring advice."""
    findings = [
        Finding(name="Atelectasis", probability=0.45, status="uncertain"),
        Finding(name="Pneumothorax", probability=0.01, status="absent"),
    ]
    report_text = "FINDINGS:\n- Possible Atelectasis.\nIMPRESSION:\nPossible Atelectasis."
    resp = generate_offline_patient_explanation(report_text, findings, "Is my chest X-ray dangerous?")

    assert "Reassurance" in resp or "emergency" in resp.lower()
    assert "doctor" in resp.lower()


def test_generate_offline_general_summary():
    """Verify offline explanation generates a comprehensive plain-English summary."""
    findings = [
        Finding(name="Effusion", probability=0.74, status="present"),
        Finding(name="Infiltration", probability=0.55, status="uncertain"),
        Finding(name="Pneumothorax", probability=0.02, status="absent"),
    ]
    report_text = "FINDINGS:\n- Effusion: Present.\n- Possible Infiltration."
    resp = generate_offline_patient_explanation(report_text, findings, "Can you explain my report in plain English?")

    assert "Plain-English Summary" in resp
    assert "Effusion" in resp or "Fluid Around the Lung" in resp
    assert "Uncertain" in resp
