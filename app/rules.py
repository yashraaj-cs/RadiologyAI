"""Rule engine for mapping model scores to categorized clinical findings.

Applies configurable thresholds (present / uncertain / absent) and enforces
clinical consistency rules to prevent contradictory or falsely certain draft reports.
"""

import logging
import math
from typing import Literal
from app.config import get_thresholds
from app.schemas import Finding

logger = logging.getLogger(__name__)


def rule_threshold_status(
    score: float,
    weights: str,
) -> Literal["present", "uncertain", "absent"]:
    """Categorize a continuous pathology model score into discrete status buckets.

    Why this rule exists:
    Clinical decision support requires partitioning continuous model confidence
    scores into actionable categories: high confidence (present), equivocal confidence
    (uncertain requiring radiologist scrutiny), and low confidence (absent).

    Args:
        score: Model score between 0.0 and 1.0.
        weights: Model weights identifier to retrieve thresholds via get_thresholds.

    Returns:
        "present", "uncertain", or "absent".

    Raises:
        ValueError: If score is NaN or weights is empty.
    """
    if score is None or math.isnan(score):
        raise ValueError("Cannot determine threshold status for NaN score.")

    if not weights:
        raise ValueError("weights argument cannot be empty.")

    present_threshold, absent_threshold = get_thresholds(weights)

    if score >= present_threshold:
        return "present"
    elif score < absent_threshold:
        return "absent"
    else:
        return "uncertain"


def rule_uncertain_is_never_present(
    finding: Finding,
    weights: str,
) -> Finding:
    """Ensure that an uncertain finding is strictly retained as uncertain and never marked present.

    Why this rule exists:
    Borderline or equivocal imaging signals must remain clearly flagged as uncertain.
    Escalating an indeterminate finding to 'present' creates false positives and could
    trigger unwarranted invasive procedures or patient harm.

    Args:
        finding: Finding object to validate.
        weights: Model weights identifier to retrieve present threshold.

    Returns:
        The validated finding.

    Raises:
        ValueError: If an uncertain finding has been incorrectly assigned status 'present'.
    """
    if not weights:
        raise ValueError("weights argument cannot be empty.")

    present_threshold, _ = get_thresholds(weights)

    if finding.status == "uncertain" and finding.probability >= present_threshold:
        raise ValueError(
            f"Consistency violation: Finding '{finding.name}' with score {finding.probability:.3f} "
            f">= {present_threshold} cannot be labeled 'uncertain'."
        )
    if finding.status == "present" and finding.probability < present_threshold:
        raise ValueError(
            f"Consistency violation: Finding '{finding.name}' with score {finding.probability:.3f} "
            f"< {present_threshold} cannot be labeled 'present'."
        )
    return finding


def rule_allow_no_acute_findings(findings: list[Finding]) -> bool:
    """Determine whether a 'No acute findings' summary statement is permissible.

    Why this rule exists:
    A radiology report must never declare 'No acute cardiopulmonary abnormality'
    when acute pathologies (such as pneumothorax, consolidation, or edema) are
    identified as present. Doing so produces a severe internal contradiction that
    compromises diagnostic integrity and patient safety.

    Args:
        findings: List of evaluated Finding objects.

    Returns:
        True if no findings are present or uncertain; False otherwise.
    """
    has_present = any(f.status == "present" for f in findings)
    has_uncertain = any(f.status == "uncertain" for f in findings)
    return not (has_present or has_uncertain)


def rule_format_uncertain_finding(name: str) -> str:
    """Format an uncertain pathology description with mandatory clinical hedging.

    Why this rule exists:
    Linguistic consistency rule: When communicating uncertain findings to the
    radiologist, language must explicitly communicate possibility/equivocality
    (e.g., 'Possible', 'cannot exclude', 'equivocal') and never definitive presence.

    Args:
        name: Name of pathology.

    Returns:
        Hedged finding string.
    """
    clean_name = name.replace("_", " ").lower()
    return f"Possible {clean_name}"


def rule_format_present_finding(name: str) -> str:
    """Format a present pathology description affirmatively.

    Why this rule exists:
    Linguistic consistency rule: Present findings that met or exceeded the
    present threshold are stated directly so the reviewing radiologist can easily
    verify them against the imaging.

    Args:
        name: Name of pathology.

    Returns:
        Clean, capitalized pathology string.
    """
    clean_name = name.replace("_", " ").title()
    return clean_name


def rule_nodule_mass_lung_lesion_consistency(findings: list[Finding]) -> list[Finding]:
    """Ensure that if Nodule or Mass has status 'present', Lung Lesion must not have status 'absent'.

    Why this rule exists:
    Clinical consistency rule: In radiological terminology, pulmonary nodules and masses
    are morphologic subtypes of lung lesions. If an imaging model detects a nodule or mass
    as present, declaring 'Lung Lesion' to be definitively absent is clinically contradictory.
    Therefore, if Nodule or Mass has status 'present', 'Lung Lesion' must not have status 'absent'
    and is adjusted to 'uncertain' to reflect the active focal pulmonary finding.
    This rule is only applied if these pathologies exist in the model's list.

    Args:
        findings: List of evaluated Finding objects.

    Returns:
        List of Finding objects with the consistency adjustment applied.
    """
    findings_map = {f.name.lower().replace("_", " "): f for f in findings}

    nodule_finding = findings_map.get("nodule")
    mass_finding = findings_map.get("mass")
    lesion_finding = findings_map.get("lung lesion")

    # Only apply if Lung Lesion and at least one of Nodule or Mass exist in the findings list
    if lesion_finding is None or (nodule_finding is None and mass_finding is None):
        return findings

    has_present_nodule_or_mass = (
        (nodule_finding is not None and nodule_finding.status == "present")
        or (mass_finding is not None and mass_finding.status == "present")
    )

    if has_present_nodule_or_mass and lesion_finding.status == "absent":
        lesion_finding.status = "uncertain"

    return findings


def findings_from_scores(
    scores: dict[str, float],
    weights: str,
) -> list[Finding]:
    """Map raw model scores to categorized Finding objects adhering to consistency rules.

    Pathologies with NaN scores (e.g. from architectures/weights with uncalibrated
    or unconfigured op_threshs) are safely skipped without creating a Finding,
    and a warning is logged.

    Args:
        scores: Mapping of pathology names to continuous model scores.
        weights: Model weights identifier to look up thresholds via get_thresholds.

    Returns:
        List of structured Finding objects.
    """
    if not weights:
        raise ValueError("weights argument cannot be empty.")

    findings: list[Finding] = []

    for name, score in scores.items():
        if score is None or math.isnan(score):
            logger.warning("Skipping pathology '%s' with NaN model score.", name)
            continue

        status = rule_threshold_status(score, weights=weights)
        finding = Finding(name=name, probability=score, status=status)
        rule_uncertain_is_never_present(finding, weights=weights)
        findings.append(finding)

    # Apply Nodule/Mass vs Lung Lesion clinical consistency rule
    findings = rule_nodule_mass_lung_lesion_consistency(findings)

    return findings


# Backward compatibility alias
findings_from_probs = findings_from_scores
