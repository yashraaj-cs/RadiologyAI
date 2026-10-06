"""Deterministic report generator for RadiologyAI Copilot.

Generates structured, rule-based draft reports (Findings and Impression)
without invoking an LLM. Serves as the ground-truth baseline and safety fallback.
"""

from app.rules import (
    rule_allow_no_acute_findings,
    rule_format_present_finding,
    rule_format_uncertain_finding,
)
from app.schemas import Finding, PatientContext, Report


def template_report(
    findings: list[Finding],
    patient: PatientContext | None = None,
) -> Report:
    """Generate a deterministic, hallucination-free template report from findings.

    Strictly complies with clinical consistency and safety rules:
    - Never outputs 'No acute findings' or 'No acute cardiopulmonary abnormality'
      if any finding is present or uncertain.
    - If a pathology is uncertain, it is explicitly worded with clinical hedging
      ('Possible X.') and never stated as present.
    - Impression is capped at the 5 highest-scoring present findings plus at most
      2 uncertain findings as 'Possible X.'.
    - Any remaining present findings beyond the top 5 are listed in Findings only,
      accompanied by a sentence saying 'Additional findings are listed above.'
    - Findings section groups remaining uncertain findings in one sentence.
    - A report may state absence of a pathology only if that pathology has status 'absent'.
      Negative sentences use 'Not detected by the model:' built strictly from absent findings.
    - Every pathology named anywhere in the report text exists in the findings list.

    Args:
        findings: List of evaluated Finding objects.
        patient: Optional synthetic clinical patient context.

    Returns:
        Report instance with generated_by="template".
    """
    present_findings = [f for f in findings if f.status == "present"]
    uncertain_findings = [f for f in findings if f.status == "uncertain"]
    absent_findings = [f for f in findings if f.status == "absent"]

    # Sort present findings descending by score (cap Impression at top 5)
    sorted_present = sorted(present_findings, key=lambda f: f.probability, reverse=True)
    top_present = sorted_present[:5]
    remaining_present = sorted_present[5:]

    # Sort uncertain findings descending by score (cap Impression at top 2)
    sorted_uncertain = sorted(uncertain_findings, key=lambda f: f.probability, reverse=True)
    top_uncertain = sorted_uncertain[:2]
    remaining_uncertain = sorted_uncertain[2:]

    # Optional clinical indication header
    indication_header = ""
    if patient and patient.clinical_note:
        indication_header = f"INDICATION: {patient.clinical_note}\n\n"

    findings_paragraphs: list[str] = []
    if indication_header:
        findings_paragraphs.append(indication_header.strip())

    impression_items: list[str] = []

    # Case 1: All findings absent (or empty findings list)
    if not present_findings and not uncertain_findings:
        n = len(findings)
        if absent_findings:
            absent_names = [f.name.replace("_", " ").lower() for f in absent_findings]
            findings_paragraphs.append(
                f"Not detected by the model: {', '.join(absent_names)}."
            )
        else:
            findings_paragraphs.append("No specific abnormalities identified.")

        findings_text = "\n".join(findings_paragraphs)
        impression_text = (
            f"No findings detected by the model among the {n} pathologies it evaluates. "
            f"This does not exclude abnormalities outside the model's scope."
        )
        cited_findings: list[str] = []

    # Case 2: One or more present findings exist (with or without uncertain findings)
    elif present_findings:
        # Consistency assertion: never output 'No acute findings' when present findings exist
        assert not rule_allow_no_acute_findings(findings), (
            "Consistency violation: present findings cannot allow 'No acute findings'."
        )

        # 1. Describe all present findings in Findings
        for f in sorted_present:
            title_name = rule_format_present_finding(f.name)
            findings_paragraphs.append(
                f"- {title_name}: Present (model score: {f.probability:.2f})."
            )

        # If more than 5 present findings, note in Findings that additional present findings are listed above
        if remaining_present:
            findings_paragraphs.append("- Additional findings are listed above.")

        # 2. Describe top <= 2 uncertain findings
        for f in top_uncertain:
            hedged_name = rule_format_uncertain_finding(f.name)
            findings_paragraphs.append(
                f"- {hedged_name}: Indeterminate appearance (model score: {f.probability:.2f}); cannot be excluded."
            )

        # 3. Group remaining uncertain findings in one sentence
        if remaining_uncertain:
            remaining_names = [f.name.replace("_", " ").lower() for f in remaining_uncertain]
            findings_paragraphs.append(
                f"- Additional equivocal findings that cannot be excluded: {', '.join(remaining_names)}."
            )

        # 4. State absence of pathologies ONLY if status is 'absent'; omit if none
        if absent_findings:
            absent_names = [f.name.replace("_", " ").lower() for f in absent_findings]
            findings_paragraphs.append(
                f"- Not detected by the model: {', '.join(absent_names)}."
            )

        findings_text = "\n".join(findings_paragraphs)

        # Build Impression: cap at 5 highest-scoring present findings
        for f in top_present:
            title_name = rule_format_present_finding(f.name)
            impression_items.append(f"{title_name}.")

        # At most 2 highest-scoring uncertain findings as 'Possible X.'
        for f in top_uncertain:
            title_name = rule_format_present_finding(f.name)
            impression_items.append(f"Possible {title_name}.")

        # If remaining present findings were omitted from Impression, indicate they are listed above
        if remaining_present:
            impression_items.append("Additional findings are listed above.")

        impression_text = "\n".join(impression_items)
        cited_findings = [f.name for f in present_findings] + [f.name for f in top_uncertain]

    # Case 3: Only uncertain findings exist (no definitive present findings)
    else:
        # 1. Describe top <= 2 uncertain findings
        for f in top_uncertain:
            hedged_name = rule_format_uncertain_finding(f.name)
            findings_paragraphs.append(
                f"- {hedged_name}: Indeterminate appearance (model score: {f.probability:.2f}); cannot be excluded."
            )

        # 2. Group remaining uncertain findings in one sentence
        if remaining_uncertain:
            remaining_names = [f.name.replace("_", " ").lower() for f in remaining_uncertain]
            findings_paragraphs.append(
                f"- Additional equivocal findings that cannot be excluded: {', '.join(remaining_names)}."
            )

        # 3. State absence of pathologies ONLY if status is 'absent'; omit if none
        if absent_findings:
            absent_names = [f.name.replace("_", " ").lower() for f in absent_findings]
            findings_paragraphs.append(
                f"- Not detected by the model: {', '.join(absent_names)}."
            )

        findings_text = "\n".join(findings_paragraphs)

        # Build impression: at most 2 top uncertain findings as 'Possible X.'
        for f in top_uncertain:
            title_name = rule_format_present_finding(f.name)
            impression_items.append(f"Possible {title_name}.")

        impression_text = "\n".join(impression_items)
        cited_findings = [f.name for f in top_uncertain]

    return Report(
        findings_text=findings_text,
        impression_text=impression_text,
        cited_findings=cited_findings,
        sources=["deterministic_rules_template_v1"],
        generated_by="template",
    )
