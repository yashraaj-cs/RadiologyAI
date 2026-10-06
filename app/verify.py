"""Verification layer stub to prevent hallucinations in generated reports."""

from app.schemas import Finding, Report


def verify_report(draft_report: Report, allowed_findings: list[Finding]) -> tuple[bool, list[str]]:
    """Validate that the draft report only mentions findings in the allowed findings list.

    Hard rule: Any pathology named by the LLM that is not in the findings list
    is an immediate verification failure.

    Args:
        draft_report: The LLM-drafted report to inspect.
        allowed_findings: The ground-truth vision findings for this case.

    Returns:
        Tuple of (is_valid: bool, issues: list[str]) (stub).
    """
    # Stub implementation: clinical entity extraction and membership validation
    raise NotImplementedError("verify_report stub - to be implemented in pipeline phase")
