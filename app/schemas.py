"""Pydantic v2 schemas for findings, reports, and clinical patient context."""

from typing import Literal
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Pathology finding detected or evaluated on a chest X-ray."""

    name: str = Field(description="Name of the pathology or clinical finding")
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score or predicted probability between 0.0 and 1.0",
    )
    status: Literal["present", "uncertain", "absent"] = Field(
        description="Categorical status determined by threshold rules",
    )


class Report(BaseModel):
    """Draft radiology report containing structured findings and impression."""

    findings_text: str = Field(description="Detailed narrative findings section")
    impression_text: str = Field(description="Summary clinical impression section")
    cited_findings: list[str] = Field(
        default_factory=list,
        description="List of specific findings referenced in the drafted report",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="References, retrieved guidelines, or template identifiers",
    )
    generated_by: Literal["template", "llm"] = Field(
        description="Origin of the report generation: rule-based template or LLM rewrite",
    )


class PatientContext(BaseModel):
    """Anonymized synthetic clinical context for the patient."""

    age: int | None = Field(default=None, description="Patient age in years")
    sex: str | None = Field(default=None, description="Patient biological sex")
    clinical_note: str | None = Field(
        default=None,
        description="Indication or brief clinical history (synthetic data only)",
    )
