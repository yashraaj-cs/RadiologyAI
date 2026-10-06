"""IU X-Ray XML report parser and loader.

Parses XML reports from the Indiana University Chest X-Ray collection
into structured records and DataFrames.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
import xml.etree.ElementTree as ET

import pandas as pd


def clean_token_xxxx(text: Optional[str]) -> str:
    """Replace the HIPAA redaction token 'XXXX' with '[REDACTED]' and normalize whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text.strip():
        return ""
    # Replace XXXX with [REDACTED]
    cleaned = re.sub(r"XXXX", "[REDACTED]", text)
    # Collapse intra-line whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Clean up spaces before punctuation (e.g. ' .' -> '.')
    cleaned = re.sub(r" +([.,;:?!])", r"\1", cleaned)
    # Strip whitespace line by line and recombine
    lines = [line.strip() for line in cleaned.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def parse_report_xml(xml_source: Union[str, Path, ET.Element, ET.ElementTree]) -> Dict[str, Any]:
    """Parse a single IU X-Ray XML report into a structured dictionary.

    Args:
        xml_source: An XML string, file Path, or parsed Element/ElementTree.

    Returns:
        Dictionary containing:
            - report_id: str (from <uId id="...">)
            - indication: str (raw text)
            - comparison: str (raw text)
            - findings: str (raw text)
            - impression: str (raw text)
            - indication_clean: str (XXXX removed)
            - comparison_clean: str (XXXX removed)
            - findings_clean: str (XXXX removed)
            - impression_clean: str (XXXX removed)
            - clean_findings: str (alias for findings_clean)
            - clean_impression: str (alias for impression_clean)
            - major_terms: list[str] (from <MeSH><major>, whitespace stripped)
            - auto_terms: list[str] (from <MeSH><automatic>, whitespace stripped)
            - image_ids: list[str] (from <parentImage id="...">)
            - has_image: bool
            - has_findings: bool
            - has_impression: bool
    """
    if isinstance(xml_source, ET.ElementTree):
        root = xml_source.getroot()
    elif isinstance(xml_source, ET.Element):
        root = xml_source
    elif isinstance(xml_source, Path):
        root = ET.parse(xml_source).getroot()
    elif isinstance(xml_source, str):
        if "<" in xml_source:
            root = ET.fromstring(xml_source)
        else:
            root = ET.parse(xml_source).getroot()
    else:
        raise TypeError(f"Unsupported xml_source type: {type(xml_source)}")

    # Report ID from <uId id="...">
    uid_el = root.find("uId")
    report_id = uid_el.get("id", "").strip() if uid_el is not None else ""

    # AbstractText sections (INDICATION, COMPARISON, FINDINGS, IMPRESSION)
    sections: Dict[str, str] = {
        "INDICATION": "",
        "COMPARISON": "",
        "FINDINGS": "",
        "IMPRESSION": "",
    }
    for at in root.findall(".//AbstractText"):
        lbl = (at.get("Label") or "").upper().strip()
        if lbl in sections and at.text is not None:
            sections[lbl] = at.text.strip()

    indication = sections["INDICATION"]
    comparison = sections["COMPARISON"]
    findings = sections["FINDINGS"]
    impression = sections["IMPRESSION"]

    indication_clean = clean_token_xxxx(indication)
    comparison_clean = clean_token_xxxx(comparison)
    findings_clean = clean_token_xxxx(findings)
    impression_clean = clean_token_xxxx(impression)

    # MeSH terms
    mesh_el = root.find("MeSH")
    major_terms: List[str] = []
    auto_terms: List[str] = []
    if mesh_el is not None:
        for m in mesh_el.findall("major"):
            if m.text is not None:
                term = m.text.strip()
                if term:
                    major_terms.append(term)
        for a in mesh_el.findall("automatic"):
            if a.text is not None:
                term = a.text.strip()
                if term:
                    auto_terms.append(term)

    # Image IDs from <parentImage id="...">
    image_ids: List[str] = []
    for pi in root.findall("parentImage"):
        iid = (pi.get("id") or "").strip()
        if iid:
            image_ids.append(iid)

    has_image = len(image_ids) > 0
    has_findings = bool(findings and findings.strip())
    has_impression = bool(impression and impression.strip())

    return {
        "report_id": report_id,
        "indication": indication,
        "comparison": comparison,
        "findings": findings,
        "impression": impression,
        "indication_clean": indication_clean,
        "comparison_clean": comparison_clean,
        "findings_clean": findings_clean,
        "impression_clean": impression_clean,
        "clean_findings": findings_clean,
        "clean_impression": impression_clean,
        "major_terms": major_terms,
        "auto_terms": auto_terms,
        "image_ids": image_ids,
        "has_image": has_image,
        "has_findings": has_findings,
        "has_impression": has_impression,
    }


def load_reports(reports_dir: Union[str, Path] = "data/iu_xray/reports/ecgen-radiology") -> pd.DataFrame:
    """Parse all XML files in reports_dir into a pandas DataFrame."""
    reports_dir = Path(reports_dir)
    xml_files = sorted(reports_dir.glob("*.xml"))
    records = [parse_report_xml(p) for p in xml_files]
    return pd.DataFrame(records)


def save_reports_csv(df: pd.DataFrame, output_path: Union[str, Path] = "data/iu_xray/index/reports.csv") -> None:
    """Save the reports DataFrame to CSV with JSON-encoded lists and clean empty strings."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_to_save = df.copy()
    text_cols = [
        "indication", "comparison", "findings", "impression",
        "indication_clean", "comparison_clean", "findings_clean", "impression_clean",
        "clean_findings", "clean_impression",
    ]
    for col in text_cols:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].fillna("").astype(str)
    for col in ["major_terms", "auto_terms", "image_ids"]:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(lambda x: json.dumps(x) if isinstance(x, (list, tuple)) else x)
    df_to_save.to_csv(output_path, index=False)


def load_reports_csv(csv_path: Union[str, Path] = "data/iu_xray/index/reports.csv") -> pd.DataFrame:
    """Load the reports index CSV and deserialize JSON lists, keeping empty sections as empty strings."""
    df = pd.read_csv(csv_path, keep_default_na=False)
    text_cols = [
        "indication", "comparison", "findings", "impression",
        "indication_clean", "comparison_clean", "findings_clean", "impression_clean",
        "clean_findings", "clean_impression",
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    for col in ["major_terms", "auto_terms", "image_ids"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith("[") else (x if isinstance(x, list) else [])
            )
    for col in ["has_image", "has_findings", "has_impression"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


if __name__ == "__main__":
    reports_dir = Path("data/iu_xray/reports/ecgen-radiology")
    output_csv = Path("data/iu_xray/index/reports.csv")
    print(f"Parsing XML reports from {reports_dir}...")
    df = load_reports(reports_dir)
    save_reports_csv(df, output_csv)
    print(f"Saved {len(df)} reports to {output_csv}")
    print(f"Reports with at least one image: {df['has_image'].sum()} / {len(df)}")
    print(f"Reports with findings: {df['has_findings'].sum()} / {len(df)}")
    print(f"Reports with impression: {df['has_impression'].sum()} / {len(df)}")
