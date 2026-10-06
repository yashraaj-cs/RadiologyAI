# RadiologyAI Copilot Specification

## 1. Goal
Assist radiologists by generating a preliminary **DRAFT report** for chest X-rays.
- **Human review is mandatory:** No output is finalized or used clinically without review and sign-off by a qualified radiologist.
- **Safety first:** Prevent clinical hallucinations and ensure full traceability back to detected findings and medical literature/guidelines.

## 2. System Pipeline
The drafting workflow follows a strict sequential pipeline with fallback guarantees:

1. **Image Ingestion & Vision Model:**
   - Input: 2D Chest X-ray image (PNG / JPEG).
   - Vision Backbone: Pretrained `TorchXRayVision` models. ResNet50 (`resnet50-res512-all`) is the headline model unless its dev-split performance is unusable (decided on dev before the test split is touched). If so, DenseNet results are reported with the OpenI contamination flagged. DenseNet (`densenet121-res224-all`) is otherwise reported as a secondary comparison and is flagged as contaminated on IU X-Ray because it was trained on OpenI. Two pathologies (`Lung Lesion`, `Enlarged Cardiomediastinum`) have no operating point on the ResNet and are dropped for it.
   - Output: Continuous model scores for chest pathologies.

2. **Rule Layer (Deterministic Categorization):**
   - Maps continuous pathology model scores into discrete status levels:
     - `present` (score >= `PRESENT_THRESHOLD`)
     - `uncertain` (`ABSENT_THRESHOLD` <= score < `PRESENT_THRESHOLD`)
     - `absent` (score < `ABSENT_THRESHOLD`)
   - Operating thresholds are per model weights in `app/config.py` (currently UNTUNED placeholders), tuned on the dev split only.
   - Outputs a structured list of `Finding` objects.

3. **Deterministic Template Report (No LLM):**
   - Synthesizes findings into a safe, rule-based baseline report.
   - Guaranteed hallucination-free; serves as the ground-truth baseline and safety fallback.

4. **RAG Context Retrieval:**
   - Queries a local vector store (ChromaDB) for institutional reporting style, relevant guideline excerpts, or differential diagnosis context matching detected findings.

5. **LLM Rewriting (Drafting):**
   - Uses Gemini (model string from `.env` `LLM_MODEL`, pinned and reported in the README) for Variant A, C, and D.
   - Paid tier is recommended for evaluation runs (free-tier content may be used by Google); record which tier was used in the README. Runs use temperature 0. Every call is cached and logged.
   - Formulates coherent **Findings** and **Impression** sections combining the structured findings, real IU `INDICATION` text (the only text input provided to the LLM), and retrieved medical context.

6. **Verifier Layer:**
   - Validates that the generated draft strictly adheres to the detected findings list.
   - **Hard Rule:** The LLM may **only** mention findings that appear in the input findings list. Any pathology named by the LLM that is not in the list constitutes an immediate **verification failure**.
   - If verification fails: Retry drafting once with an explicit correction prompt.
   - If verification fails a second time: Safely **fall back to the deterministic template report**.

## 3. Evaluation Plan

### Comparative Variants
The system will be evaluated across 4 comparative variants. Gemini is used for Variant A, C, and D, and all three get the same inputs: the same image and the real IU `INDICATION` text (the only text input provided to the LLM). Blocked or empty LLM responses are counted as their own outcome.
- **Variant A (Raw Multimodal LLM):** Direct multimodal prompt with image input and indication text (same image and indication as D; no vision model or verifier).
- **Variant B (Vision + Template):** TorchXRayVision predictions routed through deterministic rule-based template generation (no LLM).
- **Variant C (Vision + LLM, No Verifier):** TorchXRayVision + RAG + LLM report drafting without verification gate.
- **Variant D (Full System):** TorchXRayVision + Rules + RAG + LLM + Verifier + Retry/Fallback mechanism.

*Note on RAG contribution:* The effect of RAG is not isolated unless an extra variant (vision + LLM, no RAG) is run; if not run, state that RAG's contribution is not measured.

### Evaluation Protocol
- **Dataset:** IU X-Ray (CC BY-NC-ND, never committed to repository).
- **Ground Truth Labels:** Derived from MeSH and automatic index terms (`eval/labels.py`) and are noisy. Positives are only clearly stated, unhedged, fully readable findings (`eval_role` `"positive"`).
- **Dataset Split:** Split by `report_id` at seed 0, 50/50, stratified by normal; the XML has no patient IDs, so the same patient may appear in both splits.
- **Threshold Tuning:** Thresholds are tuned on dev only; the test split is used once for the final numbers.
- **Evaluated Pathologies:** Cardiomegaly, Lung Opacity, Atelectasis, Emphysema, Effusion, Nodule; all others are reported as "not evaluated (too few positives)".
- **Indication Pathology Stem Check:** Report how often the indication text contains the stem of an evaluated pathology (using `eval/labels.py`), and score those studies separately to quantify potential indication leakage.
- **Reporting:** Report ROC AUC and precision/recall with confidence intervals, and state that the enriched LLM subset changes prevalence.

### Scoring Generated Reports
- Every LLM variant returns JSON with a status per pathology from the closed list plus the findings and impression text.
- Per-finding metrics are computed from the JSON.
- The text is cross-checked with the stem/negation extractor in `eval/labels.py` and disagreements are counted.

### Hallucination Rates
Two hallucination rates are defined:
- **Unsupported:** The draft names a finding that is not in the vision findings list.
  - For **Variant C**, report the Unsupported rate.
  - For **Variant D**, the final-output Unsupported rate is zero by construction, so report instead:
    (a) the first-draft verification failure rate, and
    (b) the fallback-to-template rate.
- **Contradicted:** The draft asserts a finding whose ground-truth label is `negative_clean` only (applies to Variants A through D).
  - For Variants B, C, and D, this rate is dominated by vision-model false positives.
  - Variant A's rate represents the pure-LLM figure.
  - The evaluated LLM subset must include enough clean-normal (`negative_clean`) studies to measure it reliably.
- A sample of flagged cases is read by hand and the corrected rate is reported.

### Metrics
- **Per-finding Precision, Recall, F1 score, and ROC AUC** (with confidence intervals) across evaluated pathologies against ground-truth labels.
- **Hallucination & Verification Metrics:**
  - Variant C: Unsupported rate.
  - Variant D: First-draft verification failure rate and fallback-to-template rate.
  - Variants A through D: Contradicted rate on clean negatives (`negative_clean`), including hand-verified corrected rates.

## 4. Scope Boundaries
The following components are explicitly **out of scope** for the current phase:
- AWS deployment / cloud infra
- Model fine-tuning (vision or LLM)
- Interactive Q&A chat functionality
- Grad-CAM / saliency map generation
- MONAI framework integrations
- Qdrant vector database
