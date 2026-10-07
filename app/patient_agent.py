"""Patient Explainer Agent for RadiologyAI Copilot.

Provides empathetic, plain-language clinical explanations of chest X-ray
reports, findings, and medical terminology for patients while maintaining
rigorous clinical accuracy and safety gating.
"""

import os
import re
from typing import Any
import requests
from app.config import LLM_API_KEY, LLM_MODEL
from app.schemas import Finding

# Comprehensive clinical dictionary translating radiological terms into
# clear, reassuring, patient-friendly explanations.
PATHOLOGY_CLINICAL_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "Cardiomegaly": {
        "layman_name": "Enlarged Heart Silhouette",
        "plain_summary": (
            "The outline of your heart looks wider than typical on the X-ray image "
            "(taking up more than half the width of the chest cavity)."
        ),
        "clinical_meaning": (
            "This can happen when the heart muscle is working harder over time. Common reasons "
            "include long-standing high blood pressure, fluid retention, or heart muscle strain. "
            "It is very common and treatable with medication and lifestyle adjustments."
        ),
        "urgency_level": "Routine to semi-urgent follow-up with your primary physician or cardiologist.",
        "doctor_questions": [
            "Could my blood pressure or fluid balance be contributing to this?",
            "Would an echocardiogram (ultrasound of the heart) be helpful to check heart function?",
            "Are there any changes needed to my current medications?",
        ],
    },
    "Effusion": {
        "layman_name": "Fluid Around the Lung (Pleural Effusion)",
        "plain_summary": (
            "There is a small buildup of fluid in the natural space between the outside of your "
            "lung and your chest wall."
        ),
        "clinical_meaning": (
            "The lungs sit inside a double-layered membrane. When fluid accumulates there, it can "
            "cause mild breathlessness or a feeling of heaviness. Common causes include heart strain, "
            "recent chest infections, inflammation, or kidney fluid retention."
        ),
        "urgency_level": "Needs timely clinical evaluation, especially if accompanied by shortness of breath or fever.",
        "doctor_questions": [
            "How much fluid is present, and is it on one or both sides?",
            "Is the fluid related to an infection, heart strain, or another cause?",
            "Do I need diuretic water pills or further imaging to monitor it?",
        ],
    },
    "Infiltration": {
        "layman_name": "Lung Infiltration / Cloudiness",
        "plain_summary": (
            "A patch in your lung tissue looks denser or cloudier than the normal black air space on the radiograph."
        ),
        "clinical_meaning": (
            "Normally, healthy lungs are filled with air and appear dark on X-rays. Cloudier areas "
            "(infiltrates) occur when tiny air sacs contain fluid, mucus, or inflammatory cells. This is "
            "frequently seen with bacterial or viral chest infections, bronchitis, or inflammation."
        ),
        "urgency_level": "Prompt follow-up with a doctor, especially if you have a cough, fever, or fatigue.",
        "doctor_questions": [
            "Could this be pneumonia or bronchitis that requires antibiotics or inhalers?",
            "Should I have a follow-up chest X-ray in 4 to 6 weeks to confirm it has cleared?",
            "What warning signs should prompt me to seek urgent care?",
        ],
    },
    "Consolidation": {
        "layman_name": "Lung Consolidation (Dense Air Sac Filling)",
        "plain_summary": (
            "A specific region of the lung is filled with liquid or inflammatory material rather than air."
        ),
        "clinical_meaning": (
            "This is the hallmark radiological sign of pneumonia. When lung tissue consolidates, "
            "the spongy air sacs become solid with inflammatory fluid. Doctors will listen to your chest "
            "and often prescribe antimicrobial therapy if an active infection is present."
        ),
        "urgency_level": "Prompt medical assessment within 24 hours (or immediate if having high fever or breathing distress).",
        "doctor_questions": [
            "Does this confirm a diagnosis of pneumonia?",
            "What antibiotic or breathing treatment is best for my symptoms?",
            "When should I repeat the X-ray to ensure the consolidation has resolved?",
        ],
    },
    "Pneumonia": {
        "layman_name": "Chest Infection / Pneumonia",
        "plain_summary": (
            "An infection causing inflammation and fluid buildup in one or more areas of your lungs."
        ),
        "clinical_meaning": (
            "Pneumonia is an active respiratory infection caused by bacteria, viruses, or occasionally fungi. "
            "Symptoms often include cough with phlegm, fever, chills, and fatigue. Most people recover well "
            "with targeted medical care and adequate rest."
        ),
        "urgency_level": "Prompt medical treatment is recommended to start appropriate antibiotics or supportive care.",
        "doctor_questions": [
            "Is this likely bacterial or viral?",
            "What medications should I take, and how quickly should I expect improvement?",
            "Do I need rest, hydration, or specialized breathing exercises?",
        ],
    },
    "Atelectasis": {
        "layman_name": "Partial Air Sac Deflation (Atelectasis)",
        "plain_summary": (
            "A small section of tiny air sacs in the lung has temporarily deflated or compressed."
        ),
        "clinical_meaning": (
            "Atelectasis is very common and frequently temporary. It often occurs after lying down for long periods, "
            "taking shallow breaths due to chest soreness, or following recent surgery. Deep breathing exercises "
            "and walking usually re-expand the air sacs quickly."
        ),
        "urgency_level": "Usually mild and routine, unless accompanied by severe pain or sudden breathlessness.",
        "doctor_questions": [
            "How large is the area of deflated lung tissue?",
            "Are deep breathing exercises or an incentive spirometer recommended?",
            "Does this explain any mild cough or discomfort I have been feeling?",
        ],
    },
    "Pneumothorax": {
        "layman_name": "Air Pocket Outside the Lung (Collapsed Lung)",
        "plain_summary": (
            "Air has leaked into the space between the chest wall and the outer edge of your lung."
        ),
        "clinical_meaning": (
            "When air gets outside the lung tissue, it can press on the lung and prevent full expansion. "
            "Small pockets may heal on their own with observation and oxygen, while larger ones require a doctor "
            "to release the trapped air."
        ),
        "urgency_level": "🚨 Urgent / Emergency: If you experience sharp, sudden chest pain or shortness of breath, seek emergency care immediately.",
        "doctor_questions": [
            "How large is the pneumothorax (small rim or significant collapse)?",
            "Do I need active intervention or safe monitoring with supplemental oxygen?",
            "What activities should I avoid while this heals?",
        ],
    },
    "Edema": {
        "layman_name": "Fluid Congestion in the Lungs (Pulmonary Edema)",
        "plain_summary": (
            "Excess fluid has accumulated inside the lung tissue and air spaces."
        ),
        "clinical_meaning": (
            "This most commonly happens when the heart struggles to pump fluid forward efficiently, "
            "causing fluid pressure to back up into the lungs. It can make breathing harder when lying flat. "
            "Diuretic medications ('water pills') typically relieve this quickly by removing extra body fluid."
        ),
        "urgency_level": "Requires prompt medical attention to adjust fluid management and heart medications.",
        "doctor_questions": [
            "Is this related to heart function, blood pressure, or kidney fluid balance?",
            "Should my diuretic dosage be adjusted?",
            "What daily salt or fluid restrictions should I follow?",
        ],
    },
    "Emphysema": {
        "layman_name": "Over-Inflated Air Sacs (Emphysema / COPD)",
        "plain_summary": (
            "The air sacs in the lungs are enlarged and hold extra air, often reflecting chronic airway changes."
        ),
        "clinical_meaning": (
            "Emphysema is a form of chronic obstructive pulmonary disease (COPD). The delicate walls of the "
            "air sacs lose elasticity over time, making it harder to push air all the way out. Inhalers and pulmonary "
            "rehabilitation help open airways and improve stamina."
        ),
        "urgency_level": "Chronic condition managed through outpatient pulmonary follow-up.",
        "doctor_questions": [
            "Would a pulmonary function test (breathing spirometry) help evaluate my lung capacity?",
            "Which inhalers or maintenance medications are recommended for me?",
            "Are there breathing techniques or exercise programs that can help my endurance?",
        ],
    },
    "Fibrosis": {
        "layman_name": "Lung Tissue Scarring (Fibrosis)",
        "plain_summary": (
            "Areas of thickened, scarred, or stiffened lung tissue are visible on the radiograph."
        ),
        "clinical_meaning": (
            "Lung scarring can result from previous healed infections, occupational exposures, or chronic "
            "inflammatory conditions. Scarred tissue is less flexible than healthy lung tissue. Doctors often "
            "request a high-resolution CT scan to evaluate the pattern precisely."
        ),
        "urgency_level": "Requires scheduled outpatient evaluation with a pulmonologist.",
        "doctor_questions": [
            "Is this old, stable scar tissue from a past infection, or an active process?",
            "Would a high-resolution chest CT scan provide a clearer evaluation?",
            "Do you recommend seeing a lung specialist (pulmonologist)?",
        ],
    },
    "Nodule": {
        "layman_name": "Small Lung Spot (Pulmonary Nodule)",
        "plain_summary": (
            "A small, rounded spot (usually under 3 centimeters) seen in the lung field."
        ),
        "clinical_meaning": (
            "Lung nodules are remarkably common—more than half of adults who undergo chest imaging have one. "
            "The vast majority are completely benign (harmless), representing tiny scars from old infections "
            "or normal lymph nodes. Radiologists compare with older scans or order a follow-up low-dose CT."
        ),
        "urgency_level": "Requires structured outpatient follow-up to check stability over time.",
        "doctor_questions": [
            "How large is the nodule in millimeters, and where is it located?",
            "Can this be compared with any previous chest X-rays or CT scans I have had?",
            "Is a follow-up low-dose chest CT scan recommended in 3 to 6 months to check for stability?",
        ],
    },
    "Mass": {
        "layman_name": "Larger Lung Spot or Area (Lung Mass)",
        "plain_summary": (
            "A localized area or shadow larger than 3 centimeters in diameter."
        ),
        "clinical_meaning": (
            "Any area larger than 3 cm warrants careful, thorough diagnostic investigation, usually starting "
            "with a contrast-enhanced chest CT scan to view its exact 3D internal structure."
        ),
        "urgency_level": "High priority: Needs prompt outpatient scheduling for definitive CT imaging.",
        "doctor_questions": [
            "How quickly can we schedule a chest CT scan for detailed characterization?",
            "What diagnostic steps do you recommend next?",
            "Should I be referred to a thoracic specialist?",
        ],
    },
    "Pleural_Thickening": {
        "layman_name": "Thickened Lung Lining (Pleural Thickening)",
        "plain_summary": (
            "The smooth lining covering the outside of the lungs has become thicker or firmer."
        ),
        "clinical_meaning": (
            "This is usually a benign scar left behind by a previous episode of pleurisy, resolved pneumonia, "
            "past rib trauma, or prior asbestos exposure. It rarely causes problems on its own."
        ),
        "urgency_level": "Usually a benign, stable finding; routine follow-up.",
        "doctor_questions": [
            "Does this appear to be an old scar from a past infection or injury?",
            "Are there any symptoms I should watch out for?",
        ],
    },
    "Hernia": {
        "layman_name": "Diaphragmatic or Hiatal Hernia",
        "plain_summary": (
            "Part of the stomach or abdominal contents sits slightly above the breathing muscle (diaphragm)."
        ),
        "clinical_meaning": (
            "A hiatal hernia occurs when the upper part of the stomach pushes up through the diaphragm opening. "
            "It is extremely common and frequently causes acid reflux or heartburn, but is usually well-managed with antacids."
        ),
        "urgency_level": "Routine medical management; mention to your doctor if you experience acid reflux.",
        "doctor_questions": [
            "Could this be contributing to any acid reflux or heartburn I experience?",
            "Are dietary changes or acid-reducing medications advised?",
        ],
    },
    "Fracture": {
        "layman_name": "Bone Break or Crack (Rib/Clavicle Fracture)",
        "plain_summary": (
            "A break, crack, or healing bone line in one of the ribs, collarbones, or thoracic bones."
        ),
        "clinical_meaning": (
            "Fractures can be acute (from a recent fall or bump) or old and healed. Rib fractures can make deep "
            "breathing uncomfortable for several weeks. Pain management and gentle deep breathing help prevent secondary infections."
        ),
        "urgency_level": "Needs medical confirmation and appropriate pain management.",
        "doctor_questions": [
            "Which bone is affected, and does it look like a fresh fracture or an old healed one?",
            "What safe pain relief options will allow me to take comfortable deep breaths?",
        ],
    },
    "Lung Opacity": {
        "layman_name": "Lung Opacity / Hazy Shadow",
        "plain_summary": (
            "A generalized hazy or white area on the X-ray where clear black lung air would normally be."
        ),
        "clinical_meaning": (
            "'Opacity' is a general radiological term meaning an area that blocked X-ray beams. It can represent "
            "fluid, inflammation, infection, or atelectasis. Doctors interpret it alongside your fever, cough, and exam."
        ),
        "urgency_level": "Routine to timely clinical correlation with your clinical symptoms.",
        "doctor_questions": [
            "What does this opacity most likely represent based on my physical exam?",
            "Will I need a follow-up radiograph once my symptoms resolve?",
        ],
    },
    "Enlarged Cardiomediastinum": {
        "layman_name": "Widened Central Chest Area",
        "plain_summary": (
            "The central compartment of the chest between your lungs appears broader than usual."
        ),
        "clinical_meaning": (
            "The mediastinum contains the heart, major blood vessels (aorta), trachea, and lymph nodes. "
            "A wider appearance can be due to patient body build, shallow breathing during the picture, "
            "an enlarged heart, or prominent blood vessels."
        ),
        "urgency_level": "Scheduled evaluation by your doctor to determine the underlying cause.",
        "doctor_questions": [
            "Is the widening due to patient positioning or an anatomical structure?",
            "Does my aorta or heart shadow appear prominent?",
        ],
    },
    "Lung Lesion": {
        "layman_name": "Circumscribed Lung Abnormality",
        "plain_summary": (
            "A defined spot or density in the lung tissue that stands out from surrounding structures."
        ),
        "clinical_meaning": (
            "This is a descriptive radiological finding that requires further clarification, typically "
            "by comparing with prior scans or obtaining a chest CT scan."
        ),
        "urgency_level": "Needs structured clinical evaluation with a chest CT scan.",
        "doctor_questions": [
            "What is the recommended next imaging study (e.g., chest CT)?",
            "Can this be compared with historical imaging records?",
        ],
    },
}


def _match_pathology_key(name: str) -> str | None:
    """Find matching knowledge dictionary key for a given pathology name."""
    clean_target = name.strip().lower().replace("_", " ")
    for key in PATHOLOGY_CLINICAL_KNOWLEDGE:
        clean_key = key.lower().replace("_", " ")
        if clean_target == clean_key or clean_target in clean_key or clean_key in clean_target:
            return key
    return None


def generate_offline_patient_explanation(
    report_text: str,
    findings: list[Finding],
    patient_query: str,
) -> str:
    """Synthesize an empathetic, plain-language clinical explanation without an LLM.

    Serves as an offline, rule-based clinical fallback when no LLM API key
    is configured or when external network calls fail.
    """
    present_findings = [f for f in findings if f.status == "present"]
    uncertain_findings = [f for f in findings if f.status == "uncertain"]
    absent_findings = [f for f in findings if f.status == "absent"]

    query_lower = patient_query.lower()

    # 1. Check for specific pathology inquiry in query
    matched_key = None
    for name in PATHOLOGY_CLINICAL_KNOWLEDGE:
        clean = name.lower().replace("_", " ")
        if clean in query_lower:
            matched_key = name
            break

    # If asking about a specific pathology
    if matched_key:
        info = PATHOLOGY_CLINICAL_KNOWLEDGE[matched_key]
        status_on_xray = "Not detected (Absent)"
        f_obj = next((f for f in findings if f.name.lower().replace("_", " ") == matched_key.lower().replace("_", " ")), None)
        if f_obj:
            if f_obj.status == "present":
                status_on_xray = f"⚠️ Detected on your scan (Confidence score: {f_obj.probability:.2f})"
            elif f_obj.status == "uncertain":
                status_on_xray = f"⚡ Possible / Equivocal on your scan (Confidence score: {f_obj.probability:.2f})"
            else:
                status_on_xray = f"✅ Not detected on your scan (Absent, score: {f_obj.probability:.2f})"

        questions_formatted = "\n".join(f"- {q}" for q in info["doctor_questions"])
        return (
            f"### 🩺 Understanding {info['layman_name']}\n\n"
            f"**Status on your Chest X-ray:** {status_on_xray}\n\n"
            f"**What does this mean in plain words?**\n"
            f"{info['plain_summary']}\n\n"
            f"**Why does this happen?**\n"
            f"{info['clinical_meaning']}\n\n"
            f"**Clinical Urgency & Next Steps:**\n"
            f"{info['urgency_level']}\n\n"
            f"**Helpful questions to ask your doctor:**\n"
            f"{questions_formatted}\n\n"
            f"---\n"
            f"*Disclaimer: This explanation is for patient educational understanding and does not replace "
            f"direct consultation or clinical examination with your physician.*"
        )

    # 2. Check if asking about urgency / emergency / danger
    if any(w in query_lower for w in ["emergency", "urgent", "danger", "serious", "die", "harm", "worry"]):
        has_critical = any(f.name in ["Pneumothorax", "Edema", "Effusion"] and f.status == "present" for f in findings)
        if has_critical:
            crit_names = [f.name.replace("_", " ") for f in findings if f.name in ["Pneumothorax", "Edema", "Effusion"] and f.status == "present"]
            return (
                f"### ⚠️ Clinical Assessment: When to Seek Immediate Care\n\n"
                f"Your chest X-ray showed findings that require **timely medical attention**: **{', '.join(crit_names)}**.\n\n"
                f"**What you should do:**\n"
                f"- If you currently feel **severe shortness of breath, sudden sharp chest pain, high fever, or dizziness**, "
                f"please go to an **Emergency Department or Urgent Care facility immediately**.\n"
                f"- If you feel stable and your symptoms are mild, contact your doctor's office today to discuss these results.\n\n"
                f"*Remember: Radiographs must always be evaluated in combination with how you are feeling right now.*"
            )
        else:
            return (
                f"### ✅ Reassurance Regarding Urgent Conditions\n\n"
                f"Based on the neural model evaluation, **no critical acute emergencies** (such as a collapsed lung or acute fluid crisis) were marked as strongly present.\n\n"
                f"**Current Status:**\n"
                f"- **Active findings:** {len(present_findings)} present finding(s).\n"
                f"- **Possible findings:** {len(uncertain_findings)} equivocal finding(s).\n\n"
                f"While these results do not show an immediate emergency, you should still discuss the full report "
                f"with your doctor to address any symptoms you may have."
            )

    # 3. Check if asking what to ask the doctor
    if any(w in query_lower for w in ["ask doctor", "questions for doctor", "what should i ask", "doctor question"]):
        recs: list[str] = []
        for f in present_findings[:2]:
            key = _match_pathology_key(f.name)
            if key and key in PATHOLOGY_CLINICAL_KNOWLEDGE:
                recs.extend(PATHOLOGY_CLINICAL_KNOWLEDGE[key]["doctor_questions"][:2])

        if not recs:
            recs = [
                "Does this chest X-ray explain my current symptoms (cough, shortness of breath, or chest discomfort)?",
                "Are there any treatments, inhalers, or lifestyle changes you recommend based on these findings?",
                "Will I need a follow-up X-ray in the future to check if everything remains clear?",
            ]

        bullets = "\n".join(f"- **{i+1}.** {q}" for i, q in enumerate(recs[:4]))
        return (
            f"### 📋 Key Questions to Bring to Your Doctor\n\n"
            f"Here are the most helpful questions to ask your physician during your consultation:\n\n"
            f"{bullets}\n\n"
            f"💡 *Tip: Writing these questions down in advance can make your appointment much more productive.*"
        )

    # 4. Default: General report summary in clear plain English
    findings_explanation = []
    if present_findings:
        for f in present_findings[:3]:
            key = _match_pathology_key(f.name)
            if key and key in PATHOLOGY_CLINICAL_KNOWLEDGE:
                item = PATHOLOGY_CLINICAL_KNOWLEDGE[key]
                findings_explanation.append(
                    f"- **{item['layman_name']} ({f.name.replace('_', ' ')}):** {item['plain_summary']}"
                )
            else:
                findings_explanation.append(
                    f"- **{f.name.replace('_', ' ')}:** Present on radiograph (confidence {f.probability:.2f})."
                )
    else:
        findings_explanation.append(
            "- **No acute abnormalities detected:** The model did not detect definite disease patterns in the areas it evaluated."
        )

    uncertain_explanation = []
    if uncertain_findings:
        for f in uncertain_findings[:2]:
            uncertain_explanation.append(
                f"- **Possible {f.name.replace('_', ' ')}:** Faint or borderline appearance. Radiologists note this as 'uncertain' "
                f"because overlapping normal tissues can sometimes look like shadows."
            )

    findings_block = "\n".join(findings_explanation)
    uncertain_block = "\n".join(uncertain_explanation) if uncertain_explanation else "None"

    return (
        f"### 🩻 Plain-English Summary of Your Chest X-ray Report\n\n"
        f"Here is what the imaging analysis found, explained in simple medical terms:\n\n"
        f"**1. What was detected:**\n"
        f"{findings_block}\n\n"
        f"**2. Borderline / Equivocal shadows:**\n"
        f"{uncertain_block}\n\n"
        f"**3. What does 'Uncertain' or 'Possible' mean?**\n"
        f"In chest imaging, 'possible' means there is a mild shadow that could simply be normal overlapping ribs or blood vessels. "
        f"Doctors mention it out of clinical caution, not immediate alarm.\n\n"
        f"**Next Steps:**\n"
        f"Review this with your treating clinician, who will compare these pictures with how you are feeling and determine if any medication or follow-up imaging is needed.\n\n"
        f"---\n"
        f"*Feel free to ask me about any specific term in your report (like 'What does cardiomegaly mean?')!*"
    )


def call_gemini_patient_agent(
    report_text: str,
    findings: list[Finding],
    patient_query: str,
    conversation_history: list[dict[str, str]] | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Invoke Google Gemini LLM with an empathetic patient-explainer system prompt.

    Translates clinical reports and vision probabilities into accessible,
    reassuring, yet medically rigorous educational explanations.
    """
    key = api_key or LLM_API_KEY
    if not key:
        return generate_offline_patient_explanation(report_text, findings, patient_query)

    # Format findings context
    present_list = [f"{f.name.replace('_', ' ')} (prob: {f.probability:.2f})" for f in findings if f.status == "present"]
    uncertain_list = [f"{f.name.replace('_', ' ')} (prob: {f.probability:.2f})" for f in findings if f.status == "uncertain"]
    absent_list = [f.name.replace('_', ' ') for f in findings if f.status == "absent"]

    # Format conversation history
    history_snippets = []
    if conversation_history:
        for msg in conversation_history[-4:]:  # Include last 2 turns
            role = "Patient" if msg.get("role") == "user" else "Copilot"
            history_snippets.append(f"{role}: {msg.get('content', '')}")
    history_text = "\n".join(history_snippets) if history_snippets else "None (Start of conversation)"

    prompt = f"""You are 'RadiologyAI Patient Guide', an empathetic, articulate, and medically rigorous clinical communicator.
A patient has uploaded their chest X-ray, received an AI radiology report, and is asking you a question to understand their results.

PATIENT'S MEDICAL REPORT CONTEXT:
Official Report Text:
\"\"\"
{report_text}
\"\"\"

NEURAL VISION MODEL CLASSIFICATION:
- Detected Present Pathologies: {', '.join(present_list) if present_list else 'None'}
- Uncertain / Possible Pathologies: {', '.join(uncertain_list) if uncertain_list else 'None'}
- Absent Pathologies: {', '.join(absent_list) if absent_list else 'None'}

RECENT CONVERSATION HISTORY:
{history_text}

PATIENT'S CURRENT QUESTION:
"{patient_query}"

INSTRUCTIONS FOR YOUR RESPONSE:
1. TONE & BEDSIDE MANNER: Warm, supportive, reassuring, and calm. Avoid alarming words, but remain truthful and medically accurate.
2. PLAIN-ENGLISH TRANSLATION: If the report contains medical jargon (e.g., cardiomegaly, consolidation, effusion, atelectasis, opacity, costophrenic blunting), define and explain it using everyday words and simple analogies (e.g., fluid accumulation, air sac filling, shadow of the heart).
3. ACCURACY & CONTEXT: Strictly align with the patient's specific X-ray report above. Mention what was seen and what was NOT seen (negative findings provide immense reassurance).
4. UNCERTAINTY HEDGING: If a finding is listed as 'uncertain' or 'possible', explain that normal overlapping tissue or breathing depth can create shadows, which is why doctors note it cautiously.
5. ACTIONABLE QUESTIONS: Provide 2 to 3 practical, intelligent questions the patient can ask their doctor at their next visit.
6. SAFETY BOUNDS: Always include a gentle reminder that this educational AI cannot replace an in-person physical exam and that they should review these findings with their attending doctor.
7. FORMATTING: Use clean markdown with clear headings, bullet points, and bold text for easy reading on mobile and desktop.
"""

    model = model_name or LLM_MODEL or "gemini-3.8-flash"
    clean_model = model.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return generate_offline_patient_explanation(report_text, findings, patient_query)
        else:
            return generate_offline_patient_explanation(report_text, findings, patient_query)
    except Exception:
        return generate_offline_patient_explanation(report_text, findings, patient_query)
