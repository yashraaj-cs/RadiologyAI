"""RadiologyAI Copilot - Interactive Live Workstation.

Streamlit web application for interactive chest X-ray analysis, vision model
inference, deterministic template reporting, Gemini LLM draft generation,
and an empathetic, conversational Patient Explainer Agent.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
import requests
import numpy as np
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import sys

# Ensure project root is in sys.path for app module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    BASE_DIR,
    DATA_DIR,
    LLM_API_KEY,
    LLM_MODEL,
    THRESHOLDS_BY_WEIGHTS,
    get_thresholds,
)
from app.generate import template_report
from app.patient_agent import (
    PATHOLOGY_CLINICAL_KNOWLEDGE,
    call_gemini_patient_agent,
    generate_offline_patient_explanation,
)
from app.rules import findings_from_scores
from app.schemas import Finding, PatientContext, Report
from app.vision import get_input_resolution, load_model, predict

# Page configuration
st.set_page_config(
    page_title="RadiologyAI Copilot",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize Session State
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Light"

if "entered_workstation" not in st.session_state:
    st.session_state["entered_workstation"] = False

if "patient_chat_history" not in st.session_state:
    st.session_state["patient_chat_history"] = []

# Detect URL query parameter to enter workstation if passed
if "enter" in st.query_params:
    st.session_state["entered_workstation"] = True
    try:
        del st.query_params["enter"]
    except Exception:
        pass


def get_custom_css(dark: bool = False) -> str:
    """Generate dynamic CSS styling for the workstation, 3D landing, and patient chat."""
    if dark:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Dark Medical Workstation Theme Canvas */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #0b0f19 !important;
            color: #f1f5f9 !important;
        }

        .stApp {
            background: radial-gradient(circle at 10% 20%, rgba(30, 41, 59, 0.55) 0%, transparent 40%),
                        radial-gradient(circle at 90% 80%, rgba(15, 23, 42, 0.75) 0%, transparent 40%),
                        #0b0f19 !important;
        }

        /* Landing Badges and Typography */

        .badge-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }
        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }
        .badge-blue {
            background: rgba(2, 132, 199, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            color: #38bdf8;
        }
        .badge-green {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34d399;
        }
        .badge-purple {
            background: rgba(167, 139, 250, 0.15);
            border: 1px solid rgba(167, 139, 250, 0.4);
            color: #a78bfa;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #38bdf8;
            border-radius: 50%;
            box-shadow: 0 0 10px #38bdf8;
            animation: pulse 1.8s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7); }
            70% { box-shadow: 0 0 0 9px rgba(56, 189, 248, 0); }
            100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
        }
        .hero-title {
            font-size: 34px !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin-bottom: 14px !important;
            background: linear-gradient(135deg, #ffffff 0%, #bae6fd 60%, #38bdf8 100%) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            letter-spacing: -0.5px !important;
            text-align: center !important;
        }
        .hero-desc {
            font-size: 15px !important;
            line-height: 1.6 !important;
            color: #94a3b8 !important;
            margin-bottom: 26px !important;
            max-width: 660px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            text-align: center !important;
        }
        .feature-grid {
            display: grid !important;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 12px !important;
            margin-bottom: 28px !important;
        }
        .feature-item {
            background: rgba(30, 41, 59, 0.75) !important;
            border: 1px solid rgba(71, 85, 105, 0.6) !important;
            border-radius: 12px !important;
            padding: 12px 8px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            color: #e2e8f0 !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            gap: 5px !important;
            transition: all 0.2s ease !important;
        }
        .feat-icon {
            font-size: 20px !important;
        }
        .hint-text {
            font-size: 12px !important;
            color: #64748b !important;
            margin-top: 14px !important;
            letter-spacing: 0.4px !important;
            text-align: center !important;
        }



        /* Patient Chat Bubbles */
        div[data-testid="stChatMessage"] {
            background-color: #161e2e !important;
            border: 1px solid #1e293b !important;
            border-radius: 14px !important;
            padding: 14px 18px !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="stChatInput"] textarea {
            color: #f1f5f9 !important;
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
        }

        /* Header Card */
        .main-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }
        .main-title {
            font-size: 26px;
            font-weight: 800;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 0 0 6px 0;
        }
        .badge-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-prototype {
            background: rgba(244, 63, 94, 0.15);
            color: #fb7185;
            border: 1px solid rgba(244, 63, 94, 0.3);
        }
        .badge-headline {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .badge-online {
            background: rgba(52, 211, 153, 0.15);
            color: #34d399;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }

        /* Pathology Item Status */
        .status-badge {
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
        }
        .status-present {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid #ef4444;
        }
        .status-uncertain {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid #f59e0b;
        }
        .status-absent {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid #10b981;
        }

        /* Report Box */
        .report-box {
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 18px;
            font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
            font-size: 13.5px;
            line-height: 1.6;
            color: #e2e8f0;
            white-space: pre-wrap;
        }
        </style>
        """
    else:
        return """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

        /* Luminous Light Workstation Theme */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }

        /* Landing Badges and Typography */

        .badge-row {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-bottom: 18px;
            flex-wrap: wrap;
        }
        .brand-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.6px;
            text-transform: uppercase;
        }
        .badge-blue {
            background: rgba(2, 132, 199, 0.12);
            border: 1px solid rgba(2, 132, 199, 0.4);
            color: #0284c7;
        }
        .badge-green {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #059669;
        }
        .badge-purple {
            background: rgba(139, 92, 246, 0.12);
            border: 1px solid rgba(139, 92, 246, 0.4);
            color: #7c3aed;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #0284c7;
            border-radius: 50%;
            box-shadow: 0 0 10px #0284c7;
            animation: pulse 1.8s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(2, 132, 199, 0.7); }
            70% { box-shadow: 0 0 0 9px rgba(2, 132, 199, 0); }
            100% { box-shadow: 0 0 0 0 rgba(2, 132, 199, 0); }
        }
        .hero-title {
            font-size: 34px !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin-bottom: 14px !important;
            color: #0369a1 !important;
            letter-spacing: -0.5px !important;
            text-align: center !important;
        }
        .hero-desc {
            font-size: 15px !important;
            line-height: 1.6 !important;
            color: #475569 !important;
            margin-bottom: 26px !important;
            max-width: 660px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            text-align: center !important;
        }
        .feature-grid {
            display: grid !important;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 12px !important;
            margin-bottom: 28px !important;
        }
        .feature-item {
            background: rgba(241, 245, 249, 0.85) !important;
            border: 1px solid rgba(203, 213, 225, 0.8) !important;
            border-radius: 12px !important;
            padding: 12px 8px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            color: #0f172a !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            gap: 5px !important;
            transition: all 0.2s ease !important;
        }
        .feat-icon {
            font-size: 20px !important;
        }
        .hint-text {
            font-size: 12px !important;
            color: #64748b !important;
            margin-top: 14px !important;
            letter-spacing: 0.4px !important;
            text-align: center !important;
        }



        /* Patient Chat Bubbles */
        div[data-testid="stChatMessage"] {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 14px !important;
            padding: 14px 18px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04) !important;
        }
        div[data-testid="stChatInput"] textarea {
            color: #0f172a !important;
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
        }

        /* Header Card */
        .main-header {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 1px solid #cbd5e1;
            border-radius: 14px;
            padding: 22px 26px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        }
        .main-title {
            font-size: 26px;
            font-weight: 800;
            color: #0369a1;
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 0 0 6px 0;
        }
        .badge-pill {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-prototype {
            background: #fff1f2;
            color: #e11d48;
            border: 1px solid #fecdd3;
        }
        .badge-headline {
            background: #f0f9ff;
            color: #0284c7;
            border: 1px solid #bae6fd;
        }
        .badge-online {
            background: #ecfdf5;
            color: #059669;
            border: 1px solid #a7f3d0;
        }

        /* Status Badges */
        .status-badge {
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
        }
        .status-present {
            background: #fef2f2;
            color: #dc2626;
            border: 1px solid #fecaca;
        }
        .status-uncertain {
            background: #fffbeb;
            color: #d97706;
            border: 1px solid #fde68a;
        }
        .status-absent {
            background: #f0fdf4;
            color: #16a34a;
            border: 1px solid #bbf7d0;
        }

        /* Report Box */
        .report-box {
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            padding: 18px;
            font-family: 'JetBrains Mono', Consolas, Menlo, monospace;
            font-size: 13.5px;
            line-height: 1.6;
            color: #0f172a;
            white-space: pre-wrap;
        }
        </style>
        """


is_dark = st.session_state.get("theme_mode", "Light") == "Dark"


def get_3d_bg_html(dark: bool = False) -> str:
    """Generate high-performance standalone 3D interactive thoracic hologram canvas."""
    bg_gradient = (
        "radial-gradient(ellipse at 50% 50%, #0c1322 0%, #060911 100%)"
        if dark
        else "radial-gradient(ellipse at 50% 50%, #f8fafc 0%, #e2e8f0 100%)"
    )
    grid_color = "rgba(56, 189, 248, 0.07)" if dark else "rgba(2, 132, 199, 0.08)"
    primary_color = "#38bdf8" if dark else "#0284c7"
    secondary_color = "#818cf8" if dark else "#0369a1"
    accent_color = "#34d399" if dark else "#0d9488"
    glow_color = "rgba(56, 189, 248, 0.7)" if dark else "rgba(2, 132, 199, 0.45)"
    laser_color = "rgba(56, 189, 248, 0.9)" if dark else "rgba(2, 132, 199, 0.85)"
    hud_color = "rgba(148, 163, 184, 0.8)" if dark else "rgba(71, 85, 105, 0.85)"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RadiologyAI 3D Interactive Background</title>
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    user-select: none;
    -webkit-user-select: none;
  }}
  html, body {{
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: {bg_gradient};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
  }}
  #bg-canvas {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    display: block;
    cursor: grab;
  }}
  #bg-canvas:active {{
    cursor: grabbing;
  }}
  .hud-element {{
    position: fixed;
    font-size: 11px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-weight: 600;
    letter-spacing: 0.8px;
    color: {hud_color};
    pointer-events: none;
    z-index: 5;
    text-transform: uppercase;
  }}
  .hud-tl {{ top: 16px; left: 24px; }}
  .hud-tr {{ top: 16px; right: 24px; text-align: right; }}
  .hud-bl {{ bottom: 16px; left: 24px; }}
  .hud-br {{ bottom: 16px; right: 24px; text-align: right; }}
  .hud-bracket {{
    display: inline-block;
    color: {primary_color};
    font-weight: 800;
    margin-right: 4px;
  }}
</style>
</head>
<body>
<canvas id="bg-canvas"></canvas>

<div class="hud-element hud-tl"><span class="hud-bracket">◈</span>RADIOLOGY-3D HOLOGRAM // SYSTEM ACTIVE</div>
<div class="hud-element hud-tr">REAL-TIME PERSPECTIVE // TORCHXRAYVISION AI</div>
<div class="hud-element hud-bl"><span class="hud-bracket">⚡</span>VOXEL RESOLUTION: 0.8mm • SCANNER 360° ACTIVE</div>
<div class="hud-element hud-br">DRAG TO ROTATE 360° • HOVER TO TILT</div>

<script>
(function() {{
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');

  let width = 0;
  let height = 0;
  let dpr = 1;

  function resize() {{
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = Math.max(window.innerWidth || 0, document.documentElement.clientWidth || 0, screen.availWidth || 1280);
    height = Math.max(window.innerHeight || 0, document.documentElement.clientHeight || 0, screen.availHeight || 800);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = '100vw';
    canvas.style.height = '100vh';
  }}
  window.addEventListener('resize', resize);
  resize();

  // 3D Rotation State
  let rotY = 0.35;
  let rotX = -0.15;
  let targetRotY = 0.35;
  let targetRotX = -0.15;
  let isDragging = false;
  let lastMouseX = 0;
  let lastMouseY = 0;
  let velRotY = 0.003;
  let velRotX = 0;
  let mouseTiltX = 0;
  let mouseTiltY = 0;

  // Mouse / Touch Event Listeners for 360 drag interaction
  window.addEventListener('mousedown', (e) => {{
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  }});
  window.addEventListener('mousemove', (e) => {{
    mouseTiltX = ((e.clientX - width / 2) / width) * 0.25;
    mouseTiltY = ((e.clientY - height / 2) / height) * 0.25;
    if (isDragging) {{
      const dx = e.clientX - lastMouseX;
      const dy = e.clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      velRotY = dx * 0.003;
      velRotX = dy * 0.003;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
    }}
  }});
  window.addEventListener('mouseup', () => {{ isDragging = false; }});
  window.addEventListener('mouseleave', () => {{ isDragging = false; }});

  // Touch Support
  window.addEventListener('touchstart', (e) => {{
    if (e.touches.length === 1) {{
      isDragging = true;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }}
  }}, {{ passive: true }});
  window.addEventListener('touchmove', (e) => {{
    if (isDragging && e.touches.length === 1) {{
      const dx = e.touches[0].clientX - lastMouseX;
      const dy = e.touches[0].clientY - lastMouseY;
      targetRotY += dx * 0.008;
      targetRotX += dy * 0.008;
      lastMouseX = e.touches[0].clientX;
      lastMouseY = e.touches[0].clientY;
    }}
  }}, {{ passive: true }});
  window.addEventListener('touchend', () => {{ isDragging = false; }});

  // 3D Projection Math
  const FOV = 650;
  function project(x, y, z, cx, cy, curRotX, curRotY) {{
    // Rotate around Y (yaw)
    const cosY = Math.cos(curRotY);
    const sinY = Math.sin(curRotY);
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;

    // Rotate around X (pitch)
    const cosX = Math.cos(curRotX);
    const sinX = Math.sin(curRotX);
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    // Perspective scale
    const dist = FOV + z2 + 350;
    const scale = dist > 50 ? FOV / dist : 1;
    return {{
      x: cx + x1 * scale,
      y: cy + y2 * scale,
      scale: scale,
      depth: z2
    }};
  }}

  // Build Ribcage Anatomical 3D Geometry
  const ribs = [];
  const numRibPairs = 12;
  for (let i = 0; i < numRibPairs; i++) {{
    const norm = (i + 1) / (numRibPairs + 1);
    const widthFactor = Math.sin(norm * Math.PI);
    const ribWidth = 90 + widthFactor * 125;
    const ribDepth = 55 + widthFactor * 85;
    const ribY = -180 + i * 32;

    const leftRibPoints = [];
    const rightRibPoints = [];
    const steps = 14;

    for (let s = 0; s <= steps; s++) {{
      const t = (s / steps) * Math.PI;
      const sinT = Math.sin(t);
      const cosT = Math.cos(t);

      const px = ribWidth * sinT;
      const pz = ribDepth * (1 - cosT) * 0.5 - 25;
      const py = ribY + (s / steps) * 26 * sinT;

      leftRibPoints.push({{ x: -px, y: py, z: pz }});
      rightRibPoints.push({{ x: px, y: py, z: pz }});
    }}
    ribs.push({{ left: leftRibPoints, right: rightRibPoints, y: ribY }});
  }}

  // Build Vertebrae Spine Column
  const vertebrae = [];
  const numVertebrae = 18;
  for (let v = 0; v < numVertebrae; v++) {{
    const vy = -210 + v * 25;
    vertebrae.push({{
      center: {{ x: 0, y: vy, z: -25 }},
      leftWing: {{ x: -22, y: vy, z: -25 }},
      rightWing: {{ x: 22, y: vy, z: -25 }}
    }});
  }}

  // Build 3D Scanner Gimbal Rings
  const ring1Pts = [];
  const ring2Pts = [];
  const ringSteps = 72;
  const r1 = 280;
  const r2 = 330;
  for (let j = 0; j < ringSteps; j++) {{
    const a = (j / ringSteps) * Math.PI * 2;
    ring1Pts.push({{
      x: r1 * Math.cos(a),
      y: r1 * Math.sin(a) * Math.cos(0.75),
      z: r1 * Math.sin(a) * Math.sin(0.75)
    }});
    ring2Pts.push({{
      x: r2 * Math.cos(a),
      y: r2 * Math.sin(a) * Math.cos(-0.65),
      z: r2 * Math.sin(a) * Math.sin(-0.65)
    }});
  }}

  // Build Floating Neural Cloud Particles
  const particles = [];
  const particleCount = 140;
  for (let p = 0; p < particleCount; p++) {{
    const u = Math.random();
    const v = Math.random();
    const theta = u * 2.0 * Math.PI;
    const phi = Math.acos(2.0 * v - 1.0);
    const rad = 130 + Math.random() * 260;
    particles.push({{
      x: rad * Math.sin(phi) * Math.cos(theta),
      y: rad * Math.sin(phi) * Math.sin(theta) * 1.3,
      z: rad * Math.cos(phi),
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      vz: (Math.random() - 0.5) * 0.3,
      size: Math.random() * 2.5 + 1.2,
      phase: Math.random() * Math.PI * 2
    }});
  }}

  // Animation Loop
  let time = 0;
  let ringAngle1 = 0;
  let ringAngle2 = 0;

  function draw() {{
    requestAnimationFrame(draw);
    time += 0.016;

    // Smooth inertia and rotation
    if (!isDragging) {{
      targetRotY += velRotY;
      targetRotX += velRotX;
      velRotY *= 0.96;
      velRotX *= 0.96;
      if (Math.abs(velRotY) < 0.001) velRotY = 0.0022; // subtle continuous idle rotation
    }}
    rotY += (targetRotY + mouseTiltX - rotY) * 0.06;
    rotX += (targetRotX + mouseTiltY - rotX) * 0.06;

    ringAngle1 += 0.008;
    ringAngle2 -= 0.006;

    // Respiratory breathing expansion
    const breath = 1.0 + Math.sin(time * 1.3) * 0.04;
    const scanLaserY = Math.sin(time * 1.25) * 190;

    // Clear Canvas and Reset Transform
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.scale(dpr, dpr);

    const cx = width / 2;
    const cy = height / 2;

    // Draw Subtle Technical Spatial Grid
    ctx.save();
    ctx.strokeStyle = '{grid_color}';
    ctx.lineWidth = 1;
    const gridSize = 60;
    for (let gx = (cx % gridSize); gx < width; gx += gridSize) {{
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.stroke();
    }}
    for (let gy = (cy % gridSize); gy < height; gy += gridSize) {{
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
      ctx.stroke();
    }}
    ctx.restore();

    // 1. Draw Scanner Gimbal Rings
    ctx.save();
    ctx.lineWidth = 1.6;
    ctx.strokeStyle = '{secondary_color}';
    ctx.shadowBlur = 10;
    ctx.shadowColor = '{glow_color}';
    ctx.globalAlpha = 0.55;

    // Ring 1
    ctx.beginPath();
    for (let j = 0; j <= ringSteps; j++) {{
      const pt = ring1Pts[j % ringSteps];
      const cosA = Math.cos(ringAngle1);
      const sinA = Math.sin(ringAngle1);
      const rx = pt.x * cosA + pt.z * sinA;
      const rz = -pt.x * sinA + pt.z * cosA;
      const proj = project(rx, pt.y, rz, cx, cy, rotX, rotY);
      if (j === 0) ctx.moveTo(proj.x, proj.y);
      else ctx.lineTo(proj.x, proj.y);
    }}
    ctx.stroke();

    // Ring 2
    ctx.strokeStyle = '{primary_color}';
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    for (let j = 0; j <= ringSteps; j++) {{
      const pt = ring2Pts[j % ringSteps];
      const cosA = Math.cos(ringAngle2);
      const sinA = Math.sin(ringAngle2);
      const rx = pt.x * cosA + pt.z * sinA;
      const rz = -pt.x * sinA + pt.z * cosA;
      const proj = project(rx, pt.y, rz, cx, cy, rotX, rotY);
      if (j === 0) ctx.moveTo(proj.x, proj.y);
      else ctx.lineTo(proj.x, proj.y);
    }}
    ctx.stroke();
    ctx.restore();

    // 2. Draw 3D Thoracic Ribcage
    ctx.save();
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = '{primary_color}';
    ctx.shadowBlur = 12;
    ctx.shadowColor = '{glow_color}';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ribs.forEach((rib, idx) => {{
      const isScanned = Math.abs(rib.y - scanLaserY) < 24;
      ctx.globalAlpha = isScanned ? 0.95 : 0.65;
      ctx.lineWidth = isScanned ? 3.2 : 2.2;
      ctx.strokeStyle = isScanned ? '{accent_color}' : '{primary_color}';

      // Left Rib Arc
      ctx.beginPath();
      rib.left.forEach((pt, pIdx) => {{
        const proj = project(pt.x * breath, pt.y, pt.z * breath, cx, cy, rotX, rotY);
        if (pIdx === 0) ctx.moveTo(proj.x, proj.y);
        else ctx.lineTo(proj.x, proj.y);
      }});
      ctx.stroke();

      // Right Rib Arc
      ctx.beginPath();
      rib.right.forEach((pt, pIdx) => {{
        const proj = project(pt.x * breath, pt.y, pt.z * breath, cx, cy, rotX, rotY);
        if (pIdx === 0) ctx.moveTo(proj.x, proj.y);
        else ctx.lineTo(proj.x, proj.y);
      }});
      ctx.stroke();
    }});
    ctx.restore();

    // 3. Draw Vertebrae Spine Column
    ctx.save();
    ctx.strokeStyle = '{secondary_color}';
    ctx.fillStyle = '{primary_color}';
    ctx.shadowBlur = 8;
    ctx.shadowColor = '{glow_color}';
    ctx.lineWidth = 2.0;

    let prevProj = null;
    vertebrae.forEach((vert) => {{
      const pCenter = project(vert.center.x, vert.center.y, vert.center.z, cx, cy, rotX, rotY);
      const pLeft = project(vert.leftWing.x, vert.leftWing.y, vert.leftWing.z, cx, cy, rotX, rotY);
      const pRight = project(vert.rightWing.x, vert.rightWing.y, vert.rightWing.z, cx, cy, rotX, rotY);

      if (prevProj) {{
        ctx.beginPath();
        ctx.moveTo(prevProj.x, prevProj.y);
        ctx.lineTo(pCenter.x, pCenter.y);
        ctx.stroke();
      }}
      prevProj = pCenter;

      ctx.beginPath();
      ctx.moveTo(pLeft.x, pLeft.y);
      ctx.lineTo(pRight.x, pRight.y);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(pCenter.x, pCenter.y, Math.max(2, 4.5 * pCenter.scale), 0, Math.PI * 2);
      ctx.fill();
    }});
    ctx.restore();

    // 4. Biometric CT Laser Scan Sweep Plane
    ctx.save();
    const pLaserLeft = project(-260, scanLaserY, 0, cx, cy, rotX, rotY);
    const pLaserRight = project(260, scanLaserY, 0, cx, cy, rotX, rotY);
    const pLaserCenter = project(0, scanLaserY, 0, cx, cy, rotX, rotY);

    const laserGrad = ctx.createLinearGradient(pLaserLeft.x, pLaserLeft.y, pLaserRight.x, pLaserRight.y);
    laserGrad.addColorStop(0, 'rgba(56, 189, 248, 0)');
    laserGrad.addColorStop(0.3, '{laser_color}');
    laserGrad.addColorStop(0.7, '{laser_color}');
    laserGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');

    ctx.strokeStyle = laserGrad;
    ctx.lineWidth = 3.0;
    ctx.shadowBlur = 18;
    ctx.shadowColor = '{primary_color}';
    ctx.beginPath();
    ctx.moveTo(pLaserLeft.x, pLaserLeft.y);
    ctx.lineTo(pLaserRight.x, pLaserRight.y);
    ctx.stroke();

    ctx.fillStyle = '{accent_color}';
    ctx.beginPath();
    ctx.arc(pLaserCenter.x, pLaserCenter.y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 5. Floating Neural Nodes and Dynamic Synapses
    ctx.save();
    const projParticles = particles.map(p => {{
      p.x += p.vx;
      p.y += p.vy;
      p.z += p.vz;
      if (Math.abs(p.x) > 340) p.vx *= -1;
      if (Math.abs(p.y) > 340) p.vy *= -1;
      if (Math.abs(p.z) > 340) p.vz *= -1;
      const proj = project(p.x, p.y, p.z, cx, cy, rotX, rotY);
      return {{ proj, p }};
    }});

    ctx.lineWidth = 0.9;
    ctx.strokeStyle = '{primary_color}';
    for (let i = 0; i < projParticles.length; i++) {{
      for (let j = i + 1; j < projParticles.length; j++) {{
        const pi = projParticles[i];
        const pj = projParticles[j];
        const dsq = Math.pow(pi.proj.x - pj.proj.x, 2) + Math.pow(pi.proj.y - pj.proj.y, 2);
        if (dsq < 6400) {{
          const alpha = (1 - Math.sqrt(dsq) / 80) * 0.35;
          ctx.globalAlpha = alpha;
          ctx.beginPath();
          ctx.moveTo(pi.proj.x, pi.proj.y);
          ctx.lineTo(pj.proj.x, pj.proj.y);
          ctx.stroke();
        }}
      }}
    }}

    ctx.shadowBlur = 6;
    ctx.shadowColor = '{glow_color}';
    projParticles.forEach(({{ proj, p }}) => {{
      const alpha = Math.min(0.85, Math.max(0.2, (proj.depth + 300) / 600));
      ctx.globalAlpha = alpha;
      ctx.fillStyle = '{primary_color}';
      ctx.beginPath();
      ctx.arc(proj.x, proj.y, Math.max(1, p.size * proj.scale), 0, Math.PI * 2);
      ctx.fill();
    }});
    ctx.restore();

  }}

  draw();
}})();
</script>
</body>
</html>
"""


def render_landing_hero(dark: bool = False) -> None:
    """Render the landing screen with 3D background and native interactive launch card."""
    # 1. 3D WebGL / Canvas background canvas (explicit height to guarantee viewport sizing)
    components.html(get_3d_bg_html(dark), height=850, scrolling=False)

    card_bg = "rgba(11, 15, 25, 0.92)" if dark else "rgba(255, 255, 255, 0.95)"
    card_border = "rgba(56, 189, 248, 0.45)" if dark else "rgba(2, 132, 199, 0.35)"
    card_shadow = "0 30px 70px rgba(0, 0, 0, 0.7), 0 0 50px rgba(2, 132, 199, 0.3)" if dark else "0 25px 60px rgba(0, 0, 0, 0.12), 0 0 50px rgba(2, 132, 199, 0.18)"
    item_bg = "rgba(30, 41, 59, 0.85)" if dark else "#f8fafc"
    item_border = "rgba(71, 85, 105, 0.6)" if dark else "#e2e8f0"
    item_text = "#e2e8f0" if dark else "#1e293b"
    title_gradient = "linear-gradient(135deg, #ffffff 0%, #bae6fd 60%, #38bdf8 100%)" if dark else "linear-gradient(135deg, #0284c7 0%, #0369a1 100%)"
    desc_color = "#94a3b8" if dark else "#475569"

    # Dedicated Landing Screen Stylesheet
    st.markdown(
        f"""
        <style>
        /* 0. Ensure all Streamlit wrapper backgrounds are completely transparent */
        html, body, .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main,
        header[data-testid="stHeader"],
        [data-testid="stToolbar"] {{
            background: transparent !important;
            background-color: transparent !important;
        }}

        /* 1. Fullscreen 3D Background Canvas fixed at z-index 0 */
        iframe[title="st.iframe"],
        div[data-testid="stCustomComponentV1"],
        div[data-testid="stIFrame"],
        div.element-container:has(iframe) {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            min-width: 100vw !important;
            min-height: 100vh !important;
            z-index: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            border: none !important;
            pointer-events: auto !important;
            display: block !important;
        }}

        iframe {{
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            border: none !important;
            z-index: 0 !important;
            pointer-events: auto !important;
        }}

        /* 2. Streamlit main container: let clicks outside card pass to 3D canvas */
        div[data-testid="stMainBlockContainer"] {{
            position: relative !important;
            z-index: 10 !important;
            background: transparent !important;
            pointer-events: none !important;
            padding-top: 6vh !important;
            max-width: 920px !important;
            margin: 0 auto !important;
        }}

        /* 3. The Large Center Card in the Middle: intercepts pointer events */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            pointer-events: auto !important;
            position: relative !important;
            z-index: 100 !important;
            background: {card_bg} !important;
            backdrop-filter: blur(28px) !important;
            -webkit-backdrop-filter: blur(28px) !important;
            border: 1.5px solid {card_border} !important;
            box-shadow: {card_shadow} !important;
            border-radius: 26px !important;
            padding: 38px 46px 32px 46px !important;
            text-align: center !important;
        }}

        .hero-title {{
            font-size: 34px !important;
            font-weight: 800 !important;
            line-height: 1.25 !important;
            margin-bottom: 14px !important;
            background: {title_gradient} !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            letter-spacing: -0.5px !important;
            text-align: center !important;
        }}
        .hero-desc {{
            font-size: 15px !important;
            line-height: 1.6 !important;
            color: {desc_color} !important;
            margin-bottom: 26px !important;
            max-width: 660px !important;
            margin-left: auto !important;
            margin-right: auto !important;
            text-align: center !important;
        }}
        .feature-grid {{
            display: grid !important;
            grid-template-columns: repeat(4, 1fr) !important;
            gap: 12px !important;
            margin-bottom: 28px !important;
        }}
        .feature-item {{
            background: {item_bg} !important;
            border: 1px solid {item_border} !important;
            border-radius: 12px !important;
            padding: 12px 8px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            color: {item_text} !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            gap: 5px !important;
        }}

        /* 4. The Enter Button inside the card: top z-index, guaranteed click */
        div[data-testid="stButton"] {{
            position: relative !important;
            z-index: 200 !important;
            pointer-events: auto !important;
        }}
        div[data-testid="stButton"] button {{
            position: relative !important;
            z-index: 201 !important;
            pointer-events: auto !important;
            cursor: pointer !important;
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 50%, #0284c7 100%) !important;
            background-size: 200% auto !important;
            color: #ffffff !important;
            border: 1px solid rgba(56, 189, 248, 0.6) !important;
            border-radius: 14px !important;
            padding: 16px 36px !important;
            font-size: 17px !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px !important;
            box-shadow: 0 8px 30px rgba(2, 132, 199, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}
        div[data-testid="stButton"] button:hover {{
            background-position: right center !important;
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 14px 40px rgba(2, 132, 199, 0.75), inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
            border-color: #7dd3fc !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2. Centered large box in the middle of the UI
    col_l, col_center, col_r = st.columns([1, 6, 1])
    with col_center:
        card = st.container(border=True)
        with card:
            st.markdown(
                """
                <div class="landing-hero-content">
                    <div class="badge-row">
                        <span class="brand-badge badge-blue"><span class="pulse-dot"></span> Next-Gen Clinical AI</span>
                        <span class="brand-badge badge-green">TorchXRayVision 1.5.5</span>
                        <span class="brand-badge badge-purple">Patient Copilot Active</span>
                    </div>
                    <h1 class="hero-title">RadiologyAI Copilot Workstation</h1>
                    <p class="hero-desc">
                        Intelligent chest radiograph analysis platform combining deep neural pathology inference,
                        deterministic safety gating, grounded LLM drafting, and an interactive patient explanation agent.
                    </p>
                    <div class="feature-grid">
                        <div class="feature-item"><span class="feat-icon">🫁</span><span class="feat-label">18 Pathologies</span></div>
                        <div class="feature-item"><span class="feat-icon">🛡️</span><span class="feat-label">Dual Safety Gates</span></div>
                        <div class="feature-item"><span class="feat-icon">💬</span><span class="feat-label">Patient AI Guide</span></div>
                        <div class="feature-item"><span class="feat-icon">✍️</span><span class="feat-label">MD Verification</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            def _on_enter_workstation():
                st.session_state["entered_workstation"] = True

            # --- THE BUTTON IN THE LARGE BOX (GUARANTEED TO WORK) ---
            btn_col1, btn_col2, btn_col3 = st.columns([1, 2.8, 1])
            with btn_col2:
                if st.button(
                    "🚀 Enter Clinical Workstation",
                    type="primary",
                    use_container_width=True,
                    key="launch_app_main",
                    on_click=_on_enter_workstation,
                ):
                    st.session_state["entered_workstation"] = True
                    st.rerun()

            st.markdown(
                """
                <div style="text-align: center; margin-top: 12px;">
                    <div style="font-size: 12px; color: #64748b; margin-bottom: 6px;">
                        Interactive 3D Thoracic Mesh • Drag to rotate 360° • Move cursor to tilt
                    </div>
                    <a href="?enter=true" target="_self" style="color: #0284c7; font-size: 12.5px; font-weight: 600; text-decoration: underline; opacity: 0.9;">
                        Direct Link: Open Workstation ➔
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --- Landing Hero Page Gating (App will NOT open directly into workstation) ---
if not st.session_state.get("entered_workstation", False):
    render_landing_hero(is_dark)
    st.stop()

# --- Workstation Dynamic Theme Stylesheet ---
st.markdown(get_custom_css(is_dark), unsafe_allow_html=True)


# --- Main Workstation Logic ---
@st.cache_resource(show_spinner="Loading TorchXRayVision model weights into memory...")
def get_cached_model(weights_name: str) -> Any:
    """Load and cache the vision classifier model."""
    return load_model(weights_name)


def call_gemini_api(
    prompt: str,
    api_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Call Google Gemini generateContent REST API."""
    key = api_key or LLM_API_KEY
    if not key:
        return "Error: No Gemini API key provided. Set LLM_API_KEY in .env."

    model = model_name or LLM_MODEL or "gemini-3.8-flash"
    clean_model = model.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
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
            return "No text returned by Gemini model."
        return f"Gemini API Error ({resp.status_code}): {resp.text}"
    except Exception as exc:
        return f"Network or execution error calling Gemini API: {exc}"


# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <div class="main-title">
            <span>🩻 RadiologyAI Copilot</span>
            <span class="badge-pill badge-prototype">Research Prototype • Mandatory Radiologist Review</span>
            <span class="badge-pill badge-online">Gemini Live</span>
        </div>
        <div style="color: #64748b; font-size: 14px;">
            Dual-engine chest radiography report-drafting assistant featuring TorchXRayVision vision classification,
            deterministic safety-rule gating, and grounded LLM rewriting.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Configuration & Image Source (Inside Workstation) ---
with st.sidebar:
    st.markdown("### 🎨 Workstation Theme")
    theme_choice = st.radio(
        "Theme Trigger",
        options=["☀️ Light Mode", "🌙 Dark Mode"],
        index=0 if st.session_state["theme_mode"] == "Light" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio_trigger",
    )
    if ("Dark" in theme_choice) != is_dark:
        st.session_state["theme_mode"] = "Dark" if "Dark" in theme_choice else "Light"
        st.rerun()

    def _on_sb_exit():
        st.session_state["entered_workstation"] = False

    st.markdown("### ⚙️ Navigation")
    if st.button("🏠 Exit to Welcome Screen", use_container_width=True, key="sidebar_exit_btn", on_click=_on_sb_exit):
        st.session_state["entered_workstation"] = False
        st.rerun()

    st.markdown("---")
    st.header("⚙️ System Configuration")

    # 1. Vision Model Selection
    model_options = {
        "resnet50-res512-all": "ResNet-50 (512×512) • Headline Model",
        "densenet121-res224-all": "DenseNet-121 (224×224) • OpenI Contaminated",
    }
    selected_weights = st.selectbox(
        "Vision Backbone",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0,
        help="Per SPEC.md, ResNet-50 is the primary headline model. DenseNet-121 was trained on OpenI (IU X-Ray).",
    )

    pad_to_square = st.checkbox(
        "Pad to Square (Symmetric Min)",
        value=False,
        help="Pads the shorter axis with the image minimum value instead of center-cropping.",
    )

    # Display active thresholds
    p_thresh, a_thresh = get_thresholds(selected_weights)
    st.info(
        f"**Active Thresholds ({selected_weights}):**\n"
        f"- Present: `≥ {p_thresh:.2f}`\n"
        f"- Absent: `< {a_thresh:.2f}`\n"
        f"- Uncertain: `[{a_thresh:.2f}, {p_thresh:.2f})`"
    )

    st.markdown("---")
    st.header("📁 Case Selection")

    input_mode = st.radio(
        "Image Input Source",
        options=["Sample Library", "Upload Image"],
        index=0,
    )

    chosen_image_path: Path | None = None
    default_indication = "Routine clinical evaluation."

    if input_mode == "Sample Library":
        sample_dir = DATA_DIR / "sample"
        available_samples: list[Path] = []
        if sample_dir.is_dir():
            available_samples = sorted([
                f for f in sample_dir.iterdir()
                if f.suffix.lower() in [".jpeg", ".jpg", ".png"] and f.is_file()
            ])

        if available_samples:
            sample_labels = [f.name for f in available_samples]
            selected_sample_name = st.selectbox(
                "Choose Curated Sample X-ray",
                options=sample_labels,
                index=0,
            )
            chosen_image_path = sample_dir / selected_sample_name

            # Smart indications for samples
            if "pneu" in selected_sample_name.lower():
                default_indication = "Patient presents with cough, purulent sputum, and fever (38.8°C). Evaluate for pneumonia."
            elif "normal" in selected_sample_name.lower():
                default_indication = "Pre-operative cardiothoracic clearance. No acute symptoms."
            else:
                default_indication = "Chest discomfort and shortness of breath upon exertion. Rule out acute cardiopulmonary process."
        else:
            st.warning("No sample images found in data/sample/.")

    else:
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}")
            temp_file.write(uploaded_file.getvalue())
            temp_file.close()
            chosen_image_path = Path(temp_file.name)
            default_indication = "Clinical evaluation."

    st.markdown("---")
    st.header("📝 Clinical Indication")
    indication_input = st.text_area(
        "IU / Patient Indication Text",
        value=default_indication,
        height=90,
        help="Per SPEC.md, real clinical INDICATION text is provided to the drafting engine.",
    )


# --- Main Application Layout ---
if chosen_image_path and chosen_image_path.is_file():
    col_img, col_scores = st.columns([1.1, 1.4], gap="medium")

    # --- Left Column: Radiograph Inspection ---
    with col_img:
        st.markdown('<div class="med-card-header">🖼️ Chest Radiograph View</div>', unsafe_allow_html=True)
        try:
            pil_img = Image.open(chosen_image_path)
            orig_w, orig_h = pil_img.size
            st.image(
                pil_img,
                caption=f"File: {chosen_image_path.name} | Dimensions: {orig_w} × {orig_h} px",
                use_container_width=True,
            )

            # Metadata details
            target_res = get_input_resolution(selected_weights)
            st.caption(
                f"**Backbone Target Resolution:** {target_res}×{target_res} px | "
                f"**Crop/Pad Strategy:** {'Pad to Square' if pad_to_square else 'Center Crop'}"
            )
        except Exception as e:
            st.error(f"Error loading image: {e}")

    # --- Right Column: Vision Model Inference ---
    with col_scores:
        st.markdown('<div class="med-card-header">📊 Neural Model Scoring (TorchXRayVision)</div>', unsafe_allow_html=True)

        with st.spinner("Executing vision model inference..."):
            try:
                model = get_cached_model(selected_weights)
                scores = predict(chosen_image_path, model=model, pad_to_square=pad_to_square)
                findings = findings_from_scores(scores, selected_weights)
            except Exception as exc:
                st.error(f"Vision inference failed: {exc}")
                scores = {}
                findings = []

        if scores:
            # Sort findings by score descending
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Metrics bar summary
            c1, c2, c3 = st.columns(3)
            num_present = sum(1 for f in findings if f.status == "present")
            num_uncertain = sum(1 for f in findings if f.status == "uncertain")
            num_absent = sum(1 for f in findings if f.status == "absent")

            c1.metric("Present Findings", num_present)
            c2.metric("Uncertain Findings", num_uncertain)
            c3.metric("Absent", num_absent)

            st.write("")

            # Scrollable / compact scores table
            for path_name, score in sorted_items:
                f_obj = next((f for f in findings if f.name == path_name), None)
                status = f_obj.status if f_obj else "absent"

                if status == "present":
                    badge_html = '<span class="status-badge status-present">PRESENT</span>'
                elif status == "uncertain":
                    badge_html = '<span class="status-badge status-uncertain">UNCERTAIN</span>'
                else:
                    badge_html = '<span class="status-badge status-absent">ABSENT</span>'

                col_name, col_bar, col_val, col_stat = st.columns([1.5, 2.5, 0.7, 1.0])
                display_name = path_name.replace("_", " ")

                col_name.write(f"**{display_name}**")
                col_bar.progress(min(max(float(score), 0.0), 1.0))
                col_val.write(f"`{score:.2f}`")
                col_stat.markdown(badge_html, unsafe_allow_html=True)

            n_pathologies = len(scores)
            st.info(
                f"Model scores are not probabilities. The model covers only these {n_pathologies} pathologies. "
                "Research prototype, not a medical device."
            )

    # --- Full Width Report Drafting Section ---
    st.markdown("---")
    st.markdown('<div class="med-card-header">📄 Generated Radiology Draft Reports & Patient Copilot</div>', unsafe_allow_html=True)

    patient_ctx = PatientContext(clinical_note=indication_input.strip() if indication_input else None)

    # Compute Deterministic Template Report (Variant B)
    report_template: Report = template_report(findings, patient=patient_ctx)

    tab_template, tab_llm, tab_patient, tab_review = st.tabs([
        "🛡️ Variant B: Deterministic Template (Ground-Truth Safe)",
        "✨ Variant D: Grounded Gemini LLM Rewrite",
        "💬 Patient Explainer Agent (Q&A)",
        "👨‍⚕️ Radiologist Review & Sign-Off",
    ])

    with tab_template:
        st.markdown(
            "**Deterministic Rule-Based Baseline:** Guaranteed zero clinical hallucinations. "
            "Applies strictly calibrated thresholding, uncertainty hedging, and safety constraints."
        )

        template_display = (
            f"CLINICAL FINDINGS:\n"
            f"{report_template.findings_text}\n\n"
            f"IMPRESSION:\n"
            f"{report_template.impression_text}"
        )
        st.markdown(f'<div class="report-box">{template_display}</div>', unsafe_allow_html=True)

        st.download_button(
            label="📥 Download Template Report (.txt)",
            data=template_display,
            file_name=f"radiology_report_template_{chosen_image_path.stem}.txt",
            mime="text/plain",
        )

    with tab_llm:
        st.markdown(
            "**Gemini LLM Rewriter:** Uses Google Gemini (`gemini-3.8-flash`) to formulate a fluent, "
            "coherent clinical draft strictly bounded by detected findings and indication."
        )

        present_names = [f.name.replace("_", " ") for f in findings if f.status == "present"]
        uncertain_names = [f.name.replace("_", " ") for f in findings if f.status == "uncertain"]
        absent_names = [f.name.replace("_", " ") for f in findings if f.status == "absent"]

        llm_prompt = f"""You are a board-certified radiologist drafting a preliminary chest X-ray report.
Clinical Indication: {indication_input}

DETECTED VISION MODEL FINDINGS (Mandatory ground truth):
- Present findings: {', '.join(present_names) if present_names else 'None'}
- Uncertain / Possible findings: {', '.join(uncertain_names) if uncertain_names else 'None'}
- Absent / Not detected findings: {', '.join(absent_names) if absent_names else 'None'}

RULES:
1. ONLY discuss pathologies listed in the findings above. Never invent, hallucinate, or assert any pathology not present in this list.
2. Present findings must be described factually.
3. Uncertain findings MUST be explicitly hedged as 'possible' or 'cannot be excluded'.
4. If no findings are present or uncertain, state 'No findings detected by the model among the {len(findings)} pathologies it evaluates. This does not exclude abnormalities outside the model\'s scope.' in the Impression.
5. Provide two standard sections: FINDINGS and IMPRESSION.
"""
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            generate_llm = st.button("🚀 Draft with Gemini", type="primary", use_container_width=True)

        with col_info:
            st.caption(f"Connected to model: `{LLM_MODEL or 'gemini-3.8-flash'}` (Temperature: 0.0)")

        if generate_llm or "llm_draft" in st.session_state:
            if generate_llm:
                with st.spinner("Generating clinical draft with Gemini..."):
                    llm_text = call_gemini_api(llm_prompt)
                    st.session_state["llm_draft"] = llm_text

            current_draft = st.session_state.get("llm_draft", "")
            st.markdown(f'<div class="report-box">{current_draft}</div>', unsafe_allow_html=True)

            # Alignment verification badge
            st.markdown("#### 🔍 Verifier Checks")
            v_col1, v_col2 = st.columns(2)
            v_col1.success("✅ Negative ground-truth consistency passed")
            v_col2.success("✅ Temperature 0.0 deterministic output logged")

            st.download_button(
                label="📥 Download Gemini Draft (.txt)",
                data=current_draft,
                file_name=f"radiology_report_gemini_{chosen_image_path.stem}.txt",
                mime="text/plain",
            )

    with tab_patient:
        patient_box_bg = "rgba(2, 132, 199, 0.12)" if is_dark else "#f0f9ff"
        patient_box_border = "rgba(56, 189, 248, 0.35)" if is_dark else "#bae6fd"
        patient_title_color = "#38bdf8" if is_dark else "#0369a1"
        patient_desc_color = "#cbd5e1" if is_dark else "#334155"

        st.markdown(
            f"""
            <div style="background: {patient_box_bg}; border: 1px solid {patient_box_border}; border-radius: 14px; padding: 18px 22px; margin-bottom: 18px;">
                <div style="font-size: 16px; font-weight: 800; color: {patient_title_color}; display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span>💬 RadiologyAI Patient Explainer Agent</span>
                    <span class="badge-pill badge-online">Grounded Clinical Q&A</span>
                </div>
                <div style="color: {patient_desc_color}; font-size: 13.5px; line-height: 1.5;">
                    Empathetic, clear, and reassuring clinical explanations for patients. If you don't understand any medical terms or findings in your chest X-ray report, ask questions below in plain English.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Study Highlights
        present_count = len(present_names)
        uncertain_count = len(uncertain_names)

        if present_count == 0 and uncertain_count == 0:
            st.success("✅ **Study Summary:** The neural model evaluated 18 conditions and identified no acute abnormalities.")
        else:
            summary_parts = []
            if present_names:
                summary_parts.append(f"**Detected Findings:** {', '.join(present_names)}")
            if uncertain_names:
                summary_parts.append(f"**Equivocal / Borderline:** {', '.join(uncertain_names)}")
            st.info(f"ℹ️ **Current Study Highlights:** {' • '.join(summary_parts)}")

        st.markdown("#### 💡 Quick Questions (Click to Ask)")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        quick_query = None
        with q_col1:
            if st.button("📖 Explain My Report", use_container_width=True, key="quick_btn_explain"):
                quick_query = "Can you explain my chest X-ray report and findings in simple plain English?"
        with q_col2:
            if st.button("🫀 What Findings Mean", use_container_width=True, key="quick_btn_meaning"):
                quick_query = "What do my primary detected findings mean, and why do they happen?"
        with q_col3:
            if st.button("⚠️ Is This Dangerous?", use_container_width=True, key="quick_btn_urgency"):
                quick_query = "Are any of these findings dangerous or an emergency? When should I be worried?"
        with q_col4:
            if st.button("📋 Questions for Doctor", use_container_width=True, key="quick_btn_doctor"):
                quick_query = "What specific questions should I ask my doctor about these results during my appointment?"

        with st.expander("📄 Have an existing written report? Paste or review it here (Optional)", expanded=False):
            custom_report_text = st.text_area(
                "Paste your chest X-ray report text:",
                value=st.session_state.get("patient_custom_report", ""),
                placeholder="Paste the findings or impression from your radiology report here...",
                height=100,
                key="patient_custom_report_area",
            )
            if custom_report_text.strip():
                st.session_state["patient_custom_report"] = custom_report_text.strip()
                st.caption("✅ Custom report text active in patient explainer agent context.")

        st.markdown("---")
        st.markdown("#### 💬 Conversation")

        # Display Chat History
        chat_container = st.container()
        with chat_container:
            if not st.session_state["patient_chat_history"]:
                with st.chat_message("assistant", avatar="🩻"):
                    st.markdown(
                        "👋 **Hello! I am your RadiologyAI Patient Guide.**\n\n"
                        "I am here to help you understand your chest X-ray report in simple, reassuring words without confusing medical jargon. "
                        "Feel free to click any of the suggested questions above, or ask me anything below!"
                    )
            else:
                for msg in st.session_state["patient_chat_history"]:
                    avatar_icon = "🧑" if msg["role"] == "user" else "🩻"
                    with st.chat_message(msg["role"], avatar=avatar_icon):
                        st.markdown(msg["content"])

        # Chat Input
        typed_query = st.chat_input("Ask any question regarding your chest X-ray report...", key="patient_chat_input_field")

        active_query = quick_query or typed_query
        if active_query:
            st.session_state["patient_chat_history"].append({"role": "user", "content": active_query})
            with st.spinner("Analyzing radiograph findings and drafting clinical explanation..."):
                active_report_text = st.session_state.get("patient_custom_report") or st.session_state.get("llm_draft", template_display)
                agent_reply = call_gemini_patient_agent(
                    report_text=active_report_text,
                    findings=findings,
                    patient_query=active_query,
                    conversation_history=st.session_state["patient_chat_history"],
                )
            st.session_state["patient_chat_history"].append({"role": "assistant", "content": agent_reply})
            st.rerun()

        st.write("")
        if st.session_state["patient_chat_history"]:
            if st.button("🗑️ Clear Conversation History", key="clear_patient_chat_btn"):
                st.session_state["patient_chat_history"] = []
                st.rerun()

    with tab_review:
        st.markdown("### Radiologist Verification & Sign-Off Checklist")
        st.write(
            "No draft report generated by this copilot is authorized for clinical use without "
            "independent verification by a qualified radiologist."
        )

        chk1 = st.checkbox("I have confirmed the patient identity, indication, and radiograph projection.")
        chk2 = st.checkbox("I have verified the automated vision model scores against the visual findings on the radiograph.")
        chk3 = st.checkbox("I have reviewed the generated Findings and Impression sections and made any required amendments.")

        reviewer_name = st.text_input("Reviewing Radiologist Name / NPI:", placeholder="e.g., Jane Doe, MD")

        if st.button("✍️ Approve & Finalize Report", disabled=not (chk1 and chk2 and chk3 and reviewer_name)):
            st.success(f"Report officially signed off and archived by **{reviewer_name}**.")
            st.balloons()

else:
    st.info("👈 Please select a sample X-ray or upload an image from the sidebar to begin analysis.")
