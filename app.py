"""
RoadGuard AI - Autonomous Road Damage Detection & Severity Assessment
Powered by Google GenAI (Gemini 2.5 Flash), Pillow, and Streamlit.
RDD2022-Compatible Pavement Distress Classification (D00, D10, D20, D40).
"""

import os
import io
import re
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Configure page layout and visual identity
st.set_page_config(
    page_title="RoadGuard AI — RDD2022 Road Damage Detection & Severity Assessment",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Taxonomy & Constants: RDD2022-Compatible Classes
# ---------------------------------------------------------
RDD2022_CLASSES = {
    "D00": "D00 — Longitudinal Crack",
    "D10": "D10 — Transverse Crack",
    "D20": "D20 — Alligator Crack",
    "D40": "D40 — Pothole",
    "Other/Uncertain": "Other / Uncertain Distress"
}

SEVERITY_COLORS = {
    "HIGH": (239, 68, 68),       # Crimson Red
    "MEDIUM": (245, 158, 11),    # Amber Orange
    "LOW": (16, 185, 129),       # Emerald Green
    "OTHER": (14, 165, 233)      # Sky Blue
}

SEVERITY_HEX = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW": "#10b981",
    "OTHER": "#0ea5e9"
}

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0b1329 0%, #172554 50%, #1e1b4b 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff, #93c5fd, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        line-height: 1.5;
    }
    
    .badge-container {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .status-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 9999px;
        font-size: 0.75rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    
    /* Summary Stat Cards */
    .summary-bar {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1.2rem;
        flex-wrap: wrap;
    }
    
    .stat-card {
        flex: 1;
        min-width: 110px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    
    .stat-card.high {
        border-top: 3px solid #ef4444;
    }
    .stat-card.medium {
        border-top: 3px solid #f59e0b;
    }
    .stat-card.low {
        border-top: 3px solid #10b981;
    }
    .stat-card.total {
        border-top: 3px solid #3b82f6;
    }
    
    .stat-num {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.2;
    }
    
    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Defect Card */
    .defect-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.15rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    }
    
    .defect-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.6rem;
        margin-bottom: 0.8rem;
    }
    
    .defect-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    .rdd-badge {
        font-family: monospace;
        font-weight: 700;
        background: #0f172a;
        color: #60a5fa;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        border: 1px solid #3b82f6;
        font-size: 0.82rem;
    }
    
    .severity-badge-HIGH {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        color: #f87171;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
    }
    
    .severity-badge-MEDIUM {
        background: rgba(245, 158, 11, 0.18);
        border: 1px solid #f59e0b;
        color: #fbbf24;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
    }
    
    .severity-badge-LOW {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid #10b981;
        color: #34d399;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.6rem;
        margin-bottom: 0.75rem;
    }

    .detail-item {
        background: #0f172a;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    .detail-key {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .detail-val {
        font-size: 0.92rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-top: 0.15rem;
    }
    
    .action-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.75) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid #3b82f6;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin-top: 0.5rem;
    }
    
    .action-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    
    .action-text {
        font-size: 0.92rem;
        font-weight: 500;
        color: #f8fafc;
        line-height: 1.35;
    }

    .no-damage-card {
        background: rgba(16, 185, 129, 0.08);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 2.2rem;
        text-align: center;
        color: #e2e8f0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions: Normalization, Drawing & AI Pipeline
# ---------------------------------------------------------

def normalize_rdd2022_class(raw_class: str) -> str:
    """
    Map defects to one of the four RDD2022 target classes:
    - D00: Longitudinal Crack
    - D10: Transverse Crack
    - D20: Alligator Crack
    - D40: Pothole
    If not confidently matched, return 'Other/Uncertain'.
    """
    if not raw_class:
        return "Other/Uncertain"
    cleaned = str(raw_class).strip().upper()
    
    # Check exact codes first
    if cleaned in ("D00", "D10", "D20", "D40"):
        return cleaned
    
    # Check descriptions
    if "D00" in cleaned or "LONGITUDINAL" in cleaned:
        return "D00"
    elif "D10" in cleaned or "TRANSVERSE" in cleaned:
        return "D10"
    elif "D20" in cleaned or "ALLIGATOR" in cleaned or "FATIGUE" in cleaned:
        return "D20"
    elif "D40" in cleaned or "POTHOLE" in cleaned or "HOLE" in cleaned:
        return "D40"
    
    return "Other/Uncertain"


def normalize_severity(raw_sev: str) -> str:
    """Ensure severity is strictly LOW, MEDIUM, or HIGH."""
    cleaned = str(raw_sev or "").strip().upper()
    if "HIGH" in cleaned:
        return "HIGH"
    elif "LOW" in cleaned:
        return "LOW"
    return "MEDIUM"


def pil_to_part(pil_img: Image.Image):
    """Convert a PIL Image to a google.genai types.Part object."""
    from google.genai import types
    buf = io.BytesIO()
    if pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="JPEG", quality=92)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def draw_bounding_boxes(image: Image.Image, defects: list) -> Image.Image:
    """
    Draw normalized bounding boxes ([ymin, xmin, ymax, xmax] 0-1000) onto the image
    with high-contrast color coding by severity, clipping, and RDD2022 labels.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    W, H = annotated.size

    # Responsive font sizing
    font_size = max(13, min(22, int(H * 0.022)))
    try:
        font = ImageFont.load_default(size=font_size)
    except Exception:
        font = ImageFont.load_default()

    line_w = max(2, min(5, int(min(W, H) * 0.004)))

    for idx, defect in enumerate(defects, start=1):
        raw_box = defect.get("bounding_box", [])
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue

        try:
            ymin, xmin, ymax, xmax = [float(c) for c in raw_box]
        except (ValueError, TypeError):
            continue

        # Support both 0-1000 and 0.0-1.0 normalized formats
        if max(ymin, xmin, ymax, xmax) <= 1.0 and max(ymin, xmin, ymax, xmax) > 0:
            ymin *= 1000.0
            xmin *= 1000.0
            ymax *= 1000.0
            xmax *= 1000.0

        # Clamp to 0-1000 range
        ymin = max(0.0, min(1000.0, ymin))
        xmin = max(0.0, min(1000.0, xmin))
        ymax = max(0.0, min(1000.0, ymax))
        xmax = max(0.0, min(1000.0, xmax))

        # Ensure min < max
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        # Convert to pixel space
        py_min = int((ymin / 1000.0) * H)
        px_min = int((xmin / 1000.0) * W)
        py_max = int((ymax / 1000.0) * H)
        px_max = int((xmax / 1000.0) * W)

        # Strictly clip to image boundaries
        py_min = max(0, min(H - 1, py_min))
        px_min = max(0, min(W - 1, px_min))
        py_max = max(py_min + 3, min(H, py_max))
        px_max = max(px_min + 3, min(W, px_max))

        # Severity & RDD2022 attributes
        sev = normalize_severity(defect.get("severity"))
        color = SEVERITY_COLORS.get(sev, SEVERITY_COLORS["OTHER"])
        rdd_code = normalize_rdd2022_class(defect.get("rdd2022_class"))
        conf = float(defect.get("confidence", 0.0))
        conf_str = f" {int(conf * 100)}%" if conf > 0 else ""

        # Draw bounding rectangle
        draw.rectangle([px_min, py_min, px_max, py_max], outline=color, width=line_w)

        # Label text: e.g. "#1 [D40] HIGH 96%"
        label_text = f"#{idx} [{rdd_code}] {sev}{conf_str}"
        t_box = draw.textbbox((px_min, py_min), label_text, font=font)
        t_w = t_box[2] - t_box[0]
        t_h = t_box[3] - t_box[1]

        # Place label above box if space allows, otherwise inside
        label_ymin = py_min - t_h - 6
        if label_ymin < 0:
            label_ymin = py_min + 3
        label_ymax = label_ymin + t_h + 6
        label_xmax = min(W, px_min + t_w + 10)

        # Draw solid label background and text
        draw.rectangle([px_min, label_ymin, label_xmax, label_ymax], fill=color, outline=color)
        draw.text((px_min + 5, label_ymin + 2), label_text, fill=(255, 255, 255), font=font)

    return annotated


def extract_and_parse_json(response_text: str) -> dict:
    """
    Robust JSON parser: handles markdown fences, finds outermost JSON delimiters,
    and handles common minor JSON formatting anomalies.
    """
    clean_text = response_text.strip()
    
    # Strip markdown code blocks
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    clean_text = clean_text.strip()

    # Search for outermost JSON object { ... }
    match = re.search(r'(\{[\s\S]*\})', clean_text)
    if match:
        clean_text = match.group(1)

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Try cleaning trailing commas before closing braces
        clean_text = re.sub(r',\s*([\}\]])', r'\1', clean_text)
        return json.loads(clean_text)


def analyze_road_damage_gemini(image: Image.Image, api_key: str, model_name: str = "gemini-2.5-flash", road_context: str = ""):
    """
    Calls Google GenAI SDK using gemini-2.5-flash to perform structured RDD2022 visual road damage detection.
    Returns validated dict containing 'defects' list with bounding boxes and damage metrics.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    image_part = pil_to_part(image)

    system_prompt = """
You are an expert transportation infrastructure and civil computer vision engineer specializing in automated pavement distress inspection and RDD2022 (Road Damage Dataset 2022) taxonomy.

Examine the provided road surface image and identify all visible road-damage regions.

For each detected defect, you MUST output a structured entry containing:
- damage_type: Human-readable distress description (e.g., "Deep Pothole with Loose Aggregate", "Alligator Fatigue Cracking Network", "Longitudinal Wheel-Path Crack", "Transverse Joint Crack").
- rdd2022_class: Exactly one of the four RDD2022 target classes:
    * "D00" (Longitudinal Crack)
    * "D10" (Transverse Crack)
    * "D20" (Alligator Crack)
    * "D40" (Pothole)
    If a defect cannot confidently be mapped to one of these four classes, you MUST assign "Other/Uncertain" instead of forcing a classification.
- bounding_box: Normalized coordinates [ymin, xmin, ymax, xmax] as integers between 0 and 1000 representing the defect's boundary box.
- confidence: Detection confidence as a float between 0.0 and 1.0 (e.g., 0.94).
- severity: Distress severity level, MUST be strictly one of: "LOW", "MEDIUM", "HIGH" based on ASTM D6433 standards.
- surface_coverage_percent: Estimated percentage of the visible road surface covered by this specific defect (float, e.g. 15.5).
- maintenance_action: Clear, actionable engineering maintenance recommendation (e.g. "Full-depth hot-mix asphalt patching with tack coat and compaction", "Hot-poured rubberized crack sealant application", "Mill & 50mm asphalt overlay").

RELIABILITY & GROUND TRUTH RULES:
1. Do NOT invent or hallucinate detections when the image is unclear or shows sound, undamaged road pavement.
2. If no road damage is detected, return an empty defects list: "defects": [].
3. Respond strictly with a valid JSON object matching the schema below.

JSON Schema:
{
  "defects": [
    {
      "damage_type": string,
      "rdd2022_class": "D00" | "D10" | "D20" | "D40" | "Other/Uncertain",
      "bounding_box": [ymin, xmin, ymax, xmax],
      "confidence": float,
      "severity": "LOW" | "MEDIUM" | "HIGH",
      "surface_coverage_percent": float,
      "maintenance_action": string
    }
  ],
  "road_condition_summary": string,
  "overall_pavement_rating": "GOOD" | "FAIR" | "POOR" | "CRITICAL"
}
"""

    user_query = f"Analyze road damage regions in this image. Context: {road_context}. Provide structured RDD2022 detections with bounding boxes [ymin, xmin, ymax, xmax] 0-1000. Output strictly valid JSON."

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        system_instruction=system_prompt,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[image_part, user_query],
        config=config,
    )

    parsed = extract_and_parse_json(response.text)

    # Sanitize and validate defect items
    raw_defects = parsed.get("defects", [])
    sanitized_defects = []
    
    for d in raw_defects:
        if not isinstance(d, dict):
            continue
        
        rdd_class = normalize_rdd2022_class(d.get("rdd2022_class"))
        severity = normalize_severity(d.get("severity"))
        
        try:
            confidence = float(d.get("confidence", 0.85))
        except (ValueError, TypeError):
            confidence = 0.85
            
        try:
            coverage = float(d.get("surface_coverage_percent", 5.0))
        except (ValueError, TypeError):
            coverage = 5.0

        box = d.get("bounding_box", [0, 0, 1000, 1000])
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            box = [0, 0, 1000, 1000]

        sanitized_defects.append({
            "damage_type": str(d.get("damage_type", "Road Distress")),
            "rdd2022_class": rdd_class,
            "bounding_box": box,
            "confidence": confidence,
            "severity": severity,
            "surface_coverage_percent": coverage,
            "maintenance_action": str(d.get("maintenance_action", "Inspect and apply standard maintenance procedure"))
        })

    parsed["defects"] = sanitized_defects
    return parsed


def get_demo_rdd_assessment(sample_name: str, road_context: str = ""):
    """
    Calibrated RDD2022 assessment for instant offline demo mode,
    providing accurate bounding boxes and metrics for the bundled sample images.
    """
    sample_key = (sample_name or "").lower()
    
    if "alligator" in sample_key:
        return {
            "defects": [
                {
                    "damage_type": "Alligator (Fatigue) Cracking Network",
                    "rdd2022_class": "D20",
                    "bounding_box": [50, 120, 960, 890],
                    "confidence": 0.95,
                    "severity": "HIGH",
                    "surface_coverage_percent": 36.5,
                    "maintenance_action": "Mill & Inlay (50mm Asphalt Overlay) with sub-base stabilization"
                }
            ],
            "road_condition_summary": "Extensive alligator fatigue cracking along the travel lane indicating structural failure of the underlying base course.",
            "overall_pavement_rating": "POOR"
        }
    elif "crack" in sample_key or "longitudinal" in sample_key:
        return {
            "defects": [
                {
                    "damage_type": "Longitudinal Wheel-Path Crack",
                    "rdd2022_class": "D00",
                    "bounding_box": [220, 430, 985, 540],
                    "confidence": 0.94,
                    "severity": "MEDIUM",
                    "surface_coverage_percent": 12.4,
                    "maintenance_action": "Hot-Poured Rubberized Bituminous Crack Sealing with Routing"
                },
                {
                    "damage_type": "Transverse Branching Crack",
                    "rdd2022_class": "D10",
                    "bounding_box": [470, 210, 620, 440],
                    "confidence": 0.88,
                    "severity": "LOW",
                    "surface_coverage_percent": 3.2,
                    "maintenance_action": "Asphalt emulsion crack pour and blot with clean sand"
                }
            ],
            "road_condition_summary": "Continuous longitudinal crack in the primary wheel track with minor transverse branches. Crack sealing required to prevent water infiltration.",
            "overall_pavement_rating": "FAIR"
        }
    else:
        # Severe Pothole sample or general default
        return {
            "defects": [
                {
                    "damage_type": "Severe Asphalt Pothole Cavity",
                    "rdd2022_class": "D40",
                    "bounding_box": [235, 110, 845, 875],
                    "confidence": 0.97,
                    "severity": "HIGH",
                    "surface_coverage_percent": 24.0,
                    "maintenance_action": "Full-Depth Semi-Permanent Hot Mix Patching with Tack Coat and Mechanical Compaction"
                }
            ],
            "road_condition_summary": "Deep, hazardous bowl-shaped pothole exposing unbound base aggregate. Poses severe danger to tires and vehicle suspension.",
            "overall_pavement_rating": "CRITICAL"
        }


# ---------------------------------------------------------
# Streamlit UI: Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🛣️ RoadGuard AI — RDD2022 Road Damage & Severity Assessment</div>
    <div class="main-subtitle">
        Visual road distress detection and severity evaluation powered by <strong>Google Gemini 2.5 Flash</strong>. 
        Detects road defects, outputs localized bounding boxes, maps to <strong>RDD2022</strong> classes (D00, D10, D20, D40), 
        quantifies surface coverage, and prescribes civil engineering maintenance actions.
    </div>
    <div class="badge-container">
        <span class="status-chip">⚡ Gemini 2.5 Flash</span>
        <span class="status-chip">🎯 RDD2022 Compatible (D00, D10, D20, D40)</span>
        <span class="status-chip">📐 Visual Bounding Boxes</span>
        <span class="status-chip">⚠️ ASTM D6433 Severity Tiers</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Configuration (Requirement 1 & 2)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 Gemini API Configuration")
    
    env_api_key = os.environ.get("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Enter Gemini API Key",
        value=env_api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Enter your Google Gemini API key. If not provided, Demo Simulation Mode is available to test all features."
    )

    if api_key_input:
        st.success("🟢 API Key Configured")
    else:
        st.info("🟡 No API Key — Demo Mode available for testing.")
        st.markdown("[🔗 Get Free Gemini API Key](https://aistudio.google.com/app/apikey)")

    st.markdown("---")
    st.markdown("### ⚙️ Inspection Settings")
    model_choice = st.selectbox(
    "Vision Model",
    options=["gemini-3.8-flash"],
    index=0,
    help="Gemini 3.8 Flash is the primary multimodal model for road-damage inspection."
    )
    road_context_choice = st.selectbox(
        "Road Classification",
        options=[
            "Interstate / Expressway",
            "Urban Arterial Boulevard",
            "Residential / Municipal Street",
            "Rural Highway / Low Volume",
            "Industrial Route / Heavy Freight"
        ],
        index=1
    )

    st.markdown("---")
    st.markdown("### 🧪 Quick Sample Dataset")
    st.caption("Select a sample road image to test immediately:")

    sample_col1, sample_col2, sample_col3 = st.columns(3)

    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    pothole_path = os.path.join(sample_dir, "pothole_severe.jpg")
    alligator_path = os.path.join(sample_dir, "alligator_cracking.jpg")
    crack_path = os.path.join(sample_dir, "longitudinal_crack.jpg")

    if "current_image" not in st.session_state:
        if os.path.exists(pothole_path):
            st.session_state.current_image = Image.open(pothole_path)
            st.session_state.current_image_name = "pothole_severe.jpg (Sample 1)"
        else:
            st.session_state.current_image = None
            st.session_state.current_image_name = None

    if "assessment_result" not in st.session_state:
        st.session_state.assessment_result = None

    if sample_col1.button("🕳️ Pothole", use_container_width=True):
        if os.path.exists(pothole_path):
            st.session_state.current_image = Image.open(pothole_path)
            st.session_state.current_image_name = "pothole_severe.jpg (Sample 1)"
            st.session_state.assessment_result = None
            st.rerun()

    if sample_col2.button("🕸️ Alligator", use_container_width=True):
        if os.path.exists(alligator_path):
            st.session_state.current_image = Image.open(alligator_path)
            st.session_state.current_image_name = "alligator_cracking.jpg (Sample 2)"
            st.session_state.assessment_result = None
            st.rerun()

    if sample_col3.button("📏 Crack", use_container_width=True):
        if os.path.exists(crack_path):
            st.session_state.current_image = Image.open(crack_path)
            st.session_state.current_image_name = "longitudinal_crack.jpg (Sample 3)"
            st.session_state.assessment_result = None
            st.rerun()

    st.markdown("---")
    with st.expander("📚 RDD2022 Class Reference"):
        st.markdown("""
        **Target Damage Classes:**
        - **D00**: Longitudinal Crack *(running parallel to road lane)*
        - **D10**: Transverse Crack *(running perpendicular to lane)*
        - **D20**: Alligator Crack *(interconnected fatigue network)*
        - **D40**: Pothole *(bowl-shaped structural depression)*
        - **Other/Uncertain**: Unclassified or ambiguous pavement distress
        """)


# ---------------------------------------------------------
# Main Application Flow: Upload & Analysis
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Road or Pothole Image (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image of a road surface or pothole to run automated damage detection and severity classification."
)

if uploaded_file is not None:
    try:
        st.session_state.current_image = Image.open(uploaded_file)
        st.session_state.current_image_name = uploaded_file.name
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            st.session_state.assessment_result = None
            st.session_state.last_uploaded_name = uploaded_file.name
    except Exception as e:
        st.error(f"Error loading image: {e}")

# Check if an image is loaded
if st.session_state.current_image is not None:
    img = st.session_state.current_image
    img_name = st.session_state.current_image_name or "Uploaded Road Image"

    # Action Toolbar
    act_col1, act_col2, act_col3 = st.columns([2, 1, 1])
    with act_col1:
        run_analysis = st.button("🚀 Detect Road Damage & Assess Severity", type="primary", use_container_width=True)
    with act_col2:
        if st.button("🔄 Reset / Clear", use_container_width=True):
            st.session_state.current_image = None
            st.session_state.assessment_result = None
            st.session_state.current_image_name = None
            st.rerun()
    with act_col3:
        use_demo = st.checkbox(
            "Demo Simulation Mode", 
            value=(not bool(api_key_input)), 
            help="Run pre-calibrated RDD2022 assessment without contacting the Gemini API"
        )

    # Perform Detection & Assessment
    if run_analysis:
        context_str = f"Road Type: {road_context_choice}"
        
        if api_key_input and not use_demo:
            with st.spinner(f"Analyzing road distress with {model_choice} via Gemini API..."):
                try:
                    result = analyze_road_damage_gemini(
                        image=img,
                        api_key=api_key_input.strip(),
                        model_name=model_choice,
                        road_context=context_str
                    )
                    st.session_state.assessment_result = result
                    st.session_state.assessment_source = f"Live Gemini API ({model_choice})"
                except Exception as err:
                    st.error(f"⚠️ Gemini API Call Failed: {str(err)}")
                    st.info("💡 Falling back to Demo Simulation mode so you can view the complete assessment dashboard.")
                    result = get_demo_rdd_assessment(img_name, context_str)
                    st.session_state.assessment_result = result
                    st.session_state.assessment_source = "Demo Simulation Fallback"
        else:
            with st.spinner("Executing RDD2022 road defect detection in simulation mode..."):
                time.sleep(0.5)
                result = get_demo_rdd_assessment(img_name, context_str)
                st.session_state.assessment_result = result
                st.session_state.assessment_source = "Demo Simulation Mode (Add Gemini API key in sidebar for live analysis)"

    st.markdown("---")

    # ---------------------------------------------------------
    # Two-Column Results UI (Requirement 4)
    # ---------------------------------------------------------
    col_left, col_right = st.columns([1, 1], gap="large")

    res = st.session_state.assessment_result
    defects = res.get("defects", []) if res else []

    # =========================================================
    # LEFT COLUMN: Annotated Image with Bounding Boxes
    # =========================================================
    with col_left:
        st.subheader("📷 Annotated Visual Inspection")

        if res is not None:
            if len(defects) > 0:
                # Render bounding boxes onto image
                annotated_img = draw_bounding_boxes(img, defects)
                
                view_mode = st.radio(
                    "Display Mode:",
                    ["Annotated (With Bounding Boxes)", "Original Image"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if view_mode == "Annotated (With Bounding Boxes)":
                    st.image(
                        annotated_img,
                        caption=f"Annotated Inspection: {len(defects)} defect(s) detected in {img_name}",
                        width="stretch"
                    )
                else:
                    st.image(
                        img,
                        caption=f"Original Road Surface: {img_name}",
                        width="stretch"
                    )
            else:
                # Clear "No road damage detected" view
                st.image(
                    img,
                    caption=f"Road Surface View: {img_name}",
                    width="stretch"
                )
                st.markdown("""
                <div class="no-damage-card">
                    <div style="font-size: 2.2rem; margin-bottom: 0.4rem;">✅</div>
                    <div style="font-weight: 700; font-size: 1.2rem; color: #34d399;">No Road Damage Detected</div>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.4rem;">
                        The inspected pavement area shows sound structural surface integrity with no identifiable D00, D10, D20, or D40 distresses.
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Awaiting analysis state
            st.image(
                img,
                caption=f"Awaiting Inspection: {img_name}",
                width="stretch"
            )
            st.caption(f"Dimensions: {img.size[0]} x {img.size[1]} px | Mode: {img.mode}")

        with st.expander("🔍 Image Diagnostics & Metadata"):
            st.json({
                "Source": img_name,
                "Width (px)": img.size[0],
                "Height (px)": img.size[1],
                "Aspect Ratio": f"{round(img.size[0] / img.size[1], 2)}:1",
                "Color Mode": img.mode,
                "Context": road_context_choice,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
            })

    # =========================================================
    # RIGHT COLUMN: Defect Summary & RDD2022 Details
    # =========================================================
    with col_right:
        st.subheader("🧠 RDD2022 Damage Assessment Summary")

        if res is None:
            st.info("👆 Click **'Detect Road Damage & Assess Severity'** above to run the automated inspection.")
            
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.5); border: 2px dashed #334155; border-radius: 12px; padding: 2.5rem; text-align: center; color: #94a3b8;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
                <div style="font-weight: 600; font-size: 1.1rem; color: #cbd5e1;">Awaiting Inspection Trigger</div>
                <p style="margin-top: 0.5rem; font-size: 0.88rem;">
                    The model will locate all pavement defects and extract:
                    <br>• <strong>Damage Type & RDD2022 Class</strong> (D00, D10, D20, D40)
                    <br>• <strong>Normalized Bounding Box Coordinates</strong>
                    <br>• <strong>Severity Level</strong> (LOW / MEDIUM / HIGH)
                    <br>• <strong>Damaged Surface Area Coverage %</strong>
                    <br>• <strong>Actionable Maintenance Interventions</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            source_lbl = st.session_state.get("assessment_source", "Gemini AI")
            if "Demo" in source_lbl:
                st.caption(f"⚡ Mode: **{source_lbl}** — Provide Gemini API key in sidebar for live model calls.")
            else:
                st.success(f"✨ Analysis produced by **{source_lbl}**")

            # Calculate Defect Counters
            total_defects = len(defects)
            high_count = sum(1 for d in defects if d.get("severity") == "HIGH")
            med_count = sum(1 for d in defects if d.get("severity") == "MEDIUM")
            low_count = sum(1 for d in defects if d.get("severity") == "LOW")

            # TOP SUMMARY STAT BAR (Requirement 4)
            st.markdown(f"""
            <div class="summary-bar">
                <div class="stat-card total">
                    <div class="stat-num" style="color: #60a5fa;">{total_defects}</div>
                    <div class="stat-label">Total Defects</div>
                </div>
                <div class="stat-card high">
                    <div class="stat-num" style="color: #ef4444;">{high_count}</div>
                    <div class="stat-label">High Severity</div>
                </div>
                <div class="stat-card medium">
                    <div class="stat-num" style="color: #f59e0b;">{med_count}</div>
                    <div class="stat-label">Medium Severity</div>
                </div>
                <div class="stat-card low">
                    <div class="stat-num" style="color: #10b981;">{low_count}</div>
                    <div class="stat-label">Low Severity</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Executive condition overview
            cond_summary = res.get("road_condition_summary")
            if cond_summary:
                st.caption(f"📝 **Overview:** {cond_summary}")

            # Check if any defects were detected
            if total_defects == 0:
                st.markdown("""
                <div class="no-damage-card" style="margin-top: 0.5rem;">
                    <h4 style="color: #34d399; margin: 0;">No Road Damage Detected</h4>
                    <p style="font-size: 0.9rem; color: #94a3b8; margin-top: 0.3rem;">
                        No distress instances were identified exceeding the visual detection threshold.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Render Detection Summary for each defect (Requirement 4)
                st.markdown("#### Detected Defect Details:")
                
                for idx, defect in enumerate(defects, start=1):
                    dmg_name = defect.get("damage_type", "Road Distress")
                    rdd_code = normalize_rdd2022_class(defect.get("rdd2022_class"))
                    rdd_label = RDD2022_CLASSES.get(rdd_code, "Other / Uncertain Distress")
                    severity = normalize_severity(defect.get("severity"))
                    confidence = float(defect.get("confidence", 0.85))
                    coverage = float(defect.get("surface_coverage_percent", 5.0))
                    action = defect.get("maintenance_action", "Standard maintenance procedure")
                    box = defect.get("bounding_box", [0, 0, 1000, 1000])

                    st.markdown(f"""
                    <div class="defect-card">
                        <div class="defect-header">
                            <div>
                                <span style="color: #94a3b8; font-weight: 700; margin-right: 0.3rem;">#{idx}</span>
                                <span class="defect-title">{dmg_name}</span>
                            </div>
                            <div style="display: flex; gap: 0.4rem; align-items: center;">
                                <span class="rdd-badge">{rdd_code}</span>
                                <span class="severity-badge-{severity}">{severity}</span>
                            </div>
                        </div>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <div class="detail-key">RDD2022 Classification</div>
                                <div class="detail-val">{rdd_label}</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-key">Confidence Score</div>
                                <div class="detail-val">{int(confidence * 100)}%</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-key">Surface Coverage %</div>
                                <div class="detail-val" style="color: #38bdf8;">{coverage:.1f}%</div>
                            </div>
                            <div class="detail-item">
                                <div class="detail-key">Normalized Box [ymin, xmin, ymax, xmax]</div>
                                <div class="detail-val" style="font-family: monospace; font-size: 0.8rem;">{box}</div>
                            </div>
                        </div>
                        <div class="action-box">
                            <div class="action-label">🛠️ Recommended Maintenance Action</div>
                            <div class="action-text">{action}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Export Report Section
                with st.expander("📑 Export Inspection Data (JSON / Work Order)"):
                    export_dict = {
                        "inspection_record": {
                            "system": "RoadGuard AI (Gemini 2.5 Flash + RDD2022)",
                            "timestamp": datetime.now().isoformat(),
                            "source_image": img_name,
                            "road_classification": road_context_choice,
                            "total_defects_detected": total_defects,
                            "severity_breakdown": {
                                "HIGH": high_count,
                                "MEDIUM": med_count,
                                "LOW": low_count
                            },
                            "defects": defects
                        }
                    }
                    
                    st.download_button(
                        label="💾 Download Structured JSON",
                        data=json.dumps(export_dict, indent=2),
                        file_name=f"rdd2022_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )

else:
    st.info("💡 To begin, choose one of the **Quick Sample Dataset** buttons in the sidebar or upload a road image above.")

st.markdown("---")
st.caption("RoadGuard AI • Infrastructure Intelligence Prototype • Google GenAI (Gemini 2.5 Flash) • RDD2022-Compatible Taxonomy")
