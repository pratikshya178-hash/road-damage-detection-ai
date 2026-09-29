# 🛣️ RoadGuard AI — RDD2022 Road Damage Detection & Severity Assessment Prototype

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64+-FF4B4B.svg)](https://streamlit.io/)
[![Google GenAI SDK](https://img.shields.io/badge/Google%20GenAI-Gemini%202.5%20Flash-4285F4.svg)](https://ai.google.dev/)
[![Taxonomy](https://img.shields.io/badge/Taxonomy-RDD2022%20Compatible-green.svg)](https://crddc2022.sekilab.global/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent civil infrastructure computer vision prototype built with **Streamlit**, the official **Google GenAI SDK**, and **Pillow (PIL)**. **RoadGuard AI** uses **Gemini 2.5 Flash** for multimodal visual road damage detection, localizing pavement distresses with bounding boxes, mapping them into the standardized **RDD2022** classification taxonomy, and prescribing civil engineering maintenance actions.

> [!IMPORTANT]
> **Architecture & Methodology Notice**: This prototype is powered directly by Google's multimodal **Gemini 2.5 Flash** foundation vision model using structured JSON prompt engineering and coordinate normalization. It **does NOT** use a locally trained YOLO model, nor was the RDD2022 dataset used for model training or weight fine-tuning. The RDD2022 taxonomy is applied as a standardized classification scheme for distress labeling.

---

## 🎯 RDD2022-Compatible Taxonomy

The system classifies road distress into the standardized target classes established in the international Road Damage Detection benchmark (RDD2022):

| RDD2022 Class Code | Distress Name | Visual Description | Typical Engineering Treatment |
| :---: | :--- | :--- | :--- |
| **D00** | **Longitudinal Crack** | Cracks running parallel to the direction of travel, typically along wheel paths or paving joints. | Hot-pour rubberized crack sealing with channel routing |
| **D10** | **Transverse Crack** | Cracks running perpendicular to the pavement centerline, caused by thermal contraction stresses. | Low-viscosity asphalt emulsion injection & sand blot |
| **D20** | **Alligator Crack** | Interconnected polygonal fatigue crack networks resembling crocodile skin; indicates base fatigue. | Surface milling & 50mm structural asphalt overlay |
| **D40** | **Pothole** | Bowl-shaped cavity penetrating through wearing course with exposed aggregate and sharp edges. | Full-depth semi-permanent hot-mix patching & plate compaction |
| **Other/Uncertain** | **Other Distress** | Ravelling, rutting, edge break, or ambiguous distress not confidently fitting D00/D10/D20/D40. | Targeted engineering inspection & site-specific repair |

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Client [Streamlit Frontend]
        A[User Upload / Sample Image] --> B[Sidebar Config\nAPI Key & Road Context]
        C[Image Preprocessing\nPillow RGB Normalization]
    end

    subgraph AI [Gemini Multimodal Inference]
        B --> D[google-genai Client\nGemini 2.5 Flash]
        C --> D
        D --> E[Multimodal Defect Detection\nNormalized [ymin, xmin, ymax, xmax] 0-1000]
        E --> F[Structured JSON Output\nRDD2022 Taxonomy & ASTM D6433]
    end

    subgraph Visualization [Rendering Engine]
        F --> G[Bounding Box Generator\nPIL.ImageDraw + Coordinate Clipping]
        F --> H[Metric Engine\nTotal Count + Severity Breakdown]
        G --> I[Left Column:\nAnnotated Image Display]
        H --> J[Right Column:\nDefect Cards & Maintenance Actions]
    end
```

---

## 📐 Bounding Box Normalization & Coordinate Handling

Gemini 2.5 Flash outputs normalized bounding box coordinates in the `[ymin, xmin, ymax, xmax]` format on a `0–1000` scale. The application converts these into exact pixel coordinates:

$$\text{pixel\_ymin} = \left\lfloor \frac{\text{ymin}}{1000} \times \text{Height} \right\rfloor, \quad \text{pixel\_xmin} = \left\lfloor \frac{\text{xmin}}{1000} \times \text{Width} \right\rfloor$$

$$\text{pixel\_ymax} = \left\lfloor \frac{\text{ymax}}{1000} \times \text{Height} \right\rfloor, \quad \text{pixel\_xmax} = \left\lfloor \frac{\text{xmax}}{1000} \times \text{Width} \right\rfloor$$

- Coordinates are strictly clipped to image dimensions ($0 \le x < W$, $0 \le y < H$).
- High-contrast color-coded outlines:
  - 🔴 **HIGH Severity**: Crimson Red (`#EF4444`)
  - 🟡 **MEDIUM Severity**: Amber Orange (`#F59E0B`)
  - 🟢 **LOW Severity**: Emerald Green (`#10B981`)
- Each bounding box features a solid pill label: `#{Index} [{RDD_Class}] {Severity} {Confidence}%`.
- If no defects are identified, a clean `"No road damage detected"` banner is displayed.

---

## 🖥️ User Interface Layout

The results interface uses a balanced **Two-Column Layout**:

- **LEFT COLUMN (Annotated Visual Inspection)**:
  - Interactive display of the annotated image with visible bounding boxes and defect labels.
  - Radio toggle to switch between **Annotated View** and **Original Road Surface**.
  - Technical image diagnostics (dimensions, mode, aspect ratio, inspection timestamp).
- **RIGHT COLUMN (RDD2022 Damage Assessment Summary)**:
  - **Summary Metric Bar**:
    - Total Defects Detected
    - High Severity Defect Count (🔴)
    - Medium Severity Defect Count (🟡)
    - Low Severity Defect Count (🟢)
  - **Defect Cards**:
    - Defect Name & Severity Badge
    - RDD2022 Classification (`D00`, `D10`, `D20`, `D40`, or `Other/Uncertain`)
    - Confidence percentage score
    - Damaged Surface Area Coverage %
    - Prescribed Engineering Maintenance Action
  - **Export Options**: Download complete inspection report as structured JSON.

---

## 📂 Project Structure

```
road_damage_ai/
├── app.py                  # Core Streamlit application (RDD2022 vision pipeline)
├── test_app.py             # Unit test verification suite
├── requirements.txt        # Project dependencies (streamlit, google-genai, pillow)
├── README.md               # Architecture and documentation
└── sample_images/          # Bundled sample road distress images
    ├── pothole_severe.jpg        # Demonstrates D40 (Pothole - HIGH severity)
    ├── alligator_cracking.jpg    # Demonstrates D20 (Alligator Crack - HIGH severity)
    └── longitudinal_crack.jpg    # Demonstrates D00 & D10 (Longitudinal & Transverse Cracks)
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- **Python**: Version 3.10 to 3.13.
- **Gemini API Key**: Free key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*Contents of `requirements.txt`:*
```
streamlit
google-genai
pillow
```

---

## 💻 Running the Prototype

Start the Streamlit application:

```bash
streamlit run app.py
```

Streamlit will launch the web application at:
- **Local URL**: `http://localhost:8501`

### 🔑 Supplying the Gemini API Key
1. **Sidebar Input (Recommended)**: Enter your key in the **Sidebar -> Enter Gemini API Key** field. The key is securely masked.
2. **Environment Variable**: Alternatively, export `GEMINI_API_KEY`:
   ```powershell
   $env:GEMINI_API_KEY="AIzaSyYourSecretKey"
   streamlit run app.py
   ```
3. **Demo Simulation Mode**: If no API key is provided, the prototype runs in **Demo Mode**, utilizing calibrated RDD2022 detections for all sample images so you can immediately explore bounding boxes, metrics, and exports without an API key.

---

## 🧪 Testing

Run the included automated unit test suite to verify RDD2022 class normalization, severity parsing, robust JSON handling, and Pillow coordinate clipping:

```bash
python test_app.py
```

Expected output:
```
Test 1 Passed: RDD2022 normalization
Test 2 Passed: Severity normalization
Test 3 Passed: Robust JSON parsing
Test 4 Passed: Bounding box drawing on sample images
Test 5 Passed: Empty defect handling
>>> ALL UNIT TESTS PASSED SUCCESSFULLY! <<<
```

---

## 🛡️ License

This prototype is distributed under the MIT License.
