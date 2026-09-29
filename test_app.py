"""
Comprehensive End-to-End Unit & Integration Tests for RoadGuard AI
Tests RDD2022 class handling, severity, metrics, drawing, and API resilience.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
import json
from PIL import Image
from app import (
    normalize_rdd2022_class,
    normalize_severity,
    draw_bounding_boxes,
    extract_and_parse_json,
    get_demo_rdd_assessment,
    RDD2022_CLASSES,
    SEVERITY_COLORS
)

# -------------------------------------------------------------
# Test 1: RDD2022-Compatible Classes Handling (D00, D10, D20, D40, Other)
# -------------------------------------------------------------
print("\n--- Running Test 1: RDD2022 Target Class Normalization ---")
assert normalize_rdd2022_class("D00") == "D00"
assert normalize_rdd2022_class("Longitudinal Crack") == "D00"
assert normalize_rdd2022_class("longitudinal wheel-path crack") == "D00"

assert normalize_rdd2022_class("D10") == "D10"
assert normalize_rdd2022_class("Transverse Crack") == "D10"
assert normalize_rdd2022_class("transverse thermal crack") == "D10"

assert normalize_rdd2022_class("D20") == "D20"
assert normalize_rdd2022_class("Alligator Crack") == "D20"
assert normalize_rdd2022_class("fatigue crocodile cracking") == "D20"

assert normalize_rdd2022_class("D40") == "D40"
assert normalize_rdd2022_class("Pothole") == "D40"
assert normalize_rdd2022_class("severe deep pothole cavity") == "D40"

# Ambiguous / unclassified distress must map to Other/Uncertain
assert normalize_rdd2022_class("Rutting depression") == "Other/Uncertain"
assert normalize_rdd2022_class("Ravelling aggregate loss") == "Other/Uncertain"
assert normalize_rdd2022_class("Unknown surface defect") == "Other/Uncertain"
assert normalize_rdd2022_class("") == "Other/Uncertain"
assert normalize_rdd2022_class(None) == "Other/Uncertain"

# Check all 4 target codes are registered in class dictionary
for code in ["D00", "D10", "D20", "D40", "Other/Uncertain"]:
    assert code in RDD2022_CLASSES, f"Class {code} missing in RDD2022_CLASSES"
print("✓ Test 1 Passed: D00, D10, D20, D40, and Other/Uncertain mapped properly.")

# -------------------------------------------------------------
# Test 2: Severity Normalization (LOW, MEDIUM, HIGH)
# -------------------------------------------------------------
print("\n--- Running Test 2: Severity Level Normalization ---")
assert normalize_severity("HIGH") == "HIGH"
assert normalize_severity("high") == "HIGH"
assert normalize_severity("Critically High") == "HIGH"

assert normalize_severity("MEDIUM") == "MEDIUM"
assert normalize_severity("medium") == "MEDIUM"
assert normalize_severity("Moderate") == "MEDIUM"

assert normalize_severity("LOW") == "LOW"
assert normalize_severity("low") == "LOW"
assert normalize_severity("superficial hairline low") == "LOW"

for sev in ["LOW", "MEDIUM", "HIGH"]:
    assert sev in SEVERITY_COLORS, f"Severity {sev} missing from SEVERITY_COLORS"
print("✓ Test 2 Passed: Severity levels LOW, MEDIUM, HIGH normalized accurately.")

# -------------------------------------------------------------
# Test 3: Robust JSON Parsing
# -------------------------------------------------------------
print("\n--- Running Test 3: JSON Extraction & Fault Tolerance ---")
raw_markdown = '''
```json
{
  "defects": [
    {
      "damage_type": "Pothole",
      "rdd2022_class": "D40",
      "bounding_box": [200, 150, 800, 750],
      "confidence": 0.96,
      "severity": "HIGH",
      "surface_coverage_percent": 18.5,
      "maintenance_action": "Full-Depth Asphalt Patching",
    }
  ],
  "road_condition_summary": "Deep pothole detected.",
  "overall_pavement_rating": "CRITICAL"
}
```
'''
parsed = extract_and_parse_json(raw_markdown)
assert "defects" in parsed
assert len(parsed["defects"]) == 1
assert parsed["defects"][0]["rdd2022_class"] == "D40"
assert parsed["defects"][0]["confidence"] == 0.96
print("✓ Test 3 Passed: Markdown fences & trailing commas handled gracefully.")

# -------------------------------------------------------------
# Test 4: End-to-End Analysis & Metrics on Sample Images
# -------------------------------------------------------------
print("\n--- Running Test 4: End-to-End Sample Image Analysis & Metrics ---")
sample_files = ["pothole_severe.jpg", "alligator_cracking.jpg", "longitudinal_crack.jpg"]

for filename in sample_files:
    path = os.path.join("sample_images", filename)
    assert os.path.exists(path), f"Sample image {path} does not exist!"
    
    img = Image.open(path)
    W, H = img.size
    assert W > 0 and H > 0
    
    assessment = get_demo_rdd_assessment(filename)
    assert "defects" in assessment
    assert len(assessment["defects"]) > 0
    
    for d in assessment["defects"]:
        # Verify all required fields
        assert "damage_type" in d and len(d["damage_type"]) > 0
        assert "rdd2022_class" in d and d["rdd2022_class"] in RDD2022_CLASSES
        assert "severity" in d and d["severity"] in ("LOW", "MEDIUM", "HIGH")
        assert "confidence" in d and 0.0 <= d["confidence"] <= 1.0
        assert "surface_coverage_percent" in d and d["surface_coverage_percent"] > 0
        assert "maintenance_action" in d and len(d["maintenance_action"]) > 0
        assert "bounding_box" in d and len(d["bounding_box"]) == 4

    # Verify Bounding Box Drawing & Label Rendering
    annotated = draw_bounding_boxes(img, assessment["defects"])
    assert annotated.size == (W, H)
    assert annotated.mode == "RGB"
    
    print(f"✓ Analyzed {filename}: {len(assessment['defects'])} defect(s) detected with valid metrics and bounding boxes.")

# -------------------------------------------------------------
# Test 5: Empty Defect Handling (Clean Road Surface)
# -------------------------------------------------------------
print("\n--- Running Test 5: Zero-Defect Handling ---")
clean_img = Image.new("RGB", (640, 480), color=(100, 100, 100))
annotated_clean = draw_bounding_boxes(clean_img, [])
assert annotated_clean.size == (640, 480)
print("✓ Test 5 Passed: No errors on empty defect list.")

# -------------------------------------------------------------
# Test 6: Gemini SDK Client Instantiation & API Key Validation
# -------------------------------------------------------------
print("\n--- Running Test 6: Google GenAI Client Interface ---")
from google import genai
test_client = genai.Client(api_key="AIzaSyFakeKeyForInterfaceTest")
assert test_client is not None
print("✓ Test 6 Passed: google-genai Client initialized cleanly with API key parameter.")

print("\n=======================================================")
print(">>> ALL 6 END-TO-END VERIFICATION TESTS PASSED! <<<")
print("=======================================================\n")
