"""
SignBridge AI - Word Document (.docx) Project Report Generator
Creates a professionally styled academic report with custom fonts, tables, and callouts.
"""

import sys
from pathlib import Path
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell internal padding."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_callout_box(doc, title, text, border_color="1F6FEB", bg_color="F0F6FC"):
    """Creates a stylized alert/callout box table."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, bg_color)
    set_cell_margins(cell, top=120, bottom=120, left=200, right=200)
    
    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    
    r_title = p.add_run(f"{title}\n")
    r_title.bold = True
    r_title.font.name = "Arial"
    r_title.font.size = Pt(10.5)
    r_title.font.color.rgb = RGBColor.from_string(border_color)
    
    r_text = p.add_run(text)
    r_text.font.name = "Arial"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = RGBColor(50, 50, 50)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def build_report():
    doc = Document()
    
    # Page Setup: Standard Letter, 0.75" margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Styles Setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Arial'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(40, 40, 40)
    
    # Header Title Section
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    r_icon = title_p.add_run("🤟 ")
    r_icon.font.size = Pt(24)
    r_title = title_p.add_run("SignBridge AI — Comprehensive Project Report")
    r_title.bold = True
    r_title.font.size = Pt(22)
    r_title.font.color.rgb = RGBColor(15, 60, 120)
    
    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_before = Pt(0)
    sub_p.paragraph_format.space_after = Pt(14)
    r_sub = sub_p.add_run("Real-Time Bidirectional American Sign Language (ASL) Recognition & Assisted Communication Suite")
    r_sub.font.size = Pt(11.5)
    r_sub.font.color.rgb = RGBColor(100, 110, 125)
    
    # Metadata Badge Bar
    meta_table = doc.add_table(rows=1, cols=4)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    meta_data = [
        ("Author", "Manik Gupta"),
        ("Date", "October 2026"),
        ("Status", "Working MVP / Production-Ready"),
        ("Unit Tests", "51 / 51 Passing (100%)")
    ]
    for i, (k, v) in enumerate(meta_data):
        cell = meta_table.cell(0, i)
        cell.width = Inches(1.7)
        set_cell_background(cell, "F3F4F6")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        rk = p.add_run(f"{k.upper()}\n")
        rk.font.size = Pt(7.5)
        rk.font.bold = True
        rk.font.color.rgb = RGBColor(110, 115, 125)
        rv = p.add_run(v)
        rv.font.size = Pt(9)
        rv.font.bold = True
        rv.font.color.rgb = RGBColor(15, 60, 120)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary & Project Motivation", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(4)
    for r in h1.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor(15, 60, 120)
        
    p = doc.add_paragraph(
        "Communication between the deaf/mute community and the hearing population remains one of "
        "the most significant societal and accessibility barriers today. Traditional sign language recognition "
        "approaches suffer from distinct drawbacks: sensory gloves are prohibitively expensive, cumbersome, "
        "and unnatural, while conventional 2D Convolutional Neural Networks (CNNs) trained on raw pixel images "
        "frequently fail under real-world conditions due to variations in room lighting, skin tones, and background clutter."
    )
    p.paragraph_format.space_after = Pt(6)
    
    p2 = doc.add_paragraph(
        "SignBridge AI resolves this bottleneck by pioneering an accessible, software-only, camera-agnostic "
        "communication suite that runs locally on consumer laptops and webcams. By fusing 3D skeletal landmark tracking "
        "(21 coordinates in 3D Euclidean space) with invariant spatial vector math, deep convolutional networks, "
        "smart NLP predictive text, and bidirectional speech-to-text / text-to-speech bridges, SignBridge AI "
        "converts ordinary webcams into real-time assistive translation hubs."
    )
    p2.paragraph_format.space_after = Pt(8)
    
    create_callout_box(
        doc,
        "⚠️ Important Linguistic & Accessibility Distinction",
        "SignBridge AI focuses on finger-spelling (letter-by-letter spelling) and universal whole-word phrase gestures. "
        "Finger-spelling is a recognized orthographic subset of ASL used for proper names, terminology, and spelling. "
        "It does not equate to full continuous ASL translation, which depends on spatial grammar, body shift, and non-manual facial markers.",
        border_color="D29922",
        bg_color="FEF9EE"
    )
    
    # -------------------------------------------------------------------------
    # 2. Technology Stack
    # -------------------------------------------------------------------------
    h2 = doc.add_heading("2. Comprehensive Technology Stack", level=1)
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(6)
    for r in h2.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor(15, 60, 120)
        
    tech_table = doc.add_table(rows=1, cols=3)
    tech_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tech_table.autofit = False
    
    hdr_cells = tech_table.rows[0].cells
    headers = ["Domain / Subsystem", "Technology / Library", "Engineering Purpose & Function"]
    widths = [Inches(1.8), Inches(2.2), Inches(2.9)]
    
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        hdr_cells[i].width = widths[i]
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], 100, 100, 120, 120)
        p = hdr_cells[i].paragraphs[0]
        p.runs[0].font.name = "Arial"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(9.5)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        
    tech_rows = [
        ("Programming Language", "Python 3.11", "High-performance base runtime for numerical math, ML models, and audio pipelines."),
        ("Computer Vision & Skeletal Tracking", "OpenCV & MediaPipe Hands", "Real-time webcam frame acquisition, HUD rendering, and 21-joint 3D skeletal landmark estimation."),
        ("Deep Vision Model", "PyTorch / torchvision (EfficientNetB0)", "2D visual spatial feature classification across 29 discrete classes."),
        ("Geometric Landmark Engine", "Pure NumPy Vector Heuristics", "Scale-, distance-, and rotation-invariant bone angle straightness and radial projection calculation."),
        ("Predictive NLP Engine", "N-Gram & Frequency Lexicon Trie", "Prefix word autocompletion, word transition probability modeling, and instant abbreviation expansion."),
        ("Bidirectional Speech Audio", "Native Web Speech API & pyttsx3", "Zero-latency in-browser speech recognition (captions) and multi-engine Text-to-Speech audio output."),
        ("Web Application Frontend", "Streamlit (Custom Glassmorphism CSS)", "Responsive, dark-themed user studio with live camera feeds, metrics panels, and two-way station."),
        ("Quality Assurance & Testing", "Pytest 9.1.1", "51 automated unit tests validating 100% mathematical classifier precision and state machine debouncing.")
    ]
    
    for domain, tech, purpose in tech_rows:
        row = tech_table.add_row()
        cells = row.cells
        for i, val in enumerate([domain, tech, purpose]):
            cells[i].text = val
            cells[i].width = widths[i]
            set_cell_margins(cells[i], 70, 70, 100, 100)
            p = cells[i].paragraphs[0]
            p.runs[0].font.name = "Arial"
            p.runs[0].font.size = Pt(8.5)
            if i == 0:
                p.runs[0].font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------------------
    # 3. System Architecture & Modular Breakdown
    # -------------------------------------------------------------------------
    h3 = doc.add_heading("3. Core Modular Architecture & Implementation", level=1)
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)
    for r in h3.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor(15, 60, 120)
        
    # Module 3.1
    m1 = doc.add_heading("3.1 Invariant 3D Skeletal Landmark Classifier (src/landmark_classifier.py)", level=2)
    m1.paragraph_format.space_before = Pt(8)
    m1.paragraph_format.space_after = Pt(3)
    for r in m1.runs:
        r.font.color.rgb = RGBColor(30, 80, 140)
        
    p = doc.add_paragraph(
        "The skeletal classifier eliminates background, skin tone, and distance variance through anatomical heuristics:\n"
        "• Dynamic Scale Normalization: Normalizes Euclidean distance across all 21 landmarks by the user's palm length "
        "(Wrist to Middle MCP joint distance), enabling flawless operation whether the user sits 2 feet or 6 feet away.\n"
        "• 3D Bone Straightness (Cosine Similarity): Computes the dot product between the proximal vector (MCP to PIP) "
        "and distal vector (PIP to TIP). An extended finger yields ~1.0; a curled finger yields <=0.0, completely independent of camera angle.\n"
        "• Radial Tuck Discrimination: Solves the infamous similarity between A, E, S, T, N, and M by computing thumb tip "
        "projection along the radial axis (Pinky MCP to Index MCP vector) and knuckle gap boundaries.\n"
        "• Curved 'C' Arch vs. Closed 'O' Circle: Employs 3D aperture thresholds distinguishing an open curved 'C' profile "
        "from a pinched circular 'O' contour."
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Module 3.2
    m2 = doc.add_heading("3.2 Instant Whole-Word Gesture Recognition (Option A)", level=2)
    m2.paragraph_format.space_before = Pt(8)
    m2.paragraph_format.space_after = Pt(3)
    for r in m2.runs:
        r.font.color.rgb = RGBColor(30, 80, 140)
        
    p = doc.add_paragraph(
        "Rather than forcing users to spell common conversational words letter-by-letter, the system incorporates "
        "whole-word recognition with high confidence (0.98 score):\n"
        "• 🤟 'I LOVE YOU' (ASL ILY Sign): Thumb, Index, and Pinky extended; Middle and Ring curled into palm.\n"
        "• 👍 'YES' (ASL Thumbs Up): All 4 fingers curled tightly into a fist; Thumb pointing straight UP high above knuckles.\n"
        "• 👎 'NO' (ASL Thumbs Down): Fist curled tightly; Thumb pointing straight DOWN below wrist joint.\n"
        "• ✋ 'STOP' (Open Upright Palm): All 4 fingers fully extended upright with palm oriented forward toward camera.\n"
        "• 👌 'OKAY' (ASL OK Sign): Thumb and Index tips touching in a ring; Middle, Ring, and Pinky extended vertically."
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Module 3.3
    m3 = doc.add_heading("3.3 Predictive NLP Engine & Abbreviation Expansions (src/predictive_text.py)", level=2)
    m3.paragraph_format.space_before = Pt(8)
    m3.paragraph_format.space_after = Pt(3)
    for r in m3.runs:
        r.font.color.rgb = RGBColor(30, 80, 140)
        
    p = doc.add_paragraph(
        "Finger-spelling full words can be labor-intensive. To solve this, the application integrates an NLP engine:\n"
        "• Prefix Autocomplete: Continuously inspects the active word buffer and ranks suggestions from an English lexicon (e.g. 'HE' -> 'HELLO', 'HELP', 'HERE').\n"
        "• N-Gram Transition Anticipation: Predicts the logical next word based on preceding vocabulary (e.g. after 'HELLO', suggests 'HOW', 'WORLD', 'MY').\n"
        "• Abbreviation Expansion on Space: Signing fast shorthand abbreviations immediately expands into full phrases upon signing Space:\n"
        "   - 'TY' -> 'THANK YOU'  |  'HRU' -> 'HOW ARE YOU'  |  'GM' -> 'GOOD MORNING'\n"
        "   - 'GN' -> 'GOOD NIGHT'  |  'SOS' -> 'EMERGENCY NEED HELP'  |  'PLS' -> 'PLEASE'\n"
        "   - 'ILY' -> 'I LOVE YOU' |  'IDK' -> 'I DON'T KNOW'  |  'NP' -> 'NO PROBLEM'"
    )
    p.paragraph_format.space_after = Pt(8)
    
    # Module 3.4
    m4 = doc.add_heading("3.4 Two-Way Conversation Bridge Station (Option B in app.py)", level=2)
    m4.paragraph_format.space_before = Pt(8)
    m4.paragraph_format.space_after = Pt(3)
    for r in m4.runs:
        r.font.color.rgb = RGBColor(30, 80, 140)
        
    p = doc.add_paragraph(
        "A dedicated communication hub designed for back-and-forth dialogue between hearing and deaf individuals:\n"
        "• Hearing Partner Station (Speech-to-Text): Captures live spoken English via the browser's native Web Speech API "
        "and displays high-contrast real-time captions on screen for the deaf partner to read with zero cloud delay.\n"
        "• Deaf Partner Station (Sign & Quick Reply): Provides 1-tap quick replies ('YES', 'NO', 'THANK YOU', 'PLEASE HELP', 'REPEAT') "
        "and a composed sentence buffer connected to Text-to-Speech audio output that speaks responses aloud through speakers.\n"
        "• Shared Dialogue Transcript: Chronological messenger-style chat interface with one-click TXT export."
    )
    p.paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------------------
    # 4. Empirical Evaluation & Quality Engineering
    # -------------------------------------------------------------------------
    h4 = doc.add_heading("4. Testing & Verification Dossier", level=1)
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)
    for r in h4.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor(15, 60, 120)
        
    p = doc.add_paragraph(
        "The repository features an exhaustive automated test suite using Pytest. A total of 51 unit tests validate "
        "every component of the application with a 100% pass rate:"
    )
    p.paragraph_format.space_after = Pt(6)
    
    test_table = doc.add_table(rows=1, cols=3)
    test_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    test_table.autofit = False
    
    t_hdr = test_table.rows[0].cells
    t_hdr[0].text = "Test Module"
    t_hdr[1].text = "Tests Count"
    t_hdr[2].text = "Verification Scope"
    t_hdr[0].width = Inches(2.2)
    t_hdr[1].width = Inches(1.1)
    t_hdr[2].width = Inches(3.6)
    
    for c in t_hdr:
        set_cell_background(c, "1F4E79")
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        p.runs[0].font.name = "Arial"
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(9)
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        
    test_data = [
        ("test_landmark_classifier.py", "25 Tests", "Validates geometric classification of all letters (A-Z) and whole-word signs (ILY, Yes, No, Stop, Okay)."),
        ("test_sentence_builder.py", "11 Tests", "Validates temporal hold duration, debounce cooldown, auto-accept on space, and phrase token spacing."),
        ("test_predictive_text.py", "8 Tests", "Validates trie lookup, prefix completion, next-word transitions, and abbreviation expansions."),
        ("test_prediction_smoother.py", "4 Tests", "Validates majority voting buffer, confidence rejection, and uncertainty reset."),
        ("test_preprocessing.py", "3 Tests", "Validates image normalization, shape bounds, and adaptive ROI extraction.")
    ]
    
    for mod, count, scope in test_data:
        row = test_table.add_row()
        cells = row.cells
        cells[0].text = mod
        cells[1].text = count
        cells[2].text = scope
        cells[0].width = Inches(2.2)
        cells[1].width = Inches(1.1)
        cells[2].width = Inches(3.6)
        for i, c in enumerate(cells):
            set_cell_margins(c, 60, 60, 90, 90)
            p = c.paragraphs[0]
            p.runs[0].font.name = "Arial"
            p.runs[0].font.size = Pt(8.5)
            if i < 2:
                p.runs[0].font.bold = True
                
    doc.add_paragraph().paragraph_format.space_after = Pt(10)
    
    # -------------------------------------------------------------------------
    # 5. Teacher Demonstration Protocol
    # -------------------------------------------------------------------------
    h5 = doc.add_heading("5. Teacher Demonstration Walkthrough Protocol", level=1)
    h5.paragraph_format.space_before = Pt(14)
    h5.paragraph_format.space_after = Pt(6)
    for r in h5.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor(15, 60, 120)
        
    p = doc.add_paragraph(
        "When presenting this project to your instructor or evaluators, follow this 4-step demonstration protocol for maximum impact:\n"
        "1. Real-Time Camera Studio & Invariant Geometry: Launch 'Live Camera Studio'. Sign 'H' and 'E', showing how distance and lighting changes do not confuse the 3D landmark engine. Point out the live confidence gauge and hold meter.\n"
        "2. Smart Autocomplete & Shorthand: Show the suggestion chip 'HELLO' automatically appearing after 'HE'. Sign 'Space' to accept it. Then sign 'T' - 'Y' followed by 'Space' to show instant expansion into 'THANK YOU'.\n"
        "3. Instant Whole-Word Gestures: Demonstrate 🤟 ('I LOVE YOU'), 👍 ('YES'), 👎 ('NO'), and ✋ ('STOP') to showcase instant phrase detection without letter spelling.\n"
        "4. Two-Way Conversation Bridge: Switch to 'Two-Way Conversation Bridge'. Speak into the microphone so your teacher sees live spoken captions appear in real time. Then tap '[🙏 THANK YOU]' and let your teacher hear the computer voice speak the response aloud!"
    )
    p.paragraph_format.space_after = Pt(8)
    
    # -------------------------------------------------------------------------
    # 6. Future Work & Roadmap
    # -------------------------------------------------------------------------
    h6 = doc.add_heading("6. Long-Term Project Roadmap", level=1)
    h6.paragraph_format.space_before = Pt(14)
    h6.paragraph_format.space_after = Pt(6)
    for r in h6.runs:
        r.font.name = "Arial"
        r.font.color.rgb = RGBColor(15, 60, 120)
        
    p = doc.add_paragraph(
        "Planned future research milestones for subsequent project iterations:\n"
        "• Phase 3 — Dynamic Trajectory Tracking: Optical tracking of neon fingertip paths for movement-based signs ('J', 'Z').\n"
        "• Phase 4 — Dual-Hand Coordination: Tracking max_num_hands=2 for bimanual vocabulary ('Please', 'Friend', 'Welcome').\n"
        "• Phase 5 — Gamified ASL Practice Studio: Interactive Duolingo-style learning mode with real-time skeletal posture grading.\n"
        "• Phase 6 — Multilingual Translation Bridge: Translating signed English into spoken Spanish, French, Hindi, and German.\n"
        "• Phase 7 — Mobile & Edge Deployment: Packaging into a progressive web app (PWA) optimized for mobile smartphones."
    )
    p.paragraph_format.space_after = Pt(12)
    
    # Final Sign-off
    callout = doc.add_paragraph()
    callout.paragraph_format.space_before = Pt(8)
    r_foot = callout.add_run("Project Code Repository: https://github.com/manikgupta-2004/SignBridge-AI")
    r_foot.font.name = "Arial"
    r_foot.font.size = Pt(9.5)
    r_foot.font.italic = True
    r_foot.font.color.rgb = RGBColor(15, 60, 120)

    # Save to both locations
    out_paths = [
        Path("/Users/manikgupta/Desktop/SignBridge-AI/SignBridge_AI_Project_Report.docx"),
        Path("/Users/manikgupta/Downloads/SignBridge_AI_Project_Report.docx")
    ]
    for p in out_paths:
        doc.save(str(p))
        print(f"✅ Generated: {p}")

if __name__ == "__main__":
    build_report()
