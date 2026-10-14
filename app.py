"""
SignBridge AI - Modern Streamlit Web Application
A state-of-the-art interactive real-time American Sign Language (ASL) Alphabet recognition suite.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG
from src.inference import SignPredictor
from src.landmark_classifier import LandmarkSignClassifier
from src.preprocessing import extract_hand_roi, get_center_roi
from src.prediction_smoother import PredictionSmoother
from src.sentence_builder import SentenceBuilder
from src.tts_engine import TTSEngine


# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Design System
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SignBridge AI — ASL Alphabet Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a sleek, modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 10% 20%, #0d1117 0%, #161b22 90%);
        color: #e6edf3;
    }

    .hero-container {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.85) 0%, rgba(13, 17, 23, 0.95) 100%);
        border: 1px solid rgba(56, 139, 253, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff 0%, #7ee787 50%, #d2a8ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #8b949e;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        backdrop-filter: blur(8px);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-stable {
        background: rgba(46, 160, 67, 0.2);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    .badge-uncertain {
        background: rgba(218, 54, 51, 0.2);
        color: #f85149;
        border: 1px solid rgba(218, 54, 51, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    .sentence-display {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 24px;
        font-size: 1.6rem;
        font-weight: 600;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        min-height: 80px;
        display: flex;
        align-items: center;
        letter-spacing: 1.5px;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.5);
    }

    .suggestions-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(56, 139, 253, 0.3);
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 10px;
        margin-bottom: 12px;
    }

    .suggestion-chip {
        background: linear-gradient(135deg, rgba(56, 139, 253, 0.22) 0%, rgba(126, 231, 135, 0.18) 100%);
        color: #79c0ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
        padding: 4px 12px;
        border-radius: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .suggestion-chip-top {
        background: linear-gradient(135deg, #1f6feb 0%, #238636 100%);
        color: #ffffff;
        border: 1px solid #388bfd;
        padding: 4px 12px;
        border-radius: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        box-shadow: 0 0 8px rgba(56, 139, 253, 0.5);
    }

    .sugg-label {
        color: #8b949e;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-right: 4px;
    }

    .disclaimer-card {
        background: rgba(210, 153, 34, 0.1);
        border-left: 4px solid #d29922;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 20px;
        font-size: 0.88rem;
        color: #e3b341;
    }

    /* Two-Way Conversation Bridge Styles */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 22px;
        max-height: 440px;
        overflow-y: auto;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.4);
    }

    .chat-bubble-hearing {
        align-self: flex-start;
        background: linear-gradient(135deg, rgba(31, 111, 235, 0.22) 0%, rgba(56, 139, 253, 0.12) 100%);
        border: 1px solid rgba(56, 139, 253, 0.45);
        border-radius: 14px 14px 14px 2px;
        padding: 12px 18px;
        max-width: 82%;
        color: #e6edf3;
    }

    .chat-bubble-deaf {
        align-self: flex-end;
        background: linear-gradient(135deg, rgba(46, 160, 67, 0.22) 0%, rgba(63, 185, 80, 0.12) 100%);
        border: 1px solid rgba(46, 160, 67, 0.45);
        border-radius: 14px 14px 2px 14px;
        padding: 12px 18px;
        max-width: 82%;
        color: #e6edf3;
        text-align: right;
    }

    .chat-sender {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .chat-sender-hearing { color: #58a6ff; }
    .chat-sender-deaf { color: #7ee787; }

    .chat-text {
        font-size: 1.08rem;
        font-weight: 500;
        line-height: 1.45;
    }

    .station-card {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    .caption-highlight {
        background: #0d1117;
        border-left: 5px solid #58a6ff;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #ffffff;
        margin: 12px 0;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Global State & Lazy Cache Initializers
# -----------------------------------------------------------------------------
if "sentence_builder" not in st.session_state:
    st.session_state.sentence_builder = SentenceBuilder()

if "smoother" not in st.session_state:
    st.session_state.smoother = PredictionSmoother()

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = TTSEngine()

if "landmark_classifier" not in st.session_state:
    st.session_state.landmark_classifier = LandmarkSignClassifier()

if "copy_success" not in st.session_state:
    st.session_state.copy_success = False

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [
        {"sender": "hearing", "text": "Hello! I can speak and you'll see my words live on your screen.", "time": "Initial"},
        {"sender": "deaf", "text": "HELLO NICE TO MEET YOU", "time": "Initial"}
    ]


@st.cache_resource(show_spinner="⚡ Loading ASL Neural Network Model...")
def get_cached_predictor() -> SignPredictor:
    """Loads and caches the neural network inference engine."""
    return SignPredictor()


# -----------------------------------------------------------------------------
# Sidebar: Navigation & Runtime Hyperparameters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ SignBridge Settings")

    nav_mode = st.radio(
        "Application Mode",
        [
            "🎥 Live Camera Studio",
            "🤝 Two-Way Conversation Bridge",
            "🖼️ Image Classification",
            "📊 Diagnostics & Benchmark",
            "📖 ASL Alphabet Guide"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 🎛️ Inference Controls")

    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.78,
        step=0.02,
        help="Minimum neural network confidence required to register a gesture"
    )

    window_size = st.slider(
        "Smoothing Window (Frames)",
        min_value=3,
        max_value=21,
        value=9,
        step=2,
        help="Number of continuous frames evaluated for temporal majority voting"
    )

    hold_duration = st.slider(
        "Gesture Hold Frames",
        min_value=5,
        max_value=30,
        value=12,
        step=1,
        help="How long a stable gesture must be held before adding to sentence"
    )

    cam_index = st.selectbox(
        "Webcam Device Index",
        options=[0, 1, 2],
        index=0,
        help="0 is built-in laptop camera; 1 or 2 for external USB webcams"
    )

    show_landmarks = st.checkbox("Show MediaPipe Landmarks", value=True)

    prediction_engine = st.selectbox(
        "Prediction Engine",
        [
            "✨ Hybrid Ensemble (CNN + 3D Landmarks)",
            "📐 MediaPipe 3D Landmarks (Room Invariant)",
            "🧠 Deep CNN (EfficientNetB0 Image Model)"
        ],
        index=0,
        help="Hybrid combines neural network with 3D finger landmark geometry, eliminating background confusion."
    )

    auto_speak = st.checkbox(
        "🔊 Auto-Speak on Word / Space",
        value=False,
        help="Automatically speaks completed words out loud when you sign Space."
    )

    auto_accept_on_space = st.checkbox(
        "⚡ Auto-Complete on Space Gesture",
        value=True,
        help="When enabled, signing the Space gesture automatically accepts the top suggested word."
    )

    phrase_mode = st.checkbox(
        "🤟 Whole-Word Signs (ILY, Yes, No, Stop, Okay)",
        value=True,
        help="Instantly recognizes common ASL word gestures without spelling letter-by-letter."
    )

    st.markdown("---")
    st.markdown("""
    **Hardware & Engine:**
    - Backbone: `EfficientNetB0`
    - Preprocessing: RGB [0, 255]
    - Classes: 29 (A–Z, Space, Del, Nothing)
    - Architecture: Mobile-Optimized CNN
    """)


# Update state hyperparameters
st.session_state.smoother.confidence_threshold = conf_threshold
st.session_state.smoother.window_size = window_size
st.session_state.sentence_builder.hold_duration_frames = hold_duration
st.session_state.sentence_builder.auto_accept_on_space = auto_accept_on_space


# -----------------------------------------------------------------------------
# Header Hero Section
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">SignBridge AI</div>
    <div class="hero-subtitle">Real-Time American Sign Language (ASL) Recognition & Assisted Communication</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODE 1: Live Camera Studio
# -----------------------------------------------------------------------------
if nav_mode == "🎥 Live Camera Studio":
    try:
        predictor = get_cached_predictor()
    except Exception as e:
        st.error(f"❌ Model initialization error: {e}")
        st.stop()

    col_view, col_ctrl = st.columns([1.6, 1.0])

    with col_ctrl:
        st.markdown("### 🎛️ Session Controls")

        # Camera action buttons
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if not st.session_state.camera_running:
                if st.button("▶️ Start Camera", type="primary", use_container_width=True):
                    st.session_state.camera_running = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Camera", type="secondary", use_container_width=True):
                    st.session_state.camera_running = False
                    st.rerun()

        with c_btn2:
            if st.button("🧹 Clear Text", use_container_width=True):
                st.session_state.sentence_builder.clear()
                st.rerun()

        # Text editing controls
        e_btn1, e_btn2, e_btn3 = st.columns(3)
        with e_btn1:
            if st.button("↩️ Undo", use_container_width=True):
                st.session_state.sentence_builder.undo()
                st.rerun()
        with e_btn2:
            if st.button("🔊 Speak", use_container_width=True):
                txt = st.session_state.sentence_builder.get_text()
                if txt:
                    st.session_state.tts_engine.speak(txt)
        with e_btn3:
            if st.button("📋 Copy", use_container_width=True):
                txt = st.session_state.sentence_builder.get_text()
                if txt:
                    st.toast("Copied sentence to clipboard!")

        # Sentence Display Box
        st.markdown("#### 💬 Recognized Text:")
        sentence_placeholder = st.empty()
        current_sentence = st.session_state.sentence_builder.get_text()
        display_str = current_sentence if current_sentence else "Start signing to build text..."
        sentence_placeholder.markdown(f'<div class="sentence-display">{display_str}</div>', unsafe_allow_html=True)

        # Smart Autocomplete Suggestions UI
        suggestions_placeholder = st.empty()
        suggestions = st.session_state.sentence_builder.get_suggestions()
        if suggestions:
            chips_html = []
            for i, s in enumerate(suggestions):
                chip_cls = "suggestion-chip-top" if i == 0 else "suggestion-chip"
                hint = " [Space]" if i == 0 and auto_accept_on_space else ""
                chips_html.append(f'<span class="{chip_cls}"><b>{i+1}.</b> {s}{hint}</span>')
            chips_str = "".join(chips_html)
            suggestions_placeholder.markdown(
                f'<div class="suggestions-container"><span class="sugg-label">💡 Autocomplete:</span> {chips_str}</div>',
                unsafe_allow_html=True
            )
        else:
            suggestions_placeholder.empty()

        # Clickable Suggestion Chips for instant tap acceptance
        if suggestions:
            btn_cols = st.columns(len(suggestions))
            for idx, sugg in enumerate(suggestions):
                with btn_cols[idx]:
                    if st.button(f"{idx+1}. {sugg}", key=f"btn_sugg_{idx}_{sugg}", use_container_width=True):
                        st.session_state.sentence_builder.accept_suggestion(sugg)
                        st.rerun()

        # Quick Gestural Shortcuts expander
        with st.expander("⚡ Quick Gestural Shortcuts & Whole-Word Signs", expanded=False):
            st.markdown("""
            **Instant Whole-Word Signs (No Spelling Required):**
            - 🤟 **`I LOVE YOU`** — Thumb, index, and pinky extended; middle & ring curled
            - 👍 **`YES`** — Closed fist with thumb pointing straight UP high
            - 👎 **`NO`** — Closed fist with thumb pointing straight DOWN low
            - ✋ **`STOP`** — Flat open upright palm facing camera
            - 👌 **`OKAY`** — Index tip & thumb tip touching in a circle, other 3 fingers extended

            **Letter Abbreviations (Sign letters & sign Space to expand):**
            - **`TY`** ➔ *THANK YOU* &nbsp;|&nbsp; **`HRU`** ➔ *HOW ARE YOU* &nbsp;|&nbsp; **`GM`** ➔ *GOOD MORNING*
            - **`GN`** ➔ *GOOD NIGHT* &nbsp;|&nbsp; **`SOS`** ➔ *EMERGENCY NEED HELP* &nbsp;|&nbsp; **`PLS`** ➔ *PLEASE*
            - **`ILY`** ➔ *I LOVE YOU* &nbsp;|&nbsp; **`IDK`** ➔ *I DON'T KNOW* &nbsp;|&nbsp; **`NP`** ➔ *NO PROBLEM*
            - **`CU`** ➔ *SEE YOU* &nbsp;|&nbsp; **`TC`** ➔ *TAKE CARE* &nbsp;|&nbsp; **`YW`** ➔ *YOU'RE WELCOME*
            """)

        # Real-time Metrics Card
        st.markdown("<br>", unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Current Sign</div>
                <div class="metric-value" id="current_sign">—</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value" id="current_conf">—%</div>
            </div>
            """, unsafe_allow_html=True)

        # Recent Tokens Log
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📜 Recent Gestures History")
        recent = st.session_state.sentence_builder.recent_tokens
        if recent:
            badge_tags = " ".join([f"`{t}`" for t in recent[-12:]])
            st.markdown(badge_tags)
        else:
            st.caption("No gestures recorded yet.")

    with col_view:
        st.markdown("### 📹 Live Feed Preview")
        video_placeholder = st.empty()
        status_placeholder = st.empty()

        if st.session_state.camera_running:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                st.error(f"❌ Could not access webcam at index {cam_index}. Check permissions in System Settings -> Privacy & Security -> Camera.")
                st.session_state.camera_running = False
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                mp_hands = mp.solutions.hands
                hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.65,
                    min_tracking_confidence=0.65
                )
                mp_draw = mp.solutions.drawing_utils

                # Live webcam loop
                stop_requested = False
                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        status_placeholder.warning("⚠️ Failed to capture frame from webcam.")
                        break

                    # Mirror naturally
                    frame = cv2.flip(frame, 1)
                    clean_frame = frame.copy()
                    rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
                    
                    results = None
                    try:
                        results = hands.process(rgb)
                    except Exception:
                        results = None

                    roi = None
                    bbox = None
                    is_hand = False

                    if results and results.multi_hand_landmarks:
                        is_hand = True
                        hand_landmarks = results.multi_hand_landmarks[0]
                        # Extract clean ROI from untouched frame first
                        roi, bbox = extract_hand_roi(clean_frame, hand_landmarks, padding_ratio=0.15)
                        # Then draw visual landmarks overlay onto display frame
                        if show_landmarks:
                            try:
                                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            except Exception:
                                pass

                    if roi is None or bbox is None:
                        roi, bbox = get_center_roi(clean_frame, box_ratio=0.50)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (90, 90, 180), 2)
                    else:
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 220, 100), 2)

                    # Predict
                    raw_lbl = "Uncertain"
                    raw_cnf = 0.0
                    top_candidates_str = ""

                    if is_hand and hand_landmarks:
                        lm_lbl, lm_cnf, lm_cands = st.session_state.landmark_classifier.classify_landmarks(
                            hand_landmarks,
                            phrase_mode=phrase_mode
                        )

                        if "Landmarks" in prediction_engine:
                            raw_lbl = lm_lbl
                            raw_cnf = lm_cnf
                            top_candidates_str = ", ".join([f"{l}: {c*100:.1f}%" for l, c in lm_cands])
                        elif "Deep CNN" in prediction_engine:
                            if roi is not None and roi.size > 0:
                                try:
                                    pred = predictor.predict_roi(roi, is_bgr=True, top_k_num=3)
                                    raw_lbl = pred.label
                                    raw_cnf = pred.confidence
                                    top_candidates_str = ", ".join([f"{l.upper()}: {c*100:.1f}%" for l, c in pred.top_k])
                                except Exception:
                                    pass
                        else:  # ✨ Hybrid Ensemble
                            if roi is not None and roi.size > 0:
                                try:
                                    pred = predictor.predict_roi(roi, is_bgr=True, top_k_num=3)
                                    cnn_lbl = pred.label.upper()
                                    cnn_cnf = pred.confidence

                                    # If landmarks classifier has high certainty (e.g., U, V, W, B, I, F, L)
                                    if lm_cnf >= 0.85:
                                        raw_lbl = lm_lbl
                                        raw_cnf = 0.70 * lm_cnf + 0.30 * cnn_cnf
                                        top_candidates_str = f"{lm_lbl} (Geo: {lm_cnf*100:.0f}%), {cnn_lbl} (CNN: {cnn_cnf*100:.0f}%)"
                                    else:
                                        raw_lbl = cnn_lbl
                                        raw_cnf = cnn_cnf
                                        top_candidates_str = ", ".join([f"{l.upper()}: {c*100:.1f}%" for l, c in pred.top_k])
                                except Exception:
                                    raw_lbl = lm_lbl
                                    raw_cnf = lm_cnf
                                    top_candidates_str = ", ".join([f"{l}: {c*100:.1f}%" for l, c in lm_cands])
                            else:
                                raw_lbl = lm_lbl
                                raw_cnf = lm_cnf
                                top_candidates_str = ", ".join([f"{l}: {c*100:.1f}%" for l, c in lm_cands])

                    # Stabilize
                    stable_lbl, stable_cnf, is_stable = st.session_state.smoother.update(
                        raw_lbl if is_hand else "nothing",
                        raw_cnf if is_hand else 0.0
                    )

                    # Commit to sentence builder
                    committed_token = st.session_state.sentence_builder.update(stable_lbl, is_stable)
                    current_sentence = st.session_state.sentence_builder.get_text()

                    # Live update the text display box & suggestions
                    disp_str = current_sentence if current_sentence else "Start signing to build text..."
                    sentence_placeholder.markdown(f'<div class="sentence-display">{disp_str}</div>', unsafe_allow_html=True)

                    suggs = st.session_state.sentence_builder.get_suggestions()
                    if suggs:
                        chips_html = []
                        for i, s in enumerate(suggs):
                            chip_cls = "suggestion-chip-top" if i == 0 else "suggestion-chip"
                            hint = " [Space]" if i == 0 and auto_accept_on_space else ""
                            chips_html.append(f'<span class="{chip_cls}"><b>{i+1}.</b> {s}{hint}</span>')
                        chips_str = "".join(chips_html)
                        suggestions_placeholder.markdown(
                            f'<div class="suggestions-container"><span class="sugg-label">💡 Autocomplete:</span> {chips_str}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        suggestions_placeholder.empty()

                    # Auto-speak if enabled and a word was completed (Space or Accepted Suggestion)
                    if auto_speak and (committed_token == "[SPACE]" or (committed_token and committed_token.startswith("[") and committed_token.endswith("]"))):
                        words = current_sentence.strip().split()
                        if words:
                            st.session_state.tts_engine.speak(words[-1])

                    # Draw on-screen HUD banner on camera feed
                    h, w, _ = frame.shape
                    cv2.rectangle(frame, (0, h - 50), (w, h), (18, 22, 28), -1)
                    hud_txt = current_sentence if current_sentence else "Signing..."
                    top_hint = f"  |  Next: {suggs[0]} [Space]" if (suggs and auto_accept_on_space) else ""
                    cv2.putText(frame, f"Text: {hud_txt}{top_hint}", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    # Draw hold progress bar
                    prog = st.session_state.sentence_builder.get_progress_ratio()
                    if prog > 0:
                        bar_w = int(120 * prog)
                        cv2.rectangle(frame, (w - 140, h - 35), (w - 140 + bar_w, h - 20), (0, 220, 100), -1)
                        cv2.rectangle(frame, (w - 140, h - 35), (w - 20, h - 20), (120, 120, 120), 1)

                    # Display frame in Streamlit
                    frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_disp, channels="RGB", use_column_width=True)

                    # Status pill & Top 3 Candidates
                    status_badge = '<span class="badge-stable">🟢 STABLE GESTURE</span>' if is_stable else '<span class="badge-uncertain">🟡 UNCERTAIN / SEARCHING</span>'
                    candidates_display = f" &nbsp;|&nbsp; **Top Candidates:** `{top_candidates_str}`" if top_candidates_str else ""
                    status_placeholder.markdown(
                        f"{status_badge} &nbsp;|&nbsp; **Sign:** `{stable_lbl.upper()}` &nbsp;|&nbsp; **Confidence:** `{stable_cnf*100:.1f}%`{candidates_display}",
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)

                cap.release()
                hands.close()
        else:
            video_placeholder.info("💡 Camera is currently offline. Click **Start Camera** on the right to begin live recognition, or use the standalone desktop HUD via `python -m src.webcam`.")


# -----------------------------------------------------------------------------
# MODE 2: Two-Way Conversation Bridge
# -----------------------------------------------------------------------------
elif nav_mode == "🤝 Two-Way Conversation Bridge":
    st.markdown("### 🤝 Two-Way Conversation Bridge")
    st.markdown("A bidirectional real-time communication station bridging hearing and deaf/mute individuals seamlessly.")

    col_hearing, col_deaf = st.columns([1, 1], gap="large")

    # ---------------------------------------------------------
    # STATION 1: Hearing Partner Station (Speech-to-Text)
    # ---------------------------------------------------------
    with col_hearing:
        st.markdown("""
        <div class="station-card" style="border-top: 4px solid #58a6ff;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #58a6ff;">
                    👂 Hearing Partner Station
                </div>
                <span class="badge-stable" style="background: rgba(56, 139, 253, 0.2); color: #58a6ff; border-color: rgba(56, 139, 253, 0.4);">
                    Speech-to-Text Active
                </span>
            </div>
            <div style="color: #8b949e; font-size: 0.88rem; margin-top: 6px;">
                Speak into the microphone or type below. Spoken words are converted into large live captions on screen for the deaf partner to read immediately.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Embedded Native Web Speech Recognition Widget (Zero-latency HTML5 microphone)
        st.markdown("##### 🎙️ Browser Live Microphone Captions")
        st.components.v1.html("""
        <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 16px; color: #e6edf3; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <span id="micStatus" style="font-size: 0.88rem; font-weight: 600; color: #8b949e;">
                    ⚪ Microphone idle. Click 'Start Listening' to capture speech.
                </span>
                <button id="micBtn" onclick="toggleSpeech()" style="background: #238636; border: none; color: white; padding: 8px 18px; border-radius: 8px; font-weight: 700; cursor: pointer; transition: 0.2s; font-size: 0.9rem;">
                    🎤 Start Listening
                </button>
            </div>
            <div id="liveCaption" style="background: #161b22; border: 1px solid #388bfd; border-radius: 8px; padding: 14px; font-size: 1.25rem; font-weight: 600; color: #7ee787; min-height: 65px; line-height: 1.45; word-wrap: break-word;">
                (Spoken words will stream here in real time...)
            </div>
        </div>
        <script>
            let recognition = null;
            let active = false;
            const btn = document.getElementById('micBtn');
            const status = document.getElementById('micStatus');
            const caption = document.getElementById('liveCaption');

            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                recognition.continuous = true;
                recognition.interimResults = true;
                recognition.lang = 'en-US';

                recognition.onstart = function() {
                    active = true;
                    btn.textContent = '⏹️ Stop Listening';
                    btn.style.background = '#da3633';
                    status.textContent = '🟢 Listening actively to partner speech...';
                    status.style.color = '#7ee787';
                };

                recognition.onresult = function(event) {
                    let str = '';
                    for (let i = 0; i < event.results.length; ++i) {
                        str += event.results[i][0].transcript + ' ';
                    }
                    caption.textContent = str.trim() || '(Listening...)';
                };

                recognition.onerror = function(e) {
                    status.textContent = '⚠️ Mic notice: ' + (e.error || 'unspecified');
                    active = false;
                    btn.textContent = '🎤 Start Listening';
                    btn.style.background = '#238636';
                };

                recognition.onend = function() {
                    if (active) {
                        try { recognition.start(); } catch(err) {}
                    } else {
                        btn.textContent = '🎤 Start Listening';
                        btn.style.background = '#238636';
                        status.textContent = '⚪ Microphone paused.';
                        status.style.color = '#8b949e';
                    }
                };
            } else {
                status.textContent = 'ℹ️ Web Speech API not supported in this browser. Use manual speech box below.';
                btn.disabled = true;
                btn.style.opacity = '0.5';
            }

            function toggleSpeech() {
                if (!recognition) return;
                if (active) {
                    active = false;
                    recognition.stop();
                } else {
                    recognition.start();
                }
            }
        </script>
        """, height=160)

        # Hearing partner text input fallback / add to transcript
        with st.form("hearing_input_form", clear_on_submit=True):
            hearing_text = st.text_input(
                "💬 Or Type Hearing Partner Statement (Noisy Room Fallback):",
                placeholder="e.g. Can I help you with something? How can I assist you?",
                key="hearing_typed_box"
            )
            submit_hearing = st.form_submit_button("📢 Post to Caption & Transcript", use_container_width=True)
            if submit_hearing and hearing_text.strip():
                now_str = time.strftime("%H:%M:%S")
                st.session_state.conversation_history.append({
                    "sender": "hearing",
                    "text": hearing_text.strip(),
                    "time": now_str
                })
                st.rerun()

        # Display latest hearing caption in high contrast
        hearing_msgs = [m for m in st.session_state.conversation_history if m["sender"] == "hearing"]
        latest_hearing = hearing_msgs[-1]["text"] if hearing_msgs else "Awaiting partner speech..."
        st.markdown(f"""
        <div class="caption-highlight">
            <div style="font-size: 0.8rem; text-transform: uppercase; color: #58a6ff; font-weight: 700; margin-bottom: 6px;">
                👂 Latest Spoken Statement (For Deaf Partner to Read):
            </div>
            "{latest_hearing}"
        </div>
        """, unsafe_allow_html=True)


    # ---------------------------------------------------------
    # STATION 2: Deaf / Mute Partner Station (Sign & Quick Reply)
    # ---------------------------------------------------------
    with col_deaf:
        st.markdown("""
        <div class="station-card" style="border-top: 4px solid #7ee787;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="font-size: 1.25rem; font-weight: 700; color: #7ee787;">
                    🤟 Deaf / Mute Partner Station
                </div>
                <span class="badge-stable" style="background: rgba(46, 160, 67, 0.2); color: #7ee787; border-color: rgba(46, 160, 67, 0.4);">
                    Response Station
                </span>
            </div>
            <div style="color: #8b949e; font-size: 0.88rem; margin-top: 6px;">
                Reply instantly using 1-tap smart responses or sign your reply in the camera studio. Responses are spoken aloud to the hearing partner via Text-to-Speech!
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Response Grid (One-Tap Smart Replies)
        st.markdown("##### ⚡ One-Tap Smart Quick Replies")
        qr_cols = st.columns(4)
        quick_replies = [
            ("👍 YES", "YES, ABSOLUTELY"),
            ("👎 NO", "NO, THANK YOU"),
            ("🙏 THANK YOU", "THANK YOU VERY MUCH"),
            ("🤝 PLEASE HELP", "PLEASE HELP ME WITH THIS"),
            ("👂 REPEAT", "COULD YOU PLEASE REPEAT THAT?"),
            ("⏳ WAIT", "PLEASE GIVE ME A MOMENT"),
            ("👋 HELLO", "HELLO! NICE TO MEET YOU"),
            ("😊 I UNDERSTAND", "I UNDERSTAND PERFECTLY")
        ]
        for idx, (label, phrase) in enumerate(quick_replies):
            with qr_cols[idx % 4]:
                if st.button(label, key=f"qr_{idx}_{label}", use_container_width=True):
                    now_str = time.strftime("%H:%M:%S")
                    st.session_state.conversation_history.append({
                        "sender": "deaf",
                        "text": phrase,
                        "time": now_str
                    })
                    st.session_state.tts_engine.speak(phrase)
                    st.rerun()

        # Deaf custom compose or camera buffer input
        current_deaf_buffer = st.session_state.sentence_builder.get_text()
        with st.form("deaf_reply_form"):
            reply_val = st.text_input(
                "✍️ Composed / Signed Response Buffer:",
                value=current_deaf_buffer,
                placeholder="Type your reply or use Live Camera Studio to sign...",
                key="deaf_reply_field"
            )

            d_col1, d_col2 = st.columns([1.5, 1])
            with d_col1:
                send_deaf_reply = st.form_submit_button("📤 Send Reply & Speak Aloud 🔊", type="primary", use_container_width=True)
            with d_col2:
                clear_deaf_buffer = st.form_submit_button("🧹 Clear Buffer", use_container_width=True)

            if send_deaf_reply and reply_val.strip():
                now_str = time.strftime("%H:%M:%S")
                clean_text = reply_val.strip().upper()
                st.session_state.conversation_history.append({
                    "sender": "deaf",
                    "text": clean_text,
                    "time": now_str
                })
                st.session_state.tts_engine.speak(clean_text)
                st.session_state.sentence_builder.clear()
                st.rerun()
            elif clear_deaf_buffer:
                st.session_state.sentence_builder.clear()
                st.rerun()

        # Autocomplete suggestions preview
        suggs = st.session_state.sentence_builder.get_suggestions()
        if suggs:
            s_chips = " &nbsp;|&nbsp; ".join([f"`{s}`" for s in suggs[:4]])
            st.markdown(f"<span style='color: #8b949e; font-size: 0.85rem;'>💡 <b>Autocomplete Hints:</b> {s_chips}</span>", unsafe_allow_html=True)


    # ---------------------------------------------------------
    # SHARED CONVERSATION TRANSCRIPT
    # ---------------------------------------------------------
    st.markdown("---")
    t_head, t_act = st.columns([1.5, 1])
    with t_head:
        st.markdown("### 💬 Live Conversation Dialogue Transcript")
    with t_act:
        act1, act2 = st.columns(2)
        with act1:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.conversation_history.clear()
                st.rerun()
        with act2:
            export_text = "\n".join([
                f"[{m.get('time', '')}] {('Hearing Partner' if m['sender'] == 'hearing' else 'Deaf Partner')}: {m['text']}"
                for m in st.session_state.conversation_history
            ])
            st.download_button(
                "📥 Export (.txt)",
                data=export_text,
                file_name="signbridge_conversation_transcript.txt",
                mime="text/plain",
                use_container_width=True
            )

    # Render Conversation Chat Bubbles
    if st.session_state.conversation_history:
        chat_html = ['<div class="chat-container">']
        for msg in st.session_state.conversation_history:
            sender = msg["sender"]
            text = msg["text"]
            t_str = msg.get("time", "")
            if sender == "hearing":
                bubble_cls = "chat-bubble-hearing"
                sender_cls = "chat-sender chat-sender-hearing"
                sender_title = f"👂 Hearing Partner ({t_str})"
            else:
                bubble_cls = "chat-bubble-deaf"
                sender_cls = "chat-sender chat-sender-deaf"
                sender_title = f"🤟 Deaf / Mute Partner ({t_str})"

            chat_html.append(f"""
            <div class="{bubble_cls}">
                <div class="{sender_cls}">{sender_title}</div>
                <div class="chat-text">{text}</div>
            </div>
            """)
        chat_html.append('</div>')
        st.markdown("".join(chat_html), unsafe_allow_html=True)
    else:
        st.info("No messages in conversation yet. Speak into the microphone or send a quick reply!")


# -----------------------------------------------------------------------------
# MODE 3: Image Classification (File Upload & Test)
# -----------------------------------------------------------------------------
elif nav_mode == "🖼️ Image Classification":
    st.markdown("### 🖼️ Static ASL Image Recognition & Verification")
    st.markdown("Upload a photo of an ASL hand gesture or select a sample image from the dataset.")

    try:
        predictor = get_cached_predictor()
    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

    # Provide quick sample selector
    sample_dir = CONFIG.dataset.test_dir
    sample_files = list(sample_dir.glob("*.jpg")) if sample_dir.exists() else []

    selected_sample = None
    if sample_files:
        st.markdown("Or pick a sample test file:")
        sample_names = ["-- Select a test sample --"] + [f.name for f in sample_files[:15]]
        chosen = st.selectbox("Sample files", sample_names)
        if chosen != sample_names[0]:
            selected_sample = sample_dir / chosen

    img_to_process = None
    if uploaded_file is not None:
        img_to_process = Image.open(uploaded_file)
    elif selected_sample is not None:
        img_to_process = Image.open(selected_sample)

    if img_to_process is not None:
        col_img, col_res = st.columns([1, 1.2])
        np_img = np.array(img_to_process.convert("RGB"))

        with col_img:
            st.image(np_img, caption="Analyzed Image", use_column_width=True)

        with col_res:
            res = predictor.predict_roi(np_img, is_bgr=False, top_k_num=5)

            st.markdown(f"### Predicted Class: **{res.label.upper()}**")
            st.progress(min(1.0, float(res.confidence)))
            st.markdown(f"**Confidence:** `{res.confidence*100:.2f}%` &nbsp;|&nbsp; **Latency:** `{res.latency_ms:.2f} ms`")

            st.markdown("#### Top 5 Candidates:")
            df = pd.DataFrame({
                "Class": [c.upper() for c, _ in res.top_k],
                "Confidence": [f"{s*100:.2f}%" for _, s in res.top_k],
                "Probability": [s for _, s in res.top_k]
            })
            st.bar_chart(df.set_index("Class")["Probability"])

            if st.button("➕ Append to Recognized Sentence"):
                st.session_state.sentence_builder._commit_token(res.label)
                st.success(f"Added '{res.label.upper()}' to text buffer!")


# -----------------------------------------------------------------------------
# MODE 3: Diagnostics & Benchmark
# -----------------------------------------------------------------------------
elif nav_mode == "📊 Diagnostics & Benchmark":
    st.markdown("### 📊 System Diagnostics & Model Evaluation")

    diag_mode = st.radio(
        "Select Model Architecture to Inspect:",
        [
            "📐 3D Landmark ML Classifier (New Model — High Accuracy)",
            "🧠 Deep CNN (EfficientNetB0 Image Model)"
        ],
        horizontal=True
    )

    if "3D Landmark" in diag_mode:
        metrics_file = CONFIG.results_dir / "landmark_evaluation_metrics.json"
        cm_plot_file = CONFIG.results_dir / "landmark_confusion_matrix.png"
        model_desc = "3D Landmark Multi-Layer Classifier evaluated on normalized invariant bone coordinates."
    else:
        metrics_file = CONFIG.results_dir / "evaluation_metrics.json"
        cm_plot_file = CONFIG.results_dir / "confusion_matrix.png"
        model_desc = "EfficientNetB0 2D Convolutional Neural Network evaluated on raw pixel images."

    st.caption(model_desc)

    if metrics_file.exists():
        import json
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Overall Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        m2.metric("Macro F1-Score", f"{metrics.get('macro_f1', 0):.4f}")
        m3.metric("Inference Latency", f"{metrics.get('mean_latency_ms', 0)} ms")
        m4.metric("Model Size", f"{metrics.get('model_size_mb', 0)} MB")

    if cm_plot_file.exists():
        st.markdown("#### 🎯 Confusion Matrix Heatmap")
        st.image(str(cm_plot_file), caption=f"Confusion Matrix Heatmap ({diag_mode})", use_column_width=True)
    else:
        st.info("ℹ️ Evaluation metrics for this model are being generated.")


# -----------------------------------------------------------------------------
# MODE 4: ASL Alphabet Reference Guide
# -----------------------------------------------------------------------------
elif nav_mode == "📖 ASL Alphabet Guide":
    st.markdown("### 📖 American Sign Language Alphabet Reference")
    st.markdown("American Sign Language uses 26 hand configurations to represent the English alphabet, alongside gestures for Space, Delete, and neutral poses.")

    letters = [
        ("A", "Fist with thumb resting along side of index finger"),
        ("B", "Open palm facing forward, thumb folded across palm"),
        ("C", "Fingers curved into a 'C' shape"),
        ("D", "Index finger pointed up, other fingers forming circle with thumb"),
        ("E", "Fingers curled with fingertips touching thumb"),
        ("F", "Index and thumb form circle, other three fingers extended"),
        ("G", "Index and thumb parallel pointing horizontally"),
        ("H", "Index and middle fingers extended together horizontally"),
        ("I", "Pinky finger extended upward"),
        ("J", "Pinky finger traces a 'J' in the air"),
        ("K", "Index pointing up, middle forward, thumb resting on middle"),
        ("L", "Index and thumb form an 'L' shape"),
        ("M", "Thumb folded under three fingers"),
        ("N", "Thumb folded under two fingers"),
        ("O", "All fingers curved touching thumb to form an 'O'"),
        ("P", "Inverted 'K' pointing downward"),
        ("Q", "Inverted 'G' pointing downward"),
        ("R", "Index and middle fingers crossed"),
        ("S", "Fist with thumb wrapped across front of fingers"),
        ("T", "Thumb placed between index and middle finger"),
        ("U", "Index and middle fingers extended together vertically"),
        ("V", "Index and middle fingers separated into a 'V'"),
        ("W", "Index, middle, and ring fingers extended ('W')"),
        ("X", "Index finger bent into a hook"),
        ("Y", "Thumb and pinky extended ('hang loose')"),
        ("Z", "Index finger traces a 'Z' in the air"),
        ("del", "Palm held flat or waving to erase last character"),
        ("nothing", "Neutral relaxed hand resting or out of frame"),
        ("space", "Horizontal open palm swipe indicating word separation")
    ]

    cols = st.columns(3)
    for idx, (char, desc) in enumerate(letters):
        col = cols[idx % 3]
        with col:
            st.markdown(f"**`{char.upper()}`** — {desc}")


# -----------------------------------------------------------------------------
# Educational Linguistic Disclaimer (Mandatory)
# -----------------------------------------------------------------------------
st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Important Linguistic Distinction:</strong><br>
    SignBridge AI implements <em>finger-spelling alphabet recognition</em> (letter-by-letter transcription).
    Finger-spelling is a subset of ASL used primarily for proper nouns and technical terms.
    It is <strong>not</strong> equivalent to continuous American Sign Language (ASL) translation,
    which has its own distinct grammar, non-manual facial markers, and spatial topic-comment syntax.
</div>
""", unsafe_allow_html=True)
