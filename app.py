from pathlib import Path
import sys
import base64
import html
import tempfile
import json
from datetime import datetime

import streamlit as st
from PIL import Image
from blockchain import add_record, verify_chain

import blockchain
from scripts.rsa_signature import verify_signature

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from app_inference import detect_currency

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Indian Currency Detector",
    page_icon="₹",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "main"

if "history_serial" not in st.session_state:
    st.session_state.history_serial = ""

if "result" not in st.session_state:
    st.session_state.result = None

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None
    
if "blockchain_saved" not in st.session_state:
    st.session_state.blockchain_saved = False

# ============================================================
# BACKGROUND IMAGE
# ============================================================

BACKGROUND_IMAGE = PROJECT_ROOT / "assets" / "image.png"

if BACKGROUND_IMAGE.exists():

    try:
        with open(BACKGROUND_IMAGE, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        background_style = f"""
        background-image:
            linear-gradient(
                rgba(255, 255, 255, 0.55),
                rgba(255, 255, 255, 0.62)
            ),
            url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        """

    except Exception:
        background_style = """
        background:
            linear-gradient(
                135deg,
                #f4f6f8,
                #e8edf3
            );
        """

else:

    background_style = """
    background:
        linear-gradient(
            135deg,
            #f4f6f8,
            #e8edf3
        );
    """

# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
<style>

/* ==========================================================
   MAIN PAGE
   ========================================================== */

.stApp {{
    {background_style}
}}

.block-container {{
    max-width: 1000px;
    padding-top: 1.2rem;
    padding-bottom: 1rem;
}}

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

header {{
    background: transparent !important;
}}

/* ==========================================================
   TITLE
   ========================================================== */

.app-title {{
    text-align: center;
    color: #111827;
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -1px;
    margin-top: 0;
    margin-bottom: 18px;
    text-shadow: 0 2px 8px rgba(255,255,255,0.9);
}}

/* ==========================================================
   UPLOAD LABEL
   ========================================================== */

.upload-label {{
    color: #172033 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    margin-bottom: 5px !important;
}}

/* ==========================================================
   STREAMLIT SELECTBOX
   ========================================================== */

[data-testid="stSelectbox"] label {{
    color: #172033 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    margin-bottom: 5px !important;
}}

[data-testid="stSelectbox"] div[data-baseweb="select"] {{
    background: #172033 !important;
    border-radius: 10px !important;
    color: white !important;
}}

[data-testid="stTextInput"] label {{
    color: #172033 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
}}

[data-testid="stTextInput"] input {{
    background: #172033 !important;
    color: white !important;
    border-radius: 10px !important;
}}

[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
    color: white !important;
}}

[data-testid="stSelectbox"] input {{
    color: white !important;
    background: #172033 !important;
}}

[data-testid="stSelectbox"] svg {{
    fill: white !important;
}}

div[data-baseweb="popover"] {{
    background: #172033 !important;
}}

div[data-baseweb="popover"] * {{
    color: white !important;
}}

div[role="option"] {{
    color: white !important;
    background: #172033 !important;
}}

div[role="option"]:hover {{
    background: #1d4ed8 !important;
    color: white !important;
}}

/* PRESS ENTER TO APPLY */
[data-testid="stTextInput"] p {{
    color: #172033 !important;
    font-weight: 800 !important;
    font-size: 15px !important;
}}

div[data-testid="stCheckbox"] label {{
    color: #172033 !important;
    font-weight: 600 !important;
}}

/* ============================================================
   DETECT CURRENCY BUTTON
   ============================================================ */

/* BUTTONS */

div.stButton > button {{
    background: linear-gradient(
        180deg,
        #1d4ed8,
        #172033
    ) !important;

    color: white !important;

    border: 2px solid #0f172a !important;

    border-radius: 12px !important;

    font-weight: 700 !important;

    min-height: 46px !important;

    box-shadow:
        0 4px 10px rgba(0,0,0,0.18) !important;

    transition: all 0.2s ease !important;
}}

/* HOVER */

div.stButton > button:hover {{

    background: linear-gradient(
        180deg,
        #2563eb,
        #1d4ed8
    ) !important;

    color: white !important;

    border: 2px solid #1d4ed8 !important;

    transform: translateY(-2px);

    box-shadow:
        0 8px 18px rgba(0,0,0,0.22) !important;
}}

/* CLICKED */

div.stButton > button:active {{

    transform: translateY(2px);

    background: #0f172a !important;

    color: white !important;
}}

/* KEEP TEXT WHITE */

div.stButton > button * {{
    color: white !important;
}}

/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    border: 1px solid #dfe3e8;
    padding: 5px;
}}

[data-testid="stFileUploader"] label {{
    color: #172033 !important;
}}

/* ============================================================
   IMAGE
   ============================================================ */

[data-testid="stImage"] {{
    border-radius: 14px;
    overflow: hidden;
}}

/* ============================================================
   RESULT HEADING
   ============================================================ */

.result-heading {{
    color: #172033;
    font-size: 24px;
    font-weight: 850;
    margin-top: 18px;
    margin-bottom: 12px;
}}

/* ============================================================
   RESULT BOX
   ============================================================ */

.result-box {{
    background: rgba(255,255,255,0.97);
    border: 1px solid #e0e4e8;
    border-radius: 15px;
    padding: 17px 19px;
    min-height: 108px;
    box-shadow: 0 7px 22px rgba(0,0,0,0.08);
}}

.result-label {{
    color: #172033;
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 5px;
}}

.denomination {{
    color: #1d4ed8;
    font-size: 34px;
    line-height: 1.05;
    font-weight: 850;
}}

.real {{
    color: #16875a;
    font-size: 30px;
    line-height: 1.05;
    font-weight: 850;
}}

.fake {{
    color: #d64040;
    font-size: 30px;
    line-height: 1.05;
    font-weight: 850;
}}

.unsupported {{
    color: #a06b00;
    font-size: 22px;
    line-height: 1.1;
    font-weight: 800;
}}

.confidence {{
    color: #172033;
    font-size: 15px;
    font-weight: 800;
    margin-top: 8px;
}}

/* ============================================================
   ANALYSIS
   ============================================================ */

.analysis-title {{
    color: #172033;
    font-size: 15px;
    font-weight: 800;
    margin-top: 14px;
    margin-bottom: 6px;
}}

.analysis-box {{
    background: rgba(255,255,255,0.96);
    border: 1px solid #e1e5eb;
    border-radius: 11px;
    padding: 11px 14px;
    color: #172033;
    font-size: 15px;
    font-weight: 800;
    box-shadow: 0 5px 16px rgba(0,0,0,0.06);
}}

/* ============================================================
   SPEECH
   ============================================================ */

.speech-title {{
    color: #172033 !important;
    font-size: 15px !important;
    font-weight: 800 !important;
    margin-bottom: 6px !important;
}}

/* ============================================================
   FOOTER
   ============================================================ */

.app-footer {{
    text-align: center;
    color: #667085;
    font-size: 11px;
    margin-top: 20px;
}}

</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

if st.session_state.page == "main":

    st.markdown(
        '<div class="app-title">Indian Currency Detector</div>',
        unsafe_allow_html=True,
    )

    # ============================================================
    # SPEECH LANGUAGE
    # ============================================================

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown(
    """
    <p style="
        color:#172033;
        font-weight:800;
        margin-bottom:5px;
    ">
        🗣 Speech Language
    </p>
    """,
    unsafe_allow_html=True
)

        speech_lang = st.selectbox(
    "",
    [
        "English (en-IN)",
        "हिन्दी (hi-IN)",
        "मराठी (mr-IN)",
        "ગુજરાતી (gu-IN)",
        "বাংলা (bn-IN)",
        "తెలుగు (te-IN)",
        "தமிழ் (ta-IN)",
        "ಕನ್ನಡ (kn-IN)",
        "മലയാളം (ml-IN)",
        "नेपाली (ne-NP)",
        "اردو (ur-IN)"
    ],
    label_visibility="collapsed"
        )

    with col2:

        st.markdown(
            "<div style='height:28px'></div>",
            unsafe_allow_html=True,
        )

        speak_clicked = st.button(
            "🔊 Speak Result",
            use_container_width=True
        )

    language_options = {
        "English (en-IN)": "en-IN",
        "हिन्दी (hi-IN)": "hi-IN",
        "मराठी (mr-IN)": "mr-IN",
        "ગુજરાતી (gu-IN)": "gu-IN", 
        "বাংলা (bn-IN)": "bn-IN", 
        "తెలుగు (te-IN)": "te-IN", 
        "தமிழ் (ta-IN)": "ta-IN", 
        "ಕನ್ನಡ (kn-IN)": "kn-IN", 
        "മലയാളം (ml-IN)": "ml-IN", 
        "नेपाली (ne-NP)": "ne-NP", 
        "اردو (ur-IN)": "ur-IN",
    }

    language_code = language_options[speech_lang]

# ============================================================
# REAL-TIME CAMERA DETECTION
# ============================================================

st.markdown(
    """
    <p style="
        color:#172033;
        font-size:18px;
        font-weight:800;
    ">
        📷 Real-Time Camera Detection
    </p>
    """,
    unsafe_allow_html=True
)


camera_mode = st.checkbox(
    "Enable Camera",
    key="camera_mode"
)

frame_holder = st.empty()

camera_image = None

if camera_mode:

    st.markdown(
        '<p style="color:#172033;">Capture Currency Note</p>',
        unsafe_allow_html=True
    )

    camera_image = st.camera_input(
        "",
        key="camera_input"
    )

    # ============================================================
    # UPLOAD
    # ============================================================

st.markdown(
        '<div class="upload-label">📷 Upload Currency Note</div>',
        unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
            "bmp",
        ],
        label_visibility="collapsed",
)

    # ============================================================
    # NEW IMAGE DETECTION
    # ============================================================

if camera_image is not None:

        try:

            camera_image_bytes = camera_image.getbuffer()

            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                dir=PROJECT_ROOT / "streamlit_temp",
                delete=False,
            ) as temporary_camera_file:

                temporary_camera_file.write(camera_image_bytes)
                camera_path = Path(temporary_camera_file.name)

            camera_pil_image = Image.open(
                camera_path
            ).convert("RGB")

            st.session_state.selected_image = camera_pil_image
            st.session_state.uploaded_name = (
                "camera_" + str(camera_image.size)
            )
            st.session_state.result = None
            st.session_state.blockchain_saved = False

            camera_path.unlink(missing_ok=True)

        except Exception as error:

            st.error(
                f"Unable to process camera image: {error}"
            )

            st.stop()

elif uploaded_file is not None:

        if st.session_state.uploaded_name != uploaded_file.name:

            try:

                image = Image.open(uploaded_file).convert("RGB")

                st.session_state.selected_image = image
                st.session_state.uploaded_name = uploaded_file.name
                st.session_state.result = None
                st.session_state.blockchain_saved = False

            except Exception as error:

                st.error(f"Unable to open image: {error}")

                st.stop()

    # ============================================================
    # IMAGE + DETECT BUTTON
    # ============================================================

if st.session_state.selected_image is not None:

        image = st.session_state.selected_image

        image_column, button_column = st.columns(
            [1.45, 0.55],
            gap="large",
        )

        with image_column:

            st.image(
                image,
                use_container_width=True,
            )

        with button_column:

            st.markdown(
                '<div class="upload-label">🔍 Detection</div>',
                unsafe_allow_html=True,
            )

            detect_button = st.button(
                "Analyze Note",
                type="primary",
                use_container_width=True,
            )

        if detect_button:

            temporary_directory = PROJECT_ROOT / "streamlit_temp"

            temporary_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_file = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                dir=temporary_directory,
                delete=False,
            )

            temporary_path = Path(temporary_file.name)

            temporary_file.close()

            try:

                image.save(
                    temporary_path,
                    format="JPEG",
                    quality=95,
                )

                with st.spinner("Analysing currency..."):
                    result = detect_currency(temporary_path)

                st.session_state.result = result

            except Exception as error:

                st.session_state.result = {
                    "success": False,
                    "message": f"Detection error: {error}",
                }

            finally:

                try:
                    if temporary_path.exists():
                        temporary_path.unlink()

                except Exception:
                    pass

    # ============================================================
    # RESULT
    # ============================================================

result = st.session_state.result

    # ============================================================
    # INITIALIZE RESULT VARIABLES
    # ============================================================

denomination = None
authenticity = None
denomination_confidence = 0.0
authenticity_confidence = 0.0

if result is not None:

        if not result.get("success", False):

            st.error(
                result.get(
                    "message",
                    "Currency could not be detected.",
                )
            )

        else:

            # ----------------------------------------------------
            # UNSUPPORTED ₹2000 NOTE CHECK
            # ----------------------------------------------------

            if result.get("denomination") == "2000":
                st.warning("₹2000 notes are not supported.")
                st.stop()

            denomination = result.get("denomination")

            denomination_confidence = float(
                result.get(
                    "denomination_confidence",
                    0,
                )
            )

            authenticity = str(
                result.get(
                    "authenticity",
                    "",
                )
            ).upper()

            authenticity_confidence = float(
                result.get(
                    "authenticity_confidence",
                    0,
                )
            )

            # ----------------------------------------------------
            # Confidence conversion
            # ----------------------------------------------------

            if denomination_confidence <= 1:
                denomination_confidence *= 100

            if authenticity_confidence <= 1:
                authenticity_confidence *= 100

            # ====================================================
            # RESULT HEADING
            # ====================================================

            st.markdown(
                '<div class="result-heading">🎯 Detection Result</div>',
                unsafe_allow_html=True,
            )

            # ====================================================
            # RESULT COLUMNS
            # ====================================================

            col1, col2 = st.columns(
                2,
                gap="medium",
            )

            # ====================================================
            # DENOMINATION
            # ====================================================

            with col1:

                if denomination:

                    safe_denomination = html.escape(str(denomination))

                    result_html = (
                        '<div class="result-box">'
                        '<div class="result-label">'
                        "💵 DENOMINATION"
                        "</div>"
                        '<div class="denomination">'
                        f"₹{safe_denomination}"
                        "</div>"
                        '<div class="confidence">'
                        "Confidence: "
                        f"<b>{denomination_confidence:.2f}%</b>"
                        "</div>"
                        "</div>"
                    )

                else:

                    result_html = (
                        '<div class="result-box">'
                        '<div class="result-label">'
                        "💵 DENOMINATION"
                        "</div>"
                        '<div class="unsupported">'
                        "Not detected"
                        "</div>"
                        "</div>"
                    )

                st.markdown(
                    result_html,
                    unsafe_allow_html=True,
                )

            # ====================================================
            # AUTHENTICITY
            # ====================================================

            with col2:

                if authenticity == "REAL":

                    result_html = (
                        '<div class="result-box">'
                        '<div class="result-label">'
                        "🛡️ AUTHENTICITY"
                        "</div>"
                        '<div class="real">'
                        "✓ REAL"
                        "</div>"
                        '<div class="confidence">'
                        "Confidence: "
                        f"<b>{authenticity_confidence:.2f}%</b>"
                        "</div>"
                        "</div>"
                    )

                elif authenticity == "FAKE":

                    result_html = (
                        '<div class="result-box">'
                        '<div class="result-label">'
                        "🛡️ AUTHENTICITY"
                        "</div>"
                        '<div class="fake">'
                        "! FAKE"
                        "</div>"
                        '<div class="confidence">'
                        "Confidence: "
                        f"<b>{authenticity_confidence:.2f}%</b>"
                        "</div>"
                        "</div>"
                    )

                elif authenticity == "UNSUPPORTED":

                    result_html = (
                        '<div class="result-box">'
                        '<div class="result-label">'
                        "🛡️ AUTHENTICITY"
                        "</div>"
                        '<div class="unsupported">'
                        "— UNSUPPORTED"
                        "</div>"
                        '<div class="confidence">'
                        "Authenticity verification is not available."
                        "</div>"
                        "</div>"
                    )

                else:

                    result_html = (
                        '<div class="result-box">'
                        '<div class="result-label">'
                        "🛡️ AUTHENTICITY"
                        "</div>"
                        '<div class="unsupported">'
                        "Not available"
                        "</div>"
                        "</div>"
                    )

                st.markdown(
                    result_html,
                    unsafe_allow_html=True,
                )

            # ====================================================
            # SERIAL NUMBER
            # ====================================================

            serial_number = "NOT_FOUND"

            if result:
                serial_number = result.get("serial_number", "NOT_FOUND")

            st.markdown(
                '<div class="analysis-title">🔢 Serial Number</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
        <div class="analysis-box">
        <b>{serial_number}</b>
        </div>
        """,
                unsafe_allow_html=True,
            )

            serial_number = result.get("serial_number", "NOT_FOUND")
            
            st.markdown(
        """
        <div style="
            background:#fff8e1;
            border-left:5px solid #f59e0b;
            padding:12px;
            border-radius:8px;
            margin-top:10px;
            margin-bottom:10px;
            color:#172033;
            font-weight:600;
        ">
            ⚠️ Please verify the detected serial number.<br>
            If the OCR result is incorrect, edit it below before saving to the blockchain.
        </div>
        """,
        unsafe_allow_html=True,
        )

            if "corrected_serial" not in st.session_state:
                st.session_state.corrected_serial = serial_number

            corrected_serial = st.text_input(
                "",
                key="corrected_serial",
                label_visibility="collapsed"
            )

            if st.button("💾 Store Scan Record"):
                
                if st.session_state.blockchain_saved:
                    
                    st.warning(
                        "This result has already been saved to the blockchain."
                    )
                    
                else:
                
                    corrected_serial = (
                        st.session_state["corrected_serial"]
                        .strip()
                    )


                    record = {
                        "serial_number": corrected_serial,
                        "denomination": result.get(
                            "denomination",
                            "UNKNOWN",
                        ),
                        "authenticity": result.get(
                            "authenticity",
                            "UNKNOWN",
                        ),
                        "denomination_confidence": round(
                            denomination_confidence,
                            2,
                        ),
                        "authenticity_confidence": round(
                            authenticity_confidence,
                            2,
                        ),
                        "scan_time": datetime.now().strftime(
                            "%d %b %Y, %I:%M %p"
                        ),
                    }

                    add_record(record)
                    st.session_state.blockchain_saved = True
                    
                    st.session_state["saved_serial"] = corrected_serial
                    
                    st.success(
                        f"Saved to blockchain with Serial Number: {corrected_serial}"
                    )

        #     st.markdown(
        # f"""
        # <div style="
        #     background:#172033;
        #     color:white;
        #     padding:12px;
        #     border-radius:10px;
        #     font-weight:700;
        #     text-align:center;
        #     margin-top:10px;
        # ">
        #     ✅ Blockchain Saved Successfully<br>
        #     Serial Number: {corrected_serial}
        # </div>
        # """,
        # unsafe_allow_html=True,
        # )

        # ====================================================
        # SPEECH
        # ====================================================

        if denomination:

            denomination_text = str(denomination)

            if authenticity == "REAL":

                speech_texts = {
                    "en-IN": f"This is a {denomination_text} rupee note. "
                    "The note appears to be real.",
                    "hi-IN": f"यह {denomination_text} रुपये का नोट है। "
                    "यह नोट असली लगता है।",
                    "mr-IN": f"ही {denomination_text} रुपयांची नोट आहे. "
                    "ही नोट खरी असल्याचे दिसते.",
                    "ur-IN": f"یہ {denomination_text} روپے کا نوٹ ہے۔ " 
                    "یہ نوٹ اصلی معلوم ہوتا ہے.",
                    "ne-NP": f"यो {denomination_text} रुपैयाँको नोट हो। " 
                    "यो नोट वास्तविक देखिन्छ.", 
                    "gu-IN": f"આ {denomination_text} રૂપિયાની નોટ છે. " 
                    "આ નોટ અસલી લાગે છે.", 
                    "ta-IN": f"இது {denomination_text} ரூபாய் நோட்டு. " 
                    "இந்த நோட்டு உண்மையானதாக தெரிகிறது.", 
                    "te-IN": f"ఇది {denomination_text} రూపాయల నోటు. " 
                    "ఈ నోటు అసలైనదిగా కనిపిస్తోంది.", 
                    "bn-IN": f"এটি {denomination_text} টাকার নোট। " 
                    "নোটটি আসল বলে মনে হচ্ছে.", 
                    "kn-IN": f"ಇದು {denomination_text} ರೂಪಾಯಿ ನೋಟು. " 
                    "ಈ ನೋಟು ನಿಜವಾದಂತೆ ಕಾಣುತ್ತದೆ.", 
                    "ml-IN": f"ഇത് {denomination_text} രൂപയുടെ നോട്ടാണ്. " 
                    "ഈ നോട്ട് യഥാർത്ഥമാണെന്ന് തോന്നുന്നു.",
                }

            elif authenticity == "FAKE":

                speech_texts = {
                    "en-IN": f"This is a {denomination_text} rupee note. "
                    "The note appears to be fake.",
                    "hi-IN": f"यह {denomination_text} रुपये का नोट है। "
                    "यह नोट नकली लगता है।",
                    "ur-IN": f"یہ {denomination_text} روپے کا نوٹ ہے۔ " 
                    "یہ نوٹ جعلی معلوم ہوتا ہے.", 
                    "ne-NP": f"यो {denomination_text} रुपैयाँको नोट हो। " 
                    "यो नोट नक्कली देखिन्छ.", 
                    "gu-IN": f"આ {denomination_text} રૂપિયાની નોટ છે. " 
                    "આ નોટ નકલી લાગે છે.", 
                    "ta-IN": f"இது {denomination_text} ரூபாய் நோட்டு. " 
                    "இந்த நோட்டு போலியானதாக தெரிகிறது.", 
                    "te-IN": f"ఇది {denomination_text} రూపాయల నోటు. " 
                    "ఈ నోటు నకిలీగా కనిపిస్తోంది.", 
                    "bn-IN": f"এটি {denomination_text} টাকার নোট। " 
                    "নোটটি জাল বলে মনে হচ্ছে.", 
                    "mr-IN": f"ही {denomination_text} रुपयांची नोट आहे। " 
                    "ही नोट बनावट असल्याचे दिसते.", 
                    "kn-IN": f"ಇದು {denomination_text} ರೂಪಾಯಿ ನೋಟು. " 
                    "ಈ ನೋಟು ನಕಲಿ ಎಂದು ಕಾಣುತ್ತದೆ.", 
                    "ml-IN": f"ഇത് {denomination_text} രൂപയുടെ നോട്ടാണ്. " 
                    "ഈ നോട്ട് വ്യാജമാണെന്ന് തോന്നുന്നു.",
                }

            else:

                speech_texts = {
                    "en-IN": f"This is a {denomination_text} rupee note. "
                    "Authenticity verification is not available.",
                    "hi-IN": f"यह {denomination_text} रुपये का नोट है। "
                    "इस नोट की प्रामाणिकता की जांच उपलब्ध नहीं है।",
                    "ne-NP": f"यो {denomination_text} रुपैयाँको नोट हो। " 
                    "यस नोटको प्रमाणीकरण उपलब्ध छैन.", 
                    "ur-IN": f"یہ {denomination_text} روپے کا نوٹ ہے۔ " 
                    "اس نوٹ کی تصدیق دستیاب نہیں ہے.", 
                    "gu-IN": f"આ {denomination_text} રૂપિયાની નોટ છે. " 
                    "નોટની અસલિયતની ચકાસણી ઉપલબ્ધ નથી.", 
                    "ta-IN": f"இது {denomination_text} ரூபாய் நோட்டு. " 
                    "இந்த நோட்டின் நம்பகத்தன்மையை சரிபார்க்க முடியவில்லை.", 
                    "te-IN": f"ఇది {denomination_text} రూపాయల నోటు. " 
                    "ఈ నోటు ప్రామాణికతను ధృవీకరించలేకపోయాము.", 
                    "bn-IN": f"এটি {denomination_text} টাকার নোট। " 
                    "এই নোটের সত্যতা যাচাই করা যায়নি.", 
                    "mr-IN": f"ही {denomination_text} रुपयांची नोट आहे। " 
                    "या नोटेची सत्यता तपासता आली नाही.", 
                    "kn-IN": f"ಇದು {denomination_text} ರೂಪಾಯಿ ನೋಟು. " 
                    "ನೋಟಿನ ಪ್ರಾಮಾಣಿಕತೆ ಪರಿಶೀಲನೆ ಲಭ್ಯವಿಲ್ಲ.", 
                    "ml-IN": f"ഇത് {denomination_text} രൂപയുടെ നോട്ടാണ്. " 
                    "ഈ നോട്ടിന്റെ ആധികാരികത പരിശോധിക്കാൻ കഴിഞ്ഞില്ല.",
                }

            speech_text = speech_texts.get(
                language_code,
                speech_texts["en-IN"],
            )

            speech_text_json = json.dumps(
                speech_text,
                ensure_ascii=False,
            )

            language_json = json.dumps(language_code)

            if speak_clicked:

                speech_html = f"""
                <script>

                function speakCurrency() {{

                    if (!("speechSynthesis" in window)) {{

                        alert(
                            "Speech synthesis is not supported by this browser."
                        );

                        return;
                    }}

                    window.speechSynthesis.cancel();

                    const text = {speech_text_json};

                    const language = {language_json};

                    const utterance =
                        new SpeechSynthesisUtterance(text);

                    utterance.lang = language;

                    utterance.rate = 0.85;

                    utterance.pitch = 1.0;

                    window.speechSynthesis.speak(
                        utterance
                    );
                }}

                speakCurrency();

                </script>
                """

                st.components.v1.html(
                    speech_html,
                    height=0,
                )

# ============================================================
# HISTORY / VERIFY / VIEW BUTTONS
# ============================================================

if st.session_state.page == "main":

    st.markdown("""
    <style>

    .history-card{
        background:white;
        color:#172033;
        padding:12px;
        width:98%;
        margin:auto;
        margin-bottom:12px;
        border-radius:12px;
        border:1px solid #dfe3e8;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
    }

    .history-summary{
        background:white;
        color:#172033;
        padding:14px;
        width:66%;
        margin:auto;
        margin-bottom:15px;
        border-radius:12px;
        border:1px solid #dfe3e8;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
    }

    </style>
    """, unsafe_allow_html=True)

    if (
        st.session_state.result is not None
        and st.session_state.result.get("success", False)
    ):

        left_space, middle, right_space = st.columns([1, 3, 1])

        with middle:

            col1, col2, col3 = st.columns(3)

            with col1:
              if st.button("📜 Note History"):

                st.session_state["history_serial"] = (
                    st.session_state.get(
                        "saved_serial",
                        st.session_state.get(
                            "corrected_serial",
                            ""
                        )
                    ).strip()
                )

                st.session_state.page = "history"
                st.rerun()
                
            with col2: 

              if st.button("🔐 Verify Blockchain"):

                st.session_state.page = "verify"
                st.rerun()
                
            with col3:         

              if st.button("🧱 View Blockchain"):

                st.session_state.page = "view"
                st.rerun()

# ============================================================
# VERIFY PAGE
# ============================================================

if st.session_state.page == "verify":

    if st.button("← Back", key="verify_back"):
        st.session_state.page = "main"
        st.rerun()

    st.markdown("""
        <div style="
        color: #172033;
        font-size:28px;
        font-weight:800;
        text-align:center;
        margin-bottom:20px;
        ">
        🔗 Blockchain Verification
        </div>
        """, unsafe_allow_html=True)

    verification_result = verify_chain()

    # ============================================================
    # HASH INTEGRITY CHECK
    # ============================================================

    if verification_result["hash_check"] is True:

        st.markdown("""
        <div style="
        background:#dcfce7;
        color:#166534;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ✅ Hash Integrity Check
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:#fee2e2;
        color:#991b1b;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ❌ Hash Integrity Check
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # PREVIOUS HASH LINK CHECK
    # ============================================================

    if verification_result["link_check"] is True:

        st.markdown("""
        <div style="
        background:#dcfce7;
        color:#166534;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ✅ Previous Hash Link Check
        </div>
        """, unsafe_allow_html=True)

    elif verification_result["link_check"] is False:

        st.markdown("""
        <div style="
        background:#fee2e2;
        color:#991b1b;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ❌ Previous Hash Link Check
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:#f3f4f6;
        color:#667085;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ⏸ Previous Hash Link Check
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # RSA SIGNATURE CHECK
    # ============================================================

    if verification_result["rsa_check"] is True:

        st.markdown("""
        <div style="
        background:#dcfce7;
        color:#166534;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ✅ RSA Signature Verification
        </div>
        """, unsafe_allow_html=True)

    elif verification_result["rsa_check"] is False:

        st.markdown("""
        <div style="
        background:#fee2e2;
        color:#991b1b;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ❌ RSA Signature Verification
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:#f3f4f6;
        color:#667085;
        padding:12px;
        border-radius:10px;
        font-size:17px;
        font-weight:700;
        margin-bottom:10px;
        ">
        ⏸ RSA Signature Verification
        </div>
        """, unsafe_allow_html=True)

    # ============================================================
    # DETECTED ISSUES
    # ============================================================

    hash_issues = verification_result.get("hash_issues", [])
    link_issues = verification_result.get("link_issues", [])
    rsa_issues = verification_result.get("rsa_issues", [])

    if (
        len(hash_issues) > 0
        or len(link_issues) > 0
        or len(rsa_issues) > 0
    ):

        st.markdown("""
        <div style="
        background:#fff8e1;
        color:#172033;
        padding:14px;
        border-left:5px solid #f59e0b;
        border-radius:8px;
        font-size:18px;
        font-weight:800;
        margin-top:15px;
        margin-bottom:12px;
        ">
        ⚠️ Detected Issues
        </div>
        """, unsafe_allow_html=True)

        if len(hash_issues) > 0:

            st.markdown("""
            <div style="
            background:#fee2e2;
            color:#991b1b;
            padding:12px;
            border-radius:10px;
            margin-bottom:10px;
            font-size:15px;
            ">
            <b>Hash Integrity Issues:</b>
            </div>
            """, unsafe_allow_html=True)

            for block_number in hash_issues:

                st.markdown(
                    f"""
                    <div style="
                    background:white;
                    color:#172033;
                    padding:9px 12px;
                    border-radius:8px;
                    border:1px solid #fecaca;
                    margin-bottom:6px;
                    ">
                    • Block #{block_number} — Hash mismatch
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if len(link_issues) > 0:

            st.markdown("""
            <div style="
            background:#fee2e2;
            color:#991b1b;
            padding:12px;
            border-radius:10px;
            margin-top:12px;
            margin-bottom:10px;
            font-size:15px;
            ">
            <b>Previous Hash Link Issues:</b>
            </div>
            """, unsafe_allow_html=True)

            for block_number in link_issues:

                st.markdown(
                    f"""
                    <div style="
                    background:white;
                    color:#172033;
                    padding:9px 12px;
                    border-radius:8px;
                    border:1px solid #fecaca;
                    margin-bottom:6px;
                    ">
                    • Block #{block_number} — Previous hash mismatch
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if len(rsa_issues) > 0:

            st.markdown("""
            <div style="
            background:#fee2e2;
            color:#991b1b;
            padding:12px;
            border-radius:10px;
            margin-top:12px;
            margin-bottom:10px;
            font-size:15px;
            ">
            <b>RSA Signature Issues:</b>
            </div>
            """, unsafe_allow_html=True)

            for block_number in rsa_issues:

                st.markdown(
                    f"""
                    <div style="
                    background:white;
                    color:#172033;
                    padding:9px 12px;
                    border-radius:8px;
                    border:1px solid #fecaca;
                    margin-bottom:6px;
                    ">
                    • Block #{block_number} — Invalid RSA signature
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ============================================================
    # OVERALL STATUS
    # ============================================================

    if verification_result["valid"]:

        st.markdown("""
        <div style="
        background:#172033;
        color:white;
        padding:16px;
        border-radius:12px;
        font-size:22px;
        font-weight:800;
        text-align:center;
        margin-top:15px;
        margin-bottom:15px;
        box-shadow:0 4px 12px rgba(0,0,0,0.15);
        ">
        ✅ Overall Status: Blockchain Valid
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div style="
        background:#b91c1c;
        color:white;
        padding:16px;
        border-radius:12px;
        font-size:22px;
        font-weight:800;
        text-align:center;
        margin-top:15px;
        margin-bottom:15px;
        box-shadow:0 4px 12px rgba(0,0,0,0.15);
        ">
        ❌ Overall Status: Tampering Detected
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div style="
            background:#fff8e1;
            color:#172033;
            padding:14px;
            border-left:5px solid #f59e0b;
            border-radius:8px;
            font-size:16px;
            font-weight:700;
            margin-bottom:15px;
            ">
            <b>Reason:</b>
            {verification_result["reason"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# VIEW BLOCKCHAIN PAGE
# ============================================================

if st.session_state.page == "view":

    if st.button("← Back", key="view_back"):
        st.session_state.page = "main"
        st.rerun()

    import json

    with open("blockchain_log.json", "r") as f:
        chain = json.load(f)

    st.markdown(
    f"""
    <div style="
        color:#172033;
        font-size:22px;
        font-weight:800;
        margin-bottom:15px;
    ">
        Total Blocks: {len(chain)}
    </div>
    """,
    unsafe_allow_html=True,
    )

    for i, block in enumerate(chain):

        data = block.get("data", {})

        serial_value = data.get(
            "serial_number",
            "UNKNOWN"
        )

        denomination_value = data.get(
            "denomination",
            "UNKNOWN"
        )

        authenticity_value = data.get(
            "authenticity",
            "UNKNOWN"
        )

        block_hash = block.get(
            "hash",
            "N/A"
        )

        st.markdown(
            f"""
            <div style="
            background:white;
            color:#172033;
            padding:10px;
            border-radius:10px;
            border:1px solid #dfe3e8;
            box-shadow:0 3px 8px rgba(0,0,0,0.08);
            margin-bottom:10px;
            font-size:14px;
            ">

            <div style="
            font-size:18px;
            font-weight:800;
            color:#1d4ed8;
            margin-bottom:6px;
            ">
            🔗 Block #{i+1}
            </div>

            <b>Serial Number:</b> {serial_value}<br>
            <b>Denomination:</b> ₹{denomination_value}<br>
            <b>Authenticity:</b> {authenticity_value}<br>

            <b>Hash:</b><br>
            <span style="
            word-break:break-all;
            color:#667085;
            font-size:12px;
            ">
            {block_hash[:40]}...
            </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# HISTORY PAGE
# ============================================================

if st.session_state.page == "history":
    
    st.components.v1.html(
        """
        <script>
            window.parent.scrollTo(0, 0);
        </script>
        """,
        height=0,
    )


    if st.button("← Back", key="history_back"):
        st.session_state.page = "main"
        st.rerun()

    search_serial = st.session_state.get(
        "history_serial",
        ""
    )

    import json

    try:

        with open(
            "blockchain_log.json", 
            "r"
        ) as f:

            chain = json.load(f)

            matches = []

            for block in chain:

                stored_serial = (
                    block["data"]["serial_number"]
                    .strip()
                    .upper()
                    .replace(" ", "")
                )

                entered_serial = (
                    search_serial
                    .strip()
                    .upper()
                    .replace(" ", "")
                )

                if stored_serial == entered_serial:
                    matches.append(block)

            if len(matches) == 0:

                st.error(
                    "No record found."
                )

            else:

                st.markdown(
                    f"""
                    <div style="
                    background:#dcfce7;
                    color:#166534;
                    padding:12px;
                    border-radius:10px;
                    font-size:18px;
                    font-weight:700;
                    text-align:center;
                    margin-bottom:15px;
                    ">
                    Found {len(matches)} scans
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="
                    background:white;
                    color:#172033;
                    padding:14px;
                    width:66%;
                    margin:auto;
                    margin-bottom:15px;
                    border-radius:12px;
                    border:1px solid #dfe3e8;
                    box-shadow:0 4px 12px rgba(0,0,0,0.08);
                    ">
                    <div style="
                    background:#172033;
                    color:white;
                    padding:12px;
                    border-radius:10px;
                    width:60%;
                    margin:auto;
                    text-align:center;
                    font-size:24px;
                    font-weight:800;
                    margin-bottom:15px;
                    ">
                    🔍 Note History Summary
                    </div>

                    <b>Serial Number:</b> {search_serial}<br>

                    <b>Times Checked:</b> {len(matches)}<br>

                    <b>First Scan:</b> {matches[0]["data"].get("scan_time","N/A")}<br>

                    <b>Last Scan:</b> {matches[-1]["data"].get("scan_time","N/A")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("---")

                cols = st.columns(2)

                for i, block in enumerate(matches):

                    data = block["data"]

                    with cols[i % 2]:

                        st.markdown(
                            f"""
                            <div style="
                            background:white;
                            color:#172033;
                            padding:10px;
                            border-radius:12px;
                            border:1px solid #dfe3e8;
                            box-shadow:0 4px 12px rgba(0,0,0,0.08);
                            margin-bottom:12px;
                            ">

                            <div style="
                            font-size:18px;
                            font-weight:800;
                            color:#1d4ed8;
                            margin-bottom:6px;
                            ">
                            📄 Scan #{i+1}
                            </div>

                            <b>Result:</b> {data.get("authenticity","UNKNOWN")}<br>

                            <b>Denomination:</b> ₹{data.get("denomination","UNKNOWN")}<br>

                            <b>Denomination Confidence:</b>
                            {data.get("denomination_confidence",0)}%<br>

                            <b>Authenticity Confidence:</b>
                            {data.get("authenticity_confidence",0)}%<br>

                            <b>Scan Time:</b>
                            {data.get("scan_time","N/A")}<br>

                            <b>Hash:</b>
                            {block.get("hash","N/A")[:25]}...

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.divider()

    except Exception as e:

        st.error(
            f"History Error: {e}"
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="app-footer">'
    "Indian Currency Detection System • YOLO11 + MobileNetV2"
    "</div>",
    unsafe_allow_html=True,
)
