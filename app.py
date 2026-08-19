from pathlib import Path
import sys
import base64
import html
import tempfile
import json

import streamlit as st
from PIL import Image

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

if "result" not in st.session_state:
    st.session_state.result = None

if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None


# ============================================================
# BACKGROUND IMAGE
# ============================================================

BACKGROUND_IMAGE = PROJECT_ROOT / "assets" / "image.png"

if BACKGROUND_IMAGE.exists():

    try:
        with open(BACKGROUND_IMAGE, "rb") as f:
            image_base64 = base64.b64encode(
                f.read()
            ).decode("utf-8")

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
    font-size: 15px;
    font-weight: 800;
    margin-bottom: 5px;
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


/* ==========================================================
   DETECT CURRENCY BUTTON
   ========================================================== */

div.stButton > button[kind="primary"] {{
    background: #172033 !important;
    color: white !important;
    border: 1px solid #172033 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}}

div.stButton > button[kind="primary"]:hover {{
    background: #1d4ed8 !important;
    color: white !important;
    border: 1px solid #1d4ed8 !important;
}}


/* ==========================================================
   FILE UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {{
    background: rgba(255,255,255,0.95);
    border-radius: 12px;
    border: 1px solid #dfe3e8;
    padding: 5px;
}}

[data-testid="stFileUploader"] label {{
    color: #172033 !important;
}}


/* ==========================================================
   IMAGE
   ========================================================== */

[data-testid="stImage"] {{
    border-radius: 14px;
    overflow: hidden;
}}


/* ==========================================================
   RESULT HEADING
   ========================================================== */

.result-heading {{
    color: #172033;
    font-size: 24px;
    font-weight: 850;
    margin-top: 18px;
    margin-bottom: 12px;
}}


/* ==========================================================
   RESULT BOX
   ========================================================== */

.result-box {{
    background: rgba(255,255,255,0.97);
    border: 1px solid #e0e4e8;
    border-radius: 15px;
    padding: 17px 19px;
    min-height: 108px;
    box-shadow: 0 7px 22px rgba(0,0,0,0.08);
}}

.result-label {{
    color: #344054;
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 1px;
    margin-bottom: 7px;
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
    color: #667085;
    font-size: 12px;
    margin-top: 8px;
}}


/* ==========================================================
   ANALYSIS
   ========================================================== */

.analysis-title {{
    color: #172033;
    font-size: 16px;
    font-weight: 800;
    margin-top: 14px;
    margin-bottom: 6px;
}}

.analysis-box {{
    background: rgba(255,255,255,0.96);
    border: 1px solid #e1e5eb;
    border-radius: 11px;
    padding: 11px 14px;
    color: #344054;
    font-size: 13px;
    box-shadow: 0 5px 16px rgba(0,0,0,0.06);
}}


/* ==========================================================
   SPEECH
   ========================================================== */

.speech-title {{
    color: #172033 !important;
    font-size: 16px;
    font-weight: 800;
    margin-top: 14px;
    margin-bottom: 6px;
}}


/* ==========================================================
   FOOTER
   ========================================================== */

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

st.markdown(
    '<div class="app-title">Indian Currency Detector</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SPEECH LANGUAGE
# ============================================================

language_options = {
    "English": "en-IN",
    "हिन्दी": "hi-IN",
    "தமிழ்": "ta-IN",
    "తెలుగు": "te-IN",
    "বাংলা": "bn-IN",
    "मराठी": "mr-IN",
    "മലയാളം": "ml-IN",
}

selected_language = st.selectbox(
    "🔊 Speech Language",
    options=list(language_options.keys()),
)

language_code = language_options[selected_language]


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

if uploaded_file is not None:

    if (
        st.session_state.uploaded_name
        != uploaded_file.name
    ):

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.session_state.selected_image = image
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.result = None

        except Exception as error:

            st.error(
                f"Unable to open image: {error}"
            )

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
            "Detect Currency",
            type="primary",
            use_container_width=True,
        )

        if detect_button:

            temporary_directory = (
                PROJECT_ROOT / "streamlit_temp"
            )

            temporary_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            temporary_file = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                dir=temporary_directory,
                delete=False,
            )

            temporary_path = Path(
                temporary_file.name
            )

            temporary_file.close()

            try:

                image.save(
                    temporary_path,
                    format="JPEG",
                    quality=95,
                )

                with st.spinner(
                    "Analysing currency..."
                ):

                    result = detect_currency(
                        temporary_path
                    )

                st.session_state.result = result

            except Exception as error:

                st.session_state.result = {
                    "success": False,
                    "message": (
                        f"Detection error: {error}"
                    ),
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


if result is not None:

    if not result.get("success", False):

        st.error(
            result.get(
                "message",
                "Currency could not be detected.",
            )
        )

    else:

        denomination = result.get(
            "denomination"
        )

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

                safe_denomination = html.escape(
                    str(denomination)
                )

                result_html = (
                    '<div class="result-box">'
                    '<div class="result-label">'
                    '💵 DENOMINATION'
                    '</div>'
                    '<div class="denomination">'
                    f'₹{safe_denomination}'
                    '</div>'
                    '<div class="confidence">'
                    'Confidence: '
                    f'<b>{denomination_confidence:.2f}%</b>'
                    '</div>'
                    '</div>'
                )

            else:

                result_html = (
                    '<div class="result-box">'
                    '<div class="result-label">'
                    '💵 DENOMINATION'
                    '</div>'
                    '<div class="unsupported">'
                    'Not detected'
                    '</div>'
                    '</div>'
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
                    '🛡 AUTHENTICITY'
                    '</div>'
                    '<div class="real">'
                    '✓ REAL'
                    '</div>'
                    '<div class="confidence">'
                    'Confidence: '
                    f'<b>{authenticity_confidence:.2f}%</b>'
                    '</div>'
                    '</div>'
                )

            elif authenticity == "FAKE":

                result_html = (
                    '<div class="result-box">'
                    '<div class="result-label">'
                    '🛡 AUTHENTICITY'
                    '</div>'
                    '<div class="fake">'
                    '! FAKE'
                    '</div>'
                    '<div class="confidence">'
                    'Confidence: '
                    f'<b>{authenticity_confidence:.2f}%</b>'
                    '</div>'
                    '</div>'
                )

            elif authenticity == "UNSUPPORTED":

                result_html = (
                    '<div class="result-box">'
                    '<div class="result-label">'
                    '🛡 AUTHENTICITY'
                    '</div>'
                    '<div class="unsupported">'
                    '— UNSUPPORTED'
                    '</div>'
                    '<div class="confidence">'
                    'Authenticity verification is not available.'
                    '</div>'
                    '</div>'
                )

            else:

                result_html = (
                    '<div class="result-box">'
                    '<div class="result-label">'
                    '🛡 AUTHENTICITY'
                    '</div>'
                    '<div class="unsupported">'
                    'Not available'
                    '</div>'
                    '</div>'
                )

            st.markdown(
                result_html,
                unsafe_allow_html=True,
            )


        # ====================================================
        # ANALYSIS MESSAGE
        # ====================================================

        message = result.get(
            "message",
            "Analysis completed.",
        )

        safe_message = html.escape(
            str(message)
        )

        st.markdown(
            '<div class="analysis-title">📋 Analysis</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="analysis-box">'
            f'{safe_message}'
            '</div>',
            unsafe_allow_html=True,
        )


        # ====================================================
        # SPEECH
        # ====================================================

        if denomination:

            denomination_text = str(
                denomination
            )


            if authenticity == "REAL":

                speech_texts = {

                    "en-IN":
                        f"This is a {denomination_text} rupee note. "
                        "The note appears to be real.",

                    "hi-IN":
                        f"यह {denomination_text} रुपये का नोट है। "
                        "यह नोट असली लगता है।",

                    "ta-IN":
                        f"இது {denomination_text} ரூபாய் நோட்டு. "
                        "இந்த நோட்டு உண்மையானதாக தெரிகிறது.",

                    "te-IN":
                        f"ఇది {denomination_text} రూపాయల నోటు. "
                        "ఈ నోటు అసలైనదిగా కనిపిస్తోంది.",

                    "bn-IN":
                        f"এটি {denomination_text} টাকার নোট। "
                        "নোটটি আসল বলে মনে হচ্ছে।",

                    "mr-IN":
                        f"ही {denomination_text} रुपयांची नोट आहे. "
                        "ही नोट खरी असल्याचे दिसते.",

                    "ml-IN":
                        f"ഇത് {denomination_text} രൂപയുടെ നോട്ടാണ്. "
                        "ഈ നോട്ട് യഥാർത്ഥമാണെന്ന് തോന്നുന്നു.",
                }


            elif authenticity == "FAKE":

                speech_texts = {

                    "en-IN":
                        f"This is a {denomination_text} rupee note. "
                        "The note appears to be fake.",

                    "hi-IN":
                        f"यह {denomination_text} रुपये का नोट है। "
                        "यह नोट नकली लगता है।",

                    "ta-IN":
                        f"இது {denomination_text} ரூபாய் நோட்டு. "
                        "இந்த நோட்டு போலியானதாக தெரிகிறது.",

                    "te-IN":
                        f"ఇది {denomination_text} రూపాయల నోటు. "
                        "ఈ నోటు నకిలీగా కనిపిస్తోంది.",

                    "bn-IN":
                        f"এটি {denomination_text} টাকার নোট। "
                        "নোটটি জাল বলে মনে হচ্ছে।",

                    "mr-IN":
                        f"ही {denomination_text} रुपयांची नोट आहे. "
                        "ही नोट बनावट असल्याचे दिसते.",

                    "ml-IN":
                        f"ഇത് {denomination_text} രൂപയുടെ നോട്ടാണ്. "
                        "ഈ നോട്ട് വ്യാജമാണെന്ന് തോന്നുന്നു.",
                }


            else:

                speech_texts = {

                    "en-IN":
                        f"This is a {denomination_text} rupee note. "
                        "Authenticity verification is not available.",

                    "hi-IN":
                        f"यह {denomination_text} रुपये का नोट है। "
                        "इस नोट की प्रामाणिकता की जांच उपलब्ध नहीं है।",

                    "ta-IN":
                        f"இது {denomination_text} ரூபாய் நோட்டு. "
                        "இந்த நோட்டின் நம்பகத்தன்மையை சரிபார்க்க முடியவில்லை.",

                    "te-IN":
                        f"ఇది {denomination_text} రూపాయల నోటు. "
                        "ఈ నోటు ప్రామాణికతను ధృవీకరించలేకపోయాము.",

                    "bn-IN":
                        f"এটি {denomination_text} টাকার নোট। "
                        "এই নোটের সত্যতা যাচাই করা যায়নি।",

                    "mr-IN":
                        f"ही {denomination_text} रुपयांची नोट आहे. "
                        "या नोटेची सत्यता तपासता आली नाही.",

                    "ml-IN":
                        f"ഇത് {denomination_text} രൂപയുടെ നോട്ടാണ്. "
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

            language_json = json.dumps(
                language_code
            )


            st.markdown(
                '<div class="speech-title">'
                '🔊 Speak Result'
                '</div>',
                unsafe_allow_html=True,
            )


            speech_html = f"""
<button
    onclick="speakCurrency()"
    style="
        width:100%;
        padding:11px 14px;
        border:none;
        border-radius:10px;
        background:#172033;
        color:white;
        font-size:14px;
        font-weight:700;
        cursor:pointer;
    "
>
    🔊 Speak Result
</button>

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

</script>
"""

            st.components.v1.html(
                speech_html,
                height=55,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="app-footer">'
    'Indian Currency Detection System • YOLO11 + MobileNetV2'
    '</div>',
    unsafe_allow_html=True,
)