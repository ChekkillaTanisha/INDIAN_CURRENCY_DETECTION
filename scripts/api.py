from pathlib import Path
import sys
import tempfile

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow Python to find app_inference.py
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from app_inference import detect_currency


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

MAX_FILE_SIZE = 10 * 1024 * 1024

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ============================================================
# HELPER
# ============================================================

def allowed_file(filename):
    return (
        Path(filename).suffix.lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "Indian Currency API is running.",
    })


# ============================================================
# CURRENCY DETECTION
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "message": "No image was uploaded.",
        }), 400


    uploaded_file = request.files["image"]


    if uploaded_file.filename == "":

        return jsonify({
            "success": False,
            "message": "No image was selected.",
        }), 400


    if not allowed_file(uploaded_file.filename):

        return jsonify({
            "success": False,
            "message": (
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, WEBP, or BMP."
            ),
        }), 400


    # --------------------------------------------------------
    # Save uploaded image temporarily
    # --------------------------------------------------------

    suffix = Path(
        secure_filename(
            uploaded_file.filename
        )
    ).suffix.lower()


    temporary_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    )


    temporary_path = Path(
        temporary_file.name
    )


    temporary_file.close()


    try:

        uploaded_file.save(
            temporary_path
        )


        # ----------------------------------------------------
        # RUN EXISTING YOLO + MOBILENET PIPELINE
        # ----------------------------------------------------

        result = detect_currency(
            temporary_path
        )


        # ----------------------------------------------------
        # Convert result into JSON-safe response
        # ----------------------------------------------------

        response = {
            "success": bool(
                result.get("success", False)
            ),

            "message": result.get(
                "message",
                "",
            ),

            "denomination": result.get(
                "denomination"
            ),

            "denomination_confidence": round(
                float(
                    result.get(
                        "denomination_confidence",
                        0.0,
                    )
                ) * 100,
                2,
            ),

            "authenticity": result.get(
                "authenticity"
            ),

            "authenticity_confidence": round(
                float(
                    result.get(
                        "authenticity_confidence",
                        0.0,
                    )
                ) * 100,
                2,
            ),

            "box": result.get(
                "box"
            ),
        }


        return jsonify(response)


    except Exception as error:

        print()
        print("=" * 70)
        print("DETECTION ERROR")
        print("=" * 70)
        print(error)
        print("=" * 70)
        print()


        return jsonify({
            "success": False,
            "message": (
                "An error occurred while "
                "processing the image."
            ),
            "error": str(error),
        }), 500


    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded image
        # ----------------------------------------------------

        try:

            if temporary_path.exists():

                temporary_path.unlink()

        except Exception:

            pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("INDIAN CURRENCY DETECTION API")
    print("=" * 70)
    print()
    print("Backend:")
    print("http://127.0.0.1:5000")
    print()
    print("Health check:")
    print("http://127.0.0.1:5000/api/health")
    print()
    print("Detection endpoint:")
    print("POST /api/detect")
    print()
    print("YOLO11 + MobileNetV2 loaded through app_inference.py")
    print()
    print("=" * 70)
    print()


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
