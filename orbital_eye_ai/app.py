
from pathlib import Path
from datetime import datetime, timezone
import sqlite3
import uuid

from flask import (
    Flask, render_template, request,
    redirect, url_for, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from PIL import Image, ImageStat, UnidentifiedImageError


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "orbital_eye.db"

TEMPLATE_DIR = BASE_DIR / "app" / "templates"
STATIC_DIR = BASE_DIR / "app" / "static"

ALLOWED_EXTENSIONS = {
    "png", "jpg", "jpeg", "tif", "tiff", "bmp"
}

MAX_BYTES = 20 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)


# ==========================================
# 2. FLASK INITIALIZATION
# ==========================================

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static"
)

app.secret_key = "orbital-eye-development-key"

app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)


# Diagnostic information
print("TEMPLATE DIRECTORY:", TEMPLATE_DIR)
print("TEMPLATE EXISTS:", TEMPLATE_DIR.exists())
print("HTML FILES:", list(TEMPLATE_DIR.glob("*.html")))


# ==========================================
# 3. DATABASE
# ==========================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                width INTEGER,
                height INTEGER,
                mean_intensity REAL,
                status TEXT NOT NULL
            )
        """)


# ==========================================
# 4. IMAGE VALIDATION
# ==========================================

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def inspect_image(path):
    """
    Basic image quality inspection.
    This is NOT a medical or ophthalmic diagnosis.
    """

    with Image.open(path) as image:
        image.load()

        gray = image.convert("L")
        stats = ImageStat.Stat(gray)

        return (
            image.width,
            image.height,
            round(stats.mean[0], 2)
        )


# ==========================================
# 5. MISSION OVERVIEW
# ==========================================

@app.route("/")
def index():

    with db() as conn:

        recent = conn.execute(
            """
            SELECT * FROM scans
            ORDER BY created_at DESC
            LIMIT 6
            """
        ).fetchall()

        total = conn.execute(
            "SELECT COUNT(*) AS n FROM scans"
        ).fetchone()["n"]

    return render_template(
        "index.html",
        recent=recent,
        total=total
    )


# ==========================================
# 6. OCT IMAGE ANALYSIS
# ==========================================

@app.route("/analysis", methods=["GET", "POST"])
def analysis():

    result = None

    if request.method == "POST":

        file = request.files.get("oct_image")

        if not file or not file.filename:
            flash("Please select an OCT image.", "error")
            return redirect(url_for("analysis"))

        if not allowed_file(file.filename):
            flash("Unsupported image format.", "error")
            return redirect(url_for("analysis"))

        original = secure_filename(file.filename)

        scan_id = uuid.uuid4().hex[:12]

        stored_name = f"{scan_id}_{original}"

        destination = UPLOAD_DIR / stored_name

        file.save(destination)

        try:

            width, height, mean_intensity = inspect_image(
                destination
            )

        except (UnidentifiedImageError, OSError):

            destination.unlink(missing_ok=True)

            flash("The uploaded file is not a valid image.", "error")

            return redirect(url_for("analysis"))

        created_at = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")

        # FUTURE AI MODEL INTEGRATION
        # U-Net retinal segmentation will be connected here.

        with db() as conn:

            conn.execute(
                """
                INSERT INTO scans (
                    id, filename, stored_name,
                    created_at, width, height,
                    mean_intensity, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    original,
                    stored_name,
                    created_at,
                    width,
                    height,
                    mean_intensity,
                    "Awaiting AI model"
                )
            )

        result = {
            "id": scan_id,
            "filename": original,
            "stored_name": stored_name,
            "width": width,
            "height": height,
            "mean_intensity": mean_intensity,
            "created_at": created_at
        }

    return render_template(
        "analysis.html",
        result=result
    )


# ==========================================
# 7. SCAN HISTORY
# ==========================================

@app.route("/history")
def history():

    with db() as conn:

        scans = conn.execute(
            """
            SELECT * FROM scans
            ORDER BY created_at DESC
            """
        ).fetchall()

    return render_template(
        "history.html",
        scans=scans
    )


# ==========================================
# 8. UPLOADED IMAGE ACCESS
# ==========================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# ==========================================
# 9. ERROR HANDLING
# ==========================================

@app.errorhandler(413)
def too_large(error):

    flash(
        "File too large. Maximum size is 20 MB.",
        "error"
    )

    return redirect(url_for("analysis"))


# ==========================================
# 10. APPLICATION START
# ==========================================

if __name__ == "__main__":

    init_db()

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
