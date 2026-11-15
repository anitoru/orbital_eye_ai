"""ORBITAL EYE AI - Flask web application (routes).

Run:   python app.py            or   flask run
Demo:  flask seed-demo          (fictional astronauts + synthetic OCT scans)

Business logic lives in the `orbital/` package:
  orbital/config.py      settings from environment variables
  orbital/db.py          SQLite schema and helpers
  orbital/imaging.py     validation, preprocessing, technical quality report
  orbital/boundaries.py  boundary detection + pixel thickness (heuristic fallback)
  orbital/pipeline.py    upload -> quality -> preprocess -> segment -> measure
  orbital/monitoring.py  baseline comparison, timelines, alerts
  orbital/ml/            U-Net, inference, evaluation metrics
  orbital/demo.py        synthetic demonstration data

AI disclosure: this codebase was written with the assistance of Claude (Anthropic), an AI coding
assistant, directed and reviewed by the team. See "Use of AI tools" in the repository README.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional

import click
from flask import (
    Flask, abort, current_app, flash, g, redirect, render_template,
    request, send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

from orbital import db, demo, imaging, monitoring, pipeline
from orbital.config import BASE_DIR, Config
from orbital.ml import download, inference

PHASES = ["baseline", "in-flight", "post-flight"]
EYES = ["OD", "OS"]
MISSION_STATUSES = ["pre-flight", "in-flight", "post-flight", "completed"]
ASTRONAUT_ID_RE = re.compile(r"^[A-Z0-9][A-Z0-9-]{2,19}$")

log = logging.getLogger("orbital_eye")


def create_app(overrides: Optional[Dict[str, Any]] = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "app" / "templates"),
        static_folder=str(BASE_DIR / "app" / "static"),
        static_url_path="/static",
    )
    settings = {k: getattr(Config, k) for k in dir(Config) if k.isupper()}
    settings.update(overrides or {})
    app.config.update(settings)
    cfg = SimpleNamespace(**settings)
    app.extensions["orbital_cfg"] = cfg

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.UPLOAD_DIR / cfg.DERIVED_DIR_NAME).mkdir(parents=True, exist_ok=True)
    db.init_db(cfg.DB_PATH)  # runs for python app.py, flask run and gunicorn alike
    if cfg.MODEL_URL:
        download.ensure_model(cfg.MODEL_PATH, cfg.MODEL_URL)
    if cfg.PUBLIC_DEMO and not cfg.TESTING:
        conn = db.connect(cfg.DB_PATH)
        try:
            if db.query_one(conn, "SELECT COUNT(*) n FROM astronauts")["n"] == 0:
                log.info("Public demo: seeding demo astronauts %s", demo.seed(conn, cfg))
        finally:
            conn.close()

    register_routes(app)
    register_cli(app)
    register_filters(app)
    return app


def get_cfg() -> SimpleNamespace:
    return current_app.extensions["orbital_cfg"]


def get_db():
    if "db" not in g:
        g.db = db.connect(get_cfg().DB_PATH)
    return g.db


def parse_date(value: str) -> Optional[str]:
    try:
        return date.fromisoformat(value).isoformat()
    except (TypeError, ValueError):
        return None


def register_filters(app: Flask) -> None:
    @app.template_filter("json_list")
    def json_list(value):
        try:
            return json.loads(value) if value else []
        except ValueError:
            return []

    @app.template_filter("filesize")
    def filesize(n):
        if not n:
            return "-"
        return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"

    @app.template_filter("signed")
    def signed(v):
        return "-" if v is None else f"{v:+.1f}"

    @app.context_processor
    def inject_globals():
        status = inference.model_status(get_cfg().MODEL_PATH)
        return {"model_label": status["label"], "model_state": status["state"],
                "alert_threshold": get_cfg().ALERT_THRESHOLD_PCT, "public_demo": get_cfg().PUBLIC_DEMO}


def cleanup_demo_files(cfg) -> None:
    """Delete visitors' temporary analysis images older than DEMO_UPLOAD_TTL_MIN (public demo)."""
    cutoff = datetime.now(timezone.utc).timestamp() - cfg.DEMO_UPLOAD_TTL_MIN * 60
    for path in (cfg.UPLOAD_DIR / cfg.DERIVED_DIR_NAME).glob("tmp-*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def demo_disabled_redirect(endpoint: str = "index", **kwargs):
    flash("This action is disabled in the public demo. Run the app yourself to use it (see the README).", "error")
    return redirect(url_for(endpoint, **kwargs))


def register_routes(app: Flask) -> None:
    @app.teardown_appcontext
    def close_db(_exc):
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    # ---------------- Mission overview ----------------
    @app.route("/")
    def index():
        conn = get_db()
        stats = {
            "astronauts": db.query_one(conn, "SELECT COUNT(*) n FROM astronauts")["n"],
            "scans": db.query_one(conn, "SELECT COUNT(*) n FROM scans")["n"],
            "analysed": db.query_one(conn, "SELECT COUNT(*) n FROM scans WHERE thickness_mean_px IS NOT NULL")["n"],
            "alerts": db.query_one(conn, "SELECT COUNT(*) n FROM alerts")["n"],
        }
        recent = db.query(conn, """SELECT s.*, a.display_name FROM scans s
                                   LEFT JOIN astronauts a ON a.id = s.astronaut_id
                                   ORDER BY s.created_at DESC, s.exam_date DESC LIMIT 6""")
        alerts = db.query(conn, """SELECT al.*, a.display_name FROM alerts al
                                   JOIN astronauts a ON a.id = al.astronaut_id
                                   ORDER BY al.exam_date DESC LIMIT 5""")
        crew = db.query(conn, """SELECT a.*, COUNT(s.id) n_scans, MAX(s.exam_date) last_exam,
                                 (SELECT COUNT(*) FROM alerts al WHERE al.astronaut_id = a.id) n_alerts
                                 FROM astronauts a LEFT JOIN scans s ON s.astronaut_id = a.id
                                 GROUP BY a.id ORDER BY a.id""")
        return render_template("index.html", stats=stats, recent=recent, alerts=alerts, crew=crew)

    # ---------------- OCT analysis (upload) ----------------
    @app.route("/analysis", methods=["GET", "POST"])
    def analysis():
        conn = get_db()
        astronauts = db.query(conn, "SELECT id, display_name FROM astronauts ORDER BY id")
        if request.method == "GET":
            return render_template("analysis.html", astronauts=astronauts, phases=PHASES, eyes=EYES,
                                   today=date.today().isoformat(),
                                   preselect=request.args.get("astronaut", ""),
                                   model=inference.model_status(get_cfg().MODEL_PATH))

        cfg = get_cfg()
        file = request.files.get("oct_image")
        astronaut_id = request.form.get("astronaut_id") or None
        exam_date = parse_date(request.form.get("exam_date", ""))
        phase = request.form.get("exam_phase") or None
        eye = request.form.get("eye") or None

        if not file or not file.filename:
            flash("Please select an OCT image.", "error")
            return redirect(url_for("analysis"))
        if not imaging.allowed_file(file.filename):
            flash("Unsupported image format. Use PNG, JPEG, TIFF or BMP.", "error")
            return redirect(url_for("analysis"))
        if astronaut_id and not db.query_one(conn, "SELECT id FROM astronauts WHERE id = ?", (astronaut_id,)):
            flash("Unknown astronaut.", "error")
            return redirect(url_for("analysis"))
        if exam_date is None:
            flash("Please enter a valid examination date.", "error")
            return redirect(url_for("analysis"))
        if phase and phase not in PHASES or eye and eye not in EYES:
            flash("Invalid examination phase or eye.", "error")
            return redirect(url_for("analysis"))

        original = secure_filename(file.filename) or "upload"
        scan_id = ("tmp-" if cfg.PUBLIC_DEMO else "") + uuid.uuid4().hex[:12]
        stored_name = f"{scan_id}_{original}"
        destination = cfg.UPLOAD_DIR / stored_name
        file.save(destination)
        try:
            values = pipeline.process_image(destination, original, scan_id, cfg)
        except imaging.ImageValidationError as exc:
            destination.unlink(missing_ok=True)
            flash(str(exc), "error")
            return redirect(url_for("analysis"))
        except Exception:
            destination.unlink(missing_ok=True)
            log.exception("Processing failed for %s", original)
            flash("The image could not be processed.", "error")
            return redirect(url_for("analysis"))

        values.update({
            "id": scan_id, "filename": original, "stored_name": stored_name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "astronaut_id": astronaut_id, "exam_date": exam_date, "exam_phase": phase, "eye": eye, "is_demo": 0,
        })

        if cfg.PUBLIC_DEMO:
            # Show the result to this visitor only: nothing is stored in the database, the original
            # upload is deleted now and the derived images are removed after DEMO_UPLOAD_TTL_MIN.
            destination.unlink(missing_ok=True)
            cleanup_demo_files(cfg)
            person = db.query_one(conn, "SELECT display_name, mission FROM astronauts WHERE id = ?",
                                  (astronaut_id,)) if astronaut_id else None
            scan = {**values, "display_name": person["display_name"] if person else None,
                    "mission": person["mission"] if person else None}
            comparison, baseline = None, None
            if astronaut_id and monitoring.is_usable(scan) and phase != "baseline":
                baseline = monitoring.baseline_for(conn, astronaut_id, eye, scan["seg_method"])
                if baseline is not None:
                    comparison = monitoring.compare(baseline, scan)
            log.info("Public demo analysis %s (%s)", scan_id, values["status"])
            return render_template("scan_detail.html", scan=scan, comparison=comparison, baseline=baseline,
                                   alerts=[], usable=monitoring.is_usable(scan), ephemeral=True)
        db.insert(conn, "scans", values)
        if astronaut_id:
            monitoring.recompute_alerts(conn, astronaut_id, cfg.ALERT_THRESHOLD_PCT)
        conn.commit()
        log.info("Scan %s stored (%s, quality=%s)", scan_id, values["status"], values["quality_status"])
        flash("Examination uploaded and analysed.", "success")
        return redirect(url_for("scan_detail", scan_id=scan_id))

    # ---------------- Scan detail / delete ----------------
    @app.route("/scans/<scan_id>")
    def scan_detail(scan_id):
        conn = get_db()
        scan = db.query_one(conn, """SELECT s.*, a.display_name, a.mission FROM scans s
                                     LEFT JOIN astronauts a ON a.id = s.astronaut_id WHERE s.id = ?""", (scan_id,))
        if scan is None:
            abort(404)
        comparison, baseline = None, None
        if scan["astronaut_id"] and monitoring.is_usable(scan) and scan["exam_phase"] != "baseline":
            baseline = monitoring.baseline_for(conn, scan["astronaut_id"], scan["eye"], scan["seg_method"])
            if baseline is not None:
                comparison = monitoring.compare(baseline, scan)
        alerts = db.query(conn, "SELECT * FROM alerts WHERE scan_id = ?", (scan_id,))
        return render_template("scan_detail.html", scan=scan, comparison=comparison, baseline=baseline,
                               alerts=alerts, usable=monitoring.is_usable(scan))

    @app.route("/scans/<scan_id>/delete", methods=["POST"])
    def scan_delete(scan_id):
        conn, cfg = get_db(), get_cfg()
        if cfg.PUBLIC_DEMO:
            return demo_disabled_redirect("scan_detail", scan_id=scan_id)
        scan = db.query_one(conn, "SELECT * FROM scans WHERE id = ?", (scan_id,))
        if scan is None:
            abort(404)
        (cfg.UPLOAD_DIR / scan["stored_name"]).unlink(missing_ok=True)
        for name in (scan["preview_name"], scan["processed_name"], scan["overlay_name"]):
            if name:
                (cfg.UPLOAD_DIR / cfg.DERIVED_DIR_NAME / name).unlink(missing_ok=True)
        conn.execute("DELETE FROM alerts WHERE scan_id = ? OR baseline_scan_id = ?", (scan_id, scan_id))
        conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        if scan["astronaut_id"]:
            monitoring.recompute_alerts(conn, scan["astronaut_id"], cfg.ALERT_THRESHOLD_PCT)
        conn.commit()
        flash("Examination deleted.", "success")
        return redirect(url_for("history"))

    # ---------------- Scan history (filters + compare) ----------------
    @app.route("/history")
    def history():
        conn = get_db()
        astronaut = request.args.get("astronaut", "")
        date_from = parse_date(request.args.get("from", ""))
        date_to = parse_date(request.args.get("to", ""))
        sql = """SELECT s.*, a.display_name FROM scans s LEFT JOIN astronauts a ON a.id = s.astronaut_id WHERE 1=1"""
        params = []
        if astronaut == "unassigned":
            sql += " AND s.astronaut_id IS NULL"
        elif astronaut:
            sql += " AND s.astronaut_id = ?"; params.append(astronaut)
        if date_from:
            sql += " AND s.exam_date >= ?"; params.append(date_from)
        if date_to:
            sql += " AND s.exam_date <= ?"; params.append(date_to)
        sql += " ORDER BY s.exam_date DESC, s.created_at DESC"
        scans = db.query(conn, sql, params)
        astronauts = db.query(conn, "SELECT id, display_name FROM astronauts ORDER BY id")
        return render_template("history.html", scans=scans, astronauts=astronauts,
                               f={"astronaut": astronaut, "from": date_from or "", "to": date_to or ""})

    @app.route("/compare")
    def compare():
        ids = request.args.getlist("scan")
        if len(ids) != 2:
            flash("Select exactly two examinations to compare.", "error")
            return redirect(request.referrer or url_for("history"))
        conn = get_db()
        rows = [db.query_one(conn, """SELECT s.*, a.display_name FROM scans s
                                      LEFT JOIN astronauts a ON a.id = s.astronaut_id WHERE s.id = ?""", (i,))
                for i in ids]
        if any(r is None for r in rows):
            abort(404)
        a, b = sorted(rows, key=lambda r: (r["exam_date"] or "", r["created_at"]))
        warnings = []
        if a["astronaut_id"] != b["astronaut_id"]:
            warnings.append("These examinations belong to different astronauts or are unassigned.")
        if a["eye"] != b["eye"]:
            warnings.append("Different eyes (OD/OS) - measurements are not directly comparable.")
        if a["seg_method"] != b["seg_method"]:
            warnings.append("Different segmentation methods - measurements are not comparable.")
        result = None
        if monitoring.is_usable(a) and monitoring.is_usable(b):
            result = monitoring.compare(a, b)
        else:
            warnings.append("At least one examination has no usable measurement (quality or boundary detection).")
        return render_template("compare.html", a=a, b=b, result=result, warnings=warnings)

    # ---------------- Astronauts ----------------
    @app.route("/astronauts")
    def astronauts():
        conn = get_db()
        crew = db.query(conn, """SELECT a.*, COUNT(s.id) n_scans, MAX(s.exam_date) last_exam,
                                 (SELECT COUNT(*) FROM alerts al WHERE al.astronaut_id = a.id) n_alerts
                                 FROM astronauts a LEFT JOIN scans s ON s.astronaut_id = a.id
                                 GROUP BY a.id ORDER BY a.id""")
        return render_template("astronauts.html", crew=crew)

    @app.route("/astronauts/new", methods=["GET", "POST"])
    def astronaut_new():
        if get_cfg().PUBLIC_DEMO:
            return demo_disabled_redirect("astronauts")
        form ={"id": "", "display_name": "", "mission": "", "mission_status": "pre-flight", "notes": ""}
        if request.method == "POST":
            form = {k: (request.form.get(k) or "").strip() for k in form}
            form["id"] = form["id"].upper()
            errors = []
            if not ASTRONAUT_ID_RE.match(form["id"]):
                errors.append("ID must be 3-20 characters: capital letters, digits and hyphens (e.g. AST-005).")
            if not form["display_name"] or len(form["display_name"]) > 60:
                errors.append("Display name is required (max 60 characters). Use an anonymised name.")
            if not form["mission"] or len(form["mission"]) > 60:
                errors.append("Mission is required (max 60 characters).")
            if form["mission_status"] not in MISSION_STATUSES:
                errors.append("Invalid mission status.")
            conn = get_db()
            if not errors and db.query_one(conn, "SELECT id FROM astronauts WHERE id = ?", (form["id"],)):
                errors.append("An astronaut with this ID already exists.")
            if errors:
                for e in errors:
                    flash(e, "error")
            else:
                db.insert(conn, "astronauts", {**form, "notes": form["notes"][:500], "is_demo": 0,
                                               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                conn.commit()
                flash("Astronaut profile created.", "success")
                return redirect(url_for("astronaut_detail", astronaut_id=form["id"]))
        return render_template("astronaut_form.html", form=form, statuses=MISSION_STATUSES)

    @app.route("/astronauts/<astronaut_id>")
    def astronaut_detail(astronaut_id):
        conn = get_db()
        person = db.query_one(conn, "SELECT * FROM astronauts WHERE id = ?", (astronaut_id,))
        if person is None:
            abort(404)
        points = monitoring.timeline(conn, astronaut_id)
        scans = db.query(conn, "SELECT * FROM scans WHERE astronaut_id = ? ORDER BY exam_date, created_at",
                         (astronaut_id,))
        alerts = db.query(conn, "SELECT * FROM alerts WHERE astronaut_id = ? ORDER BY exam_date DESC",
                          (astronaut_id,))
        baselines = {}
        for eye in {s["eye"] for s in scans}:
            for method in {s["seg_method"] for s in scans}:
                b = monitoring.baseline_for(conn, astronaut_id, eye, method)
                if b is not None:
                    baselines[b["id"]] = b
        return render_template("astronaut_detail.html", person=person, points=points, scans=scans,
                               alerts=alerts, baselines=list(baselines.values()))

    # ---------------- Alerts & model ----------------
    @app.route("/alerts")
    def alerts():
        rows = db.query(get_db(), """SELECT al.*, a.display_name, s.eye FROM alerts al
                                     JOIN astronauts a ON a.id = al.astronaut_id
                                     JOIN scans s ON s.id = al.scan_id
                                     ORDER BY al.exam_date DESC""")
        return render_template("alerts.html", alerts=rows)

    @app.route("/nasa")
    def nasa():
        return render_template("nasa.html")

    # ---------------- Files (not inside static/) ----------------
    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_from_directory(get_cfg().UPLOAD_DIR, filename)

    @app.errorhandler(413)
    def too_large(_error):
        flash(f"File too large. Maximum size is {get_cfg().MAX_UPLOAD_MB} MB.", "error")
        return redirect(url_for("analysis"))

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("error.html", code=404, message="Page or record not found."), 404

    @app.errorhandler(500)
    def server_error(_error):
        return render_template("error.html", code=500, message="Unexpected server error. See the log."), 500


def register_cli(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db_command():
        """Create or upgrade the database schema."""
        db.init_db(get_cfg().DB_PATH)
        click.echo("Database ready.")

    @app.cli.command("seed-demo")
    def seed_demo_command():
        """Add fictional astronauts with SYNTHETIC OCT scans."""
        conn = db.connect(get_cfg().DB_PATH)
        try:
            created = demo.seed(conn, get_cfg())
        finally:
            conn.close()
        click.echo(f"Demo astronauts added: {', '.join(created) or 'none (already present)'}")

    @app.cli.command("remove-demo")
    def remove_demo_command():
        """Delete all demo astronauts, scans and files."""
        conn = db.connect(get_cfg().DB_PATH)
        try:
            n = demo.remove(conn, get_cfg())
        finally:
            conn.close()
        click.echo(f"Removed {n} demo scans.")


if __name__ == "__main__":
    application = create_app()
    cfg = application.extensions["orbital_cfg"]
    application.run(host=cfg.HOST, port=cfg.PORT, debug=cfg.DEBUG)
