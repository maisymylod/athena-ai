"""Flask server: serves frontend + POST /api/detect for deepfake detection."""

import logging
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Lazy-loaded inference engine
_inference = None


def _get_inference():
    """Lazy-load the inference model on first request."""
    global _inference
    if _inference is not None:
        return _inference

    checkpoint_path = os.environ.get(
        "ATHENA_CHECKPOINT", "checkpoints/best_model.pt"
    )

    if not Path(checkpoint_path).exists():
        logger.warning(
            "No checkpoint found at %s — /api/detect will return an error. "
            "Train a model first with: python scripts/run_training.py",
            checkpoint_path,
        )
        return None

    from ml.inference import DeepfakeInference

    _inference = DeepfakeInference.from_checkpoint(checkpoint_path)
    return _inference


def create_app() -> Flask:
    """Create and configure the Flask application."""
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/")
    def serve_index():
        return send_from_directory(str(project_root), "index.html")

    @app.route("/api/detect", methods=["POST"])
    def detect():
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "No file selected"}), 400

        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return (
                jsonify(
                    {
                        "error": f"Unsupported file type: {ext}. "
                        f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                    }
                ),
                400,
            )

        inference = _get_inference()
        if inference is None:
            return (
                jsonify(
                    {
                        "error": "No trained model available. "
                        "Train a model first with: python scripts/run_training.py"
                    }
                ),
                503,
            )

        # Save to temp file and run inference
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        try:
            is_synthetic, confidence, indicators = inference.predict(tmp_path)
            return jsonify(
                {
                    "is_synthetic": is_synthetic,
                    "confidence": round(confidence, 4),
                    "label": "AI-Generated" if is_synthetic else "Real",
                    "indicators": indicators,
                }
            )
        finally:
            os.unlink(tmp_path)

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413

    return app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Athena detection server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
