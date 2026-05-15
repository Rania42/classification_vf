"""
Routes utilitaires : status, toggles agents.
Inclut désormais l'état du microservice d'extraction (port 5001).
"""
from flask import Blueprint, request, jsonify
import core.agents as agents_module
from core.agents import check_ollama_available, GEMMA_MODEL
from core.model import DOC_CLASSES, val_acc, CONFIDENCE_THRESHOLD
from services.mongo import is_mongo_available, get_mongo_db
from config import DEVICE

status_bp = Blueprint("status", __name__)


<<<<<<< HEAD
def _get_extraction_service_status() -> dict:
    try:
        from services.extraction_client import get_service_status
        return get_service_status()
    except ImportError:
        return {"available": False, "error": "extraction_client non installé"}


@status_bp.route("/status", methods=["GET"])
=======
@status_bp.route("/gedia/status", methods=["GET"])
>>>>>>> 70a8997 (local modifications)
def status():
    check_ollama_available()
    mongo_ok      = is_mongo_available()
    mongo_version = None

    if mongo_ok:
        db = get_mongo_db()
        if db is not None:
            try:
                mongo_version = db.client.server_info().get("version")
            except Exception:
                pass

    extraction_svc = _get_extraction_service_status()

    return jsonify({
        "model_loaded":        True,
        "device":              str(DEVICE),
        "classes":             DOC_CLASSES,
        "val_acc":             round(val_acc * 100, 2),
        "threshold":           round(CONFIDENCE_THRESHOLD * 100, 1),
        # Agents LLM
        "qwen":                agents_module.USE_QWEN,
        "qwen_model":          "qwen2.5vl:3b",
        "ollama":              agents_module.USE_OLLAMA,
        "ollama_model":        GEMMA_MODEL,
        "gemma":               agents_module.USE_OLLAMA,
        "gemma_model":         GEMMA_MODEL,
        # Base de données
        "mongodb":             mongo_ok,
        "mongodb_version":     mongo_version,
        # Microservice d'extraction
        "extraction_service":  extraction_svc.get("available", False),
        "extraction_service_url": extraction_svc.get("url", "http://localhost:5001"),
        "extraction_service_detail": extraction_svc,
    })


@status_bp.route("/gedia/toggle_ollama", methods=["POST"])
def toggle_ollama():
    agents_module.USE_OLLAMA = request.json.get("enabled", agents_module.USE_OLLAMA)
    return jsonify({"ollama_enabled": agents_module.USE_OLLAMA,
                    "gemma_enabled":  agents_module.USE_OLLAMA})


@status_bp.route("/toggle_gemma", methods=["POST"])
def toggle_gemma():
    agents_module.USE_OLLAMA = request.json.get("enabled", agents_module.USE_OLLAMA)
    return jsonify({"gemma_enabled": agents_module.USE_OLLAMA})


@status_bp.route("/gedia/toggle_qwen", methods=["POST"])
def toggle_qwen():
    agents_module.USE_QWEN = request.json.get("enabled", agents_module.USE_QWEN)
    return jsonify({"qwen_enabled": agents_module.USE_QWEN})


@status_bp.route("/extraction_service/status", methods=["GET"])
def extraction_service_status():
    """Endpoint dédié au statut du microservice d'extraction."""
    return jsonify(_get_extraction_service_status())