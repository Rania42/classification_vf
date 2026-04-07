"""
Routes utilitaires : status, toggles agents.
"""
from flask import Blueprint, request, jsonify
import core.agents as agents_module
from core.agents import check_ollama_available
from core.model import DOC_CLASSES, val_acc, CONFIDENCE_THRESHOLD
from services.mongo import is_mongo_available, get_mongo_db
from config import DEVICE

status_bp = Blueprint("status", __name__)


@status_bp.route("/status", methods=["GET"])
def status():
    check_ollama_available()
    mongo_ok = is_mongo_available()
    mongo_version = None
    
    if mongo_ok:
        db = get_mongo_db()
        if db is not None:  # ← CORRECTION: utiliser is not None au lieu de if db:
            try:
                mongo_version = db.client.server_info().get("version")
            except Exception:
                pass

    return jsonify({
        "model_loaded":  True,
        "device":        str(DEVICE),
        "classes":       DOC_CLASSES,
        "val_acc":       round(val_acc * 100, 2),
        "threshold":     round(CONFIDENCE_THRESHOLD * 100, 1),
        "ollama":        agents_module.USE_OLLAMA,
        "ollama_model":  agents_module.OLLAMA_MODEL if hasattr(agents_module, "OLLAMA_MODEL") else "llama3.2",
        "qwen":          agents_module.USE_QWEN,
        "qwen_model":    "qwen2.5vl:3b",
        "mongodb":       mongo_ok,
        "mongodb_version": mongo_version,
    })


@status_bp.route("/toggle_ollama", methods=["POST"])
def toggle_ollama():
    agents_module.USE_OLLAMA = request.json.get("enabled", agents_module.USE_OLLAMA)
    return jsonify({"ollama_enabled": agents_module.USE_OLLAMA})


@status_bp.route("/toggle_qwen", methods=["POST"])
def toggle_qwen():
    agents_module.USE_QWEN = request.json.get("enabled", agents_module.USE_QWEN)
    return jsonify({"qwen_enabled": agents_module.USE_QWEN})