"""
Route POST /classify — pipeline complet synchrone.
"""
import os
import uuid
from flask import Blueprint, request, jsonify

from config import UPLOAD_TEMP
from core.pipeline import run_pipeline
from services.mongo import save_document_to_mongo

classify_bp = Blueprint("classify", __name__)


@classify_bp.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file = request.files["file"]
    ext  = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    tmp  = os.path.join(UPLOAD_TEMP, f"{uuid.uuid4().hex}.{ext}")
    file.save(tmp)

    try:
        result = run_pipeline(tmp, file.filename)

        mongo_id = save_document_to_mongo(tmp, file.filename, result)
        if mongo_id:
            result["mongo_id"] = mongo_id
            result["steps"].append({
                "step": "Sauvegarde MongoDB", "status": "ok",
                "detail": f"ID : {mongo_id} → {result['prediction']}",
            })
        else:
            result["steps"].append({
                "step": "Sauvegarde MongoDB", "status": "warning",
                "detail": "MongoDB non disponible — document non sauvegardé",
            })

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"[/classify ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e), "prediction": "erreur", "confidence": 0}), 500

    finally:
        if os.path.exists(tmp):
            os.remove(tmp)