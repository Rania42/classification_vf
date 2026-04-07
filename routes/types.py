"""
Routes gestion des types de documents (natifs + personnalisés).
"""
import os
import re
from datetime import datetime
from flask import Blueprint, request, jsonify

from services.mongo import is_mongo_available, get_mongo_db
from config import DOCS_FOLDER, TRAINING_THRESHOLD

types_bp = Blueprint("types", __name__)


def _require_mongo():
    if not is_mongo_available():
        return jsonify({"error": "MongoDB non disponible"}), 503
    return None


def _get_db():
    return get_mongo_db()


@types_bp.route("/types", methods=["GET"])
def list_types():
    if not is_mongo_available():
        native = ["carte_identite","rib","cheque","tableau_amortissement",
                  "acte_naissance","acte_heredite","assurance","attestation_solde"]
        return jsonify({"types": [{"name": t, "label": t.replace("_"," ").title(),
                                   "is_custom": False, "count": 0} for t in native]})
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    types = list(db.doc_types.find({}, {"_id": 0}).sort("name", 1))
    for t in types:
        t["count"] = db.documents.count_documents({"prediction": t["name"]})
        t["ready_for_training"] = t["count"] >= TRAINING_THRESHOLD
        if t.get("created_at"):
            t["created_at"] = t["created_at"].isoformat() + "Z"
    return jsonify({"types": types, "training_threshold": TRAINING_THRESHOLD})


@types_bp.route("/types", methods=["POST"])
def create_type():
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    data  = request.json or {}
    name  = data.get("name", "").strip().lower().replace(" ", "_").replace("-", "_")
    label = data.get("label", "").strip() or name.replace("_", " ").title()

    if not name:
        return jsonify({"error": "Nom requis"}), 400
    if not re.match(r"^[a-z][a-z0-9_]{1,49}$", name):
        return jsonify({"error": "Nom invalide"}), 400
    if db.doc_types.find_one({"name": name}):
        return jsonify({"error": f"Type '{name}' existe déjà"}), 409

    db.doc_types.insert_one({
        "name": name, "label": label, "is_custom": True,
        "count": 0, "ready_for_training": False,
        "created_at": datetime.utcnow(),
        "description": data.get("description", ""),
    })
    folder = os.path.join(DOCS_FOLDER, name)
    os.makedirs(folder, exist_ok=True)
    return jsonify({
        "success": True,
        "type": {"name": name, "label": label, "is_custom": True,
                 "count": 0, "ready_for_training": False}
    }), 201


@types_bp.route("/types/<type_name>", methods=["DELETE"])
def delete_type(type_name):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    doc_type = db.doc_types.find_one({"name": type_name})
    if not doc_type:
        return jsonify({"error": "Type introuvable"}), 404
    if not doc_type.get("is_custom"):
        return jsonify({"error": "Types natifs non supprimables"}), 403
    count = db.documents.count_documents({"prediction": type_name})
    if count > 0:
        return jsonify({"error": f"Ce type contient {count} document(s)"}), 409
    db.doc_types.delete_one({"name": type_name})
    return jsonify({"success": True})


@types_bp.route("/types/stats", methods=["GET"])
def types_stats():
    if not is_mongo_available():
        return jsonify({"custom_types": [], "alerts": [], "threshold": TRAINING_THRESHOLD})
    
    db = _get_db()
    if db is None:
        return jsonify({"custom_types": [], "alerts": [], "threshold": TRAINING_THRESHOLD})
    
    custom = list(db.doc_types.find({"is_custom": True}, {"_id": 0}))
    alerts = []
    for t in custom:
        t["count"] = db.documents.count_documents({"prediction": t["name"]})
        t["ready_for_training"] = t["count"] >= TRAINING_THRESHOLD
        if t.get("created_at"):
            t["created_at"] = t["created_at"].isoformat() + "Z"
        if t["count"] >= TRAINING_THRESHOLD:
            alerts.append({"type": t["name"], "label": t["label"], "count": t["count"]})
    return jsonify({"custom_types": custom, "alerts": alerts, "threshold": TRAINING_THRESHOLD})