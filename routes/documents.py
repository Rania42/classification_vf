"""
Routes CRUD pour les documents MongoDB.
"""
import io
import os
import shutil
from datetime import datetime

from flask import Blueprint, request, jsonify, send_file
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId

from services.mongo import get_mongo_db, get_gridfs, is_mongo_available, _serialize_doc
from config import DOCS_FOLDER
from core.ocr import extract_text_ocr
from services.extraction import extract_metadata_with_llm, DOC_FIELD_SCHEMAS

docs_bp = Blueprint("documents", __name__)


def _get_db():
    """Retourne l'instance MongoDB avec vérification."""
    db = get_mongo_db()
    if db is None:
        return None
    return db


def _get_fs():
    """Retourne l'instance GridFS avec vérification."""
    fs = get_gridfs()
    if fs is None:
        return None
    return fs


def _require_mongo():
    """Vérifie que MongoDB est disponible."""
    if not is_mongo_available():
        return jsonify({"error": "MongoDB non disponible", "documents": [], "total": 0}), 503
    return None


# ── Liste ──────────────────────────────────────────────
@docs_bp.route("/documents", methods=["GET"])
def list_documents():
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible", "documents": [], "total": 0}), 503

    category = request.args.get("category", "")
    q = request.args.get("q", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 24)), 100)
    skip = (page - 1) * per_page
    sort = request.args.get("sort", "date_desc")
    conf_min = float(request.args.get("conf_min", 0))

    query = {}
    if category:
        query["prediction"] = category
    if conf_min > 0:
        query["confidence"] = {"$gte": conf_min}
    if q:
        query["$or"] = [
            {"original_filename": {"$regex": q, "$options": "i"}},
            {"ocr_text": {"$regex": q, "$options": "i"}},
            {"prediction": {"$regex": q, "$options": "i"}},
        ]

    sort_map = {
        "date_desc": [("uploaded_at", DESCENDING)],
        "date_asc": [("uploaded_at", ASCENDING)],
        "conf_desc": [("confidence", DESCENDING)],
        "conf_asc": [("confidence", ASCENDING)],
        "name_asc": [("original_filename", ASCENDING)],
    }
    mongo_sort = sort_map.get(sort, [("uploaded_at", DESCENDING)])

    total = db.documents.count_documents(query)
    docs = list(
        db.documents.find(query, {"ocr_text": 0})
        .sort(mongo_sort).skip(skip).limit(per_page)
    )
    return jsonify({
        "documents": [_serialize_doc(d) for d in docs],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": max((total + per_page - 1) // per_page, 1),
    })


# ── Stats ──────────────────────────────────────────────
@docs_bp.route("/documents/stats", methods=["GET"])
def documents_stats():
    if not is_mongo_available():
        return jsonify({"stats": [], "total": 0, "mongodb": False})
    
    db = _get_db()
    if db is None:
        return jsonify({"stats": [], "total": 0, "mongodb": False})

    pipeline = [
        {"$group": {
            "_id": "$prediction",
            "count": {"$sum": 1},
            "avg_conf": {"$avg": "$confidence"},
            "last_at": {"$max": "$uploaded_at"},
        }},
        {"$sort": {"count": -1}},
    ]
    stats = list(db.documents.aggregate(pipeline))
    total = db.documents.count_documents({})
    return jsonify({
        "stats": [{
            "category": s["_id"],
            "count": s["count"],
            "avg_confidence": round(s["avg_conf"] or 0, 1),
            "last_upload": s["last_at"].isoformat() + "Z" if s["last_at"] else None,
        } for s in stats],
        "total": total,
        "mongodb": True,
    })


# ── Détail ─────────────────────────────────────────────
@docs_bp.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404
        return jsonify(_serialize_doc(doc))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Corriger catégorie ─────────────────────────────────
@docs_bp.route("/documents/<doc_id>/correct", methods=["PATCH"])
def correct_document(doc_id):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        data = request.json or {}
        new_pred = data.get("prediction", "").strip()
        if not new_pred:
            return jsonify({"error": "Catégorie manquante"}), 400

        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        old_pred = doc.get("prediction", "inconnu")
        old_path = doc.get("stored_path", "")
        new_path = old_path

        if old_path and os.path.exists(old_path) and old_pred != new_pred:
            fn = doc.get("stored_filename", os.path.basename(old_path))
            new_folder = os.path.join(DOCS_FOLDER, new_pred)
            os.makedirs(new_folder, exist_ok=True)
            new_path = os.path.join(new_folder, fn)
            try:
                shutil.move(old_path, new_path)
            except Exception as e:
                print(f"[Correct] Déplacement échoué : {e}")
                new_path = old_path

        upd = {
            "prediction": new_pred,
            "corrected": True,
            "original_prediction": old_pred,
            "corrected_at": datetime.utcnow(),
        }
        if new_path != old_path:
            upd["stored_path"] = new_path
        db.documents.update_one({"_id": ObjectId(doc_id)}, {"$set": upd})
        return jsonify({"success": True, "old": old_pred, "new": new_pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Téléchargement ─────────────────────────────────────
@docs_bp.route("/documents/<doc_id>/download", methods=["GET"])
def download_document(doc_id):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    fs = _get_fs()
    if db is None or fs is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404
        stored = doc.get("stored_path", "")
        if stored and os.path.exists(stored):
            return send_file(stored, as_attachment=True, download_name=doc["original_filename"])
        gid = doc.get("gridfs_id")
        if gid:
            gf = fs.get(gid)
            return send_file(
                io.BytesIO(gf.read()), as_attachment=True,
                download_name=doc["original_filename"],
                mimetype=gf.content_type or "application/octet-stream",
            )
        return jsonify({"error": "Fichier introuvable"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Suppression ────────────────────────────────────────
@docs_bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    fs = _get_fs()
    if db is None or fs is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404
        stored = doc.get("stored_path", "")
        if stored and os.path.exists(stored):
            os.remove(stored)
        gid = doc.get("gridfs_id")
        if gid:
            try:
                fs.delete(gid)
            except Exception:
                pass
        db.documents.delete_one({"_id": ObjectId(doc_id)})
        return jsonify({"success": True, "deleted_id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Extraction métadonnées LLM ─────────────────────────
@docs_bp.route("/documents/<doc_id>/extract", methods=["POST"])
def extract_document_metadata(doc_id):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        import time
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        doc_type = doc.get("prediction", "inconnu")
        ocr_text = doc.get("ocr_text", "")
        stored_path = doc.get("stored_path", "")
        force = (request.json or {}).get("force", False)

        # Re-OCR si texte trop court
        from config import OCR_MIN_LENGTH
        if stored_path and os.path.exists(stored_path) and len(ocr_text) < OCR_MIN_LENGTH:
            fresh, _ = extract_text_ocr(stored_path)
            if len(fresh) > len(ocr_text):
                ocr_text = fresh
                db.documents.update_one(
                    {"_id": ObjectId(doc_id)}, {"$set": {"ocr_text": ocr_text}}
                )

        # Cache
        if not force and doc.get("extracted_fields"):
            cached = doc["extracted_fields"]
            if any(v for v in cached.values()):
                return jsonify({
                    "fields": cached,
                    "source": doc.get("extraction_source", "cache"),
                    "doc_type": doc_type,
                    "cached": True,
                })

        if not ocr_text or ocr_text == "[NO_TEXT]":
            return jsonify({"error": "Aucun texte OCR disponible"}), 400
        if doc_type not in DOC_FIELD_SCHEMAS:
            return jsonify({
                "fields": {},
                "source": "none",
                "doc_type": doc_type,
                "error": f"Type '{doc_type}' non supporté"
            })

        t0 = time.time()
        fields = extract_metadata_with_llm(ocr_text, doc_type, stored_path)
        source = fields.pop("_source", "none")

        db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "extracted_fields": fields,
                "extraction_source": source,
                "extracted_at": datetime.utcnow(),
            }}
        )
        return jsonify({
            "fields": fields,
            "source": source,
            "doc_type": doc_type,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "cached": False,
        })
    except Exception as e:
        import traceback
        print(f"[Extract ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ── Classification manuelle ────────────────────────────
@docs_bp.route("/documents/<doc_id>/classify_manual", methods=["POST"])
def classify_manual(doc_id):
    err = _require_mongo()
    if err:
        return err
    
    db = _get_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        import re
        data = request.json or {}
        type_name = data.get("type_name", "").strip().lower().replace(" ", "_")
        create_new = data.get("create_new", False)
        label = data.get("label", type_name.replace("_", " ").title())

        if not type_name:
            return jsonify({"error": "type_name requis"}), 400

        from config import TRAINING_THRESHOLD
        if create_new:
            if not db.doc_types.find_one({"name": type_name}):
                if not re.match(r"^[a-z][a-z0-9_]{1,49}$", type_name):
                    return jsonify({"error": "Nom de type invalide"}), 400
                db.doc_types.insert_one({
                    "name": type_name,
                    "label": label,
                    "is_custom": True,
                    "count": 0,
                    "ready_for_training": False,
                    "created_at": datetime.utcnow(),
                    "description": data.get("description", ""),
                })
                os.makedirs(os.path.join(DOCS_FOLDER, type_name), exist_ok=True)
        else:
            if not db.doc_types.find_one({"name": type_name}):
                return jsonify({"error": f"Type '{type_name}' introuvable"}), 404

        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        old_type = doc.get("prediction", "inconnu")
        old_path = doc.get("stored_path", "")
        new_path = old_path

        if old_path and os.path.exists(old_path) and old_type != type_name:
            fn = doc.get("stored_filename", os.path.basename(old_path))
            new_folder = os.path.join(DOCS_FOLDER, type_name)
            os.makedirs(new_folder, exist_ok=True)
            new_path = os.path.join(new_folder, fn)
            try:
                shutil.move(old_path, new_path)
            except Exception as e:
                print(f"[Manual] Déplacement échoué : {e}")
                new_path = old_path

        upd = {
            "prediction": type_name,
            "manually_classified": True,
            "manual_classifier": data.get("user", "human"),
            "classified_at": datetime.utcnow(),
            "original_prediction": old_type,
            "original_confidence": doc.get("confidence", 0),
            "needs_manual": False,
        }
        if new_path != old_path:
            upd["stored_path"] = new_path
        db.documents.update_one({"_id": ObjectId(doc_id)}, {"$set": upd})

        new_count = db.documents.count_documents({"prediction": type_name})
        ready = new_count >= TRAINING_THRESHOLD

        return jsonify({
            "success": True,
            "old_type": old_type,
            "new_type": type_name,
            "doc_count": new_count,
            "ready_for_training": ready,
            "training_threshold": TRAINING_THRESHOLD,
        })
    except Exception as e:
        import traceback
        print(f"[Manual ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500