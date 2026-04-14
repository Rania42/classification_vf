"""
Initialisation MongoDB, sauvegarde et sérialisation des documents.
Version avec lazy loading et singleton robuste.
"""
import os
import uuid
import shutil
import gridfs
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId

from config import MONGO_URI, MONGO_DB, DOCS_FOLDER

# État interne - ne pas accéder directement en dehors de ce module
_mongo_client = None
_mongo_db = None
_fs = None
_mongo_available = False
_mongo_initialized = False


def init_mongodb(force_reconnect=False) -> bool:
    """
    Initialise la connexion MongoDB.
    
    Args:
        force_reconnect: Force une reconnexion même si déjà initialisé
    
    Returns:
        bool: True si connexion réussie, False sinon
    """
    global _mongo_client, _mongo_db, _fs, _mongo_available, _mongo_initialized
    
    # Éviter de réinitialiser si déjà connecté
    if _mongo_initialized and _mongo_available and not force_reconnect:
        return True
    
    print("[MongoDB] Initialisation...")
    try:
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        _mongo_client.server_info()  # Vérifie la connexion
        _mongo_db = _mongo_client[MONGO_DB]
        _fs = gridfs.GridFS(_mongo_db)

        # Création des index
        _mongo_db.documents.create_index("prediction")
        _mongo_db.documents.create_index("original_filename")
        _mongo_db.documents.create_index("uploaded_at")
        _mongo_db.documents.create_index("confidence")
        _mongo_db.documents.create_index([("ocr_text", "text"), ("original_filename", "text")])

        # Types natifs
        from config import TRAINING_THRESHOLD
        native = ["rib", "cheque", "tableau_amortissement",
                  "acte_naissance", "acte_heredite", "assurance", "attestation_solde"]
        for t in native:
            _mongo_db.doc_types.update_one(
                {"name": t},
                {"$setOnInsert": {
                    "name": t,
                    "label": t.replace("_", " ").title(),
                    "is_custom": False,
                    "count": 0,
                    "ready_for_training": False,
                    "created_at": datetime.utcnow(),
                }},
                upsert=True,
            )
        _mongo_db.doc_types.create_index("name", unique=True)

        _mongo_available = True
        _mongo_initialized = True
        print("[MongoDB] ✅ Connecté — base : bankdoc")
        return True
        
    except Exception as e:
        print(f"[MongoDB] ❌ Erreur : {e}")
        _mongo_available = False
        _mongo_initialized = True  # Marqué comme initialisé même en échec
        return False


def is_mongo_available() -> bool:
    """
    Vérifie si MongoDB est disponible.
    Tente une reconnexion si nécessaire.
    """
    global _mongo_available, _mongo_initialized
    
    # Si déjà marqué comme disponible, retourner True
    if _mongo_available:
        return True
    
    # Si déjà initialisé mais échoué, ne pas réessayer automatiquement
    if _mongo_initialized and not _mongo_available:
        return False
    
    # Première initialisation
    return init_mongodb()


def get_mongo_db():
    """Retourne l'instance de la base de données MongoDB."""
    if is_mongo_available():
        return _mongo_db
    return None


def get_gridfs():
    """Retourne l'instance GridFS."""
    if is_mongo_available():
        return _fs
    return None


def _serialize_doc(doc: dict) -> dict:
    """Sérialise un document MongoDB en JSON-safe dict."""
    if doc is None:
        return {}
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    if "gridfs_id" in doc and doc.get("gridfs_id"):
        doc["gridfs_id"] = str(doc["gridfs_id"])
    if doc.get("uploaded_at"):
        doc["uploaded_at"] = doc["uploaded_at"].isoformat() + "Z"
    if doc.get("corrected_at"):
        doc["corrected_at"] = doc["corrected_at"].isoformat() + "Z"
    if doc.get("extracted_at"):
        doc["extracted_at"] = doc["extracted_at"].isoformat() + "Z"
    return doc


def save_document_to_mongo(file_path: str, original_filename: str, result: dict) -> str | None:
    """Sauvegarde un document dans MongoDB."""
    if not is_mongo_available():
        print("[MongoDB] Sauvegarde ignorée : MongoDB non disponible")
        return None
    
    try:
        ext = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else "jpg"
        doc_id = str(uuid.uuid4())
        category = result.get("prediction", "inconnu")

        cat_folder = os.path.join(DOCS_FOLDER, category)
        os.makedirs(cat_folder, exist_ok=True)
        stored_filename = f"{doc_id}.{ext}"
        stored_path = os.path.join(cat_folder, stored_filename)
        shutil.copy2(file_path, stored_path)

        with open(file_path, "rb") as f_bin:
            gridfs_id = _fs.put(
                f_bin,
                filename=stored_filename,
                content_type=f"image/{ext}" if ext != "pdf" else "application/pdf",
                metadata={"category": category, "doc_id": doc_id},
            )

        doc = {
            "doc_id": doc_id,
            "original_filename": original_filename,
            "stored_filename": stored_filename,
            "stored_path": stored_path,
            "gridfs_id": gridfs_id,
            "prediction": category,
            "confidence": result.get("confidence", 0),
            "path": result.get("path", ""),
            "ocr_text": result.get("ocr_text", ""),
            "all_scores": result.get("all_scores", []),
            "qwen": result.get("qwen", {}),
            "votes": result.get("votes", {}),
            "agreement": result.get("agreement", True),
            "needs_manual": result.get("needs_manual", False),
            "degraded_image": result.get("degraded_image", False),
            "agents_used": result.get("agents_used", []),
            "time_ms": result.get("time_ms", 0),
            "uploaded_at": datetime.utcnow(),
            "file_size_bytes": os.path.getsize(file_path),
            "corrected": False,
        }
        insert_result = _mongo_db.documents.insert_one(doc)
        mongo_id = str(insert_result.inserted_id)
        print(f"[MongoDB] ✅ Document sauvegardé : {mongo_id} → {category}/{stored_filename}")
        return mongo_id
        
    except Exception as e:
        print(f"[MongoDB] ❌ Erreur sauvegarde : {e}")
        return None