"""
Worker de queue persistant côté serveur.
Traitement séquentiel en background thread (daemon).
Utilise le même pipeline complet que /classify.
"""
import os
import uuid
import threading
from datetime import datetime

from config import QUEUE_UPLOAD_DIR
from core.pipeline import run_pipeline
from services.mongo import save_document_to_mongo

_queue_store: dict = {}
_queue_lock        = threading.Lock()
_queue_worker      = None


def get_queue_store() -> dict:
    return _queue_store


def _worker_fn():
    """Thread background : traite les items pending avec le pipeline complet."""
    while True:
        item = None
        with _queue_lock:
            for it in _queue_store.values():
                if it["status"] == "pending":
                    it["status"] = "processing"
                    item = it
                    break
        if item is None:
            break

        tmp_path = item["path"]
        try:
            # ── Pipeline complet identique à /classify ──
            result = run_pipeline(tmp_path, item["filename"])

            # Sauvegarde MongoDB
            mongo_id = save_document_to_mongo(tmp_path, item["filename"], result)
            if mongo_id:
                result["mongo_id"] = mongo_id
                result["steps"].append({
                    "step": "Sauvegarde MongoDB", "status": "ok",
                    "detail": f"ID : {mongo_id} → {result['prediction']}",
                })

            with _queue_lock:
                item["status"]      = "done"
                item["result"]      = result
                item["finished_at"] = datetime.utcnow().isoformat() + "Z"

        except Exception as exc:
            import traceback
            print(f"[Queue Worker] Erreur {item['filename']}: {traceback.format_exc()}")
            with _queue_lock:
                item["status"]      = "error"
                item["result"]      = {"error": str(exc), "prediction": "erreur", "confidence": 0}
                item["finished_at"] = datetime.utcnow().isoformat() + "Z"
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    global _queue_worker
    _queue_worker = None
    print("[Queue Worker] Thread terminé (queue vide)")


def ensure_worker():
    """Lance le worker si pas déjà actif."""
    global _queue_worker
    if _queue_worker is None or not _queue_worker.is_alive():
        _queue_worker = threading.Thread(target=_worker_fn, daemon=True)
        _queue_worker.start()
        print("[Queue Worker] Démarré")


def add_items_to_queue(files: list, use_qwen: bool = False) -> list:
    """
    Ajoute une liste de fichiers (werkzeug FileStorage) à la queue.
    Retourne la liste des items créés.
    """
    items = []
    for file in files:
        if not file or file.filename == "":
            continue
        ext      = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
        item_id  = str(uuid.uuid4())
        tmp_path = os.path.join(QUEUE_UPLOAD_DIR, f"{item_id}.{ext}")
        file.save(tmp_path)

        item = {
            "id":          item_id,
            "filename":    file.filename,
            "path":        tmp_path,
            "status":      "pending",
            "result":      None,
            "added_at":    datetime.utcnow().isoformat() + "Z",
            "finished_at": None,
            "use_qwen":    use_qwen,
        }
        with _queue_lock:
            _queue_store[item_id] = item
        items.append({"id": item_id, "filename": file.filename, "status": "pending"})

    ensure_worker()
    return items


def get_all_items() -> list:
    with _queue_lock:
        result = []
        for it in _queue_store.values():
            slim = {k: v for k, v in it.items() if k != "path"}
            # Tronquer l'OCR dans la réponse de liste
            if slim.get("result") and slim["result"].get("ocr_text"):
                slim["result"] = {**slim["result"],
                                  "ocr_text": slim["result"]["ocr_text"][:200]}
            result.append(slim)
        return result


def delete_item(item_id: str) -> tuple[bool, str]:
    with _queue_lock:
        item = _queue_store.get(item_id)
        if not item:
            return False, "introuvable"
        if item["status"] == "processing":
            return False, "en cours de traitement"
        path = item.get("path", "")
        del _queue_store[item_id]
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    return True, "ok"


def clear_items(mode: str = "finished") -> int:
    """mode='finished' | 'all'"""
    with _queue_lock:
        to_del = [
            iid for iid, it in _queue_store.items()
            if (mode == "all" and it["status"] != "processing")
            or (mode == "finished" and it["status"] in ("done", "error"))
        ]
        for iid in to_del:
            try:
                if os.path.exists(_queue_store[iid].get("path", "")):
                    os.remove(_queue_store[iid]["path"])
            except Exception:
                pass
            del _queue_store[iid]
    return len(to_del)


def worker_alive() -> bool:
    return _queue_worker is not None and _queue_worker.is_alive()