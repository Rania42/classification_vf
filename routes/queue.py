"""
Routes de la file d'attente batch.
"""
from flask import Blueprint, request, jsonify
from services.queue_worker import (
    add_items_to_queue, get_all_items, delete_item,
    clear_items, worker_alive, get_queue_store,
)

queue_bp = Blueprint("queue", __name__)


@queue_bp.route("/gedia/queue/upload", methods=["POST"])
def queue_upload():
    files     = request.files.getlist("files")
    use_qwen = True

    if not files:
        return jsonify({"error": "Aucun fichier"}), 400

    items = add_items_to_queue(files, use_qwen=use_qwen)
    if not items:
        return jsonify({"error": "Aucun fichier valide"}), 400

    return jsonify({"uploaded": len(items), "items": items})


@queue_bp.route("/gedia/queue/status", methods=["GET"])
def queue_status():
    return jsonify({
        "items":        get_all_items(),
        "worker_alive": worker_alive(),
    })


@queue_bp.route("/gedia/queue/item/<item_id>", methods=["GET"])
def queue_item_get(item_id):
    store = get_queue_store()
    item  = store.get(item_id)
    if not item:
        return jsonify({"error": "Item introuvable"}), 404
    return jsonify({k: v for k, v in item.items() if k != "path"})


@queue_bp.route("/gedia/queue/item/<item_id>", methods=["DELETE"])
def queue_item_delete(item_id):
    ok, reason = delete_item(item_id)
    if not ok:
        return jsonify({"error": reason}), 409 if reason == "en cours de traitement" else 404
    return jsonify({"success": True})


@queue_bp.route("/gedia/queue/clear", methods=["POST"])
def queue_clear():
    mode    = (request.json or {}).get("mode", "finished")
    cleared = clear_items(mode)
    return jsonify({"cleared": cleared})