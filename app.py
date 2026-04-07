"""
BankDoc — Point d'entrée Flask.
Initialise les services et enregistre les blueprints.
"""
from flask import Flask
from flask_cors import CORS

# ── Services ───────────────────────────────────────────
from services.mongo import init_mongodb, is_mongo_available
from core.agents import check_ollama_available
from core.model import DOC_CLASSES, val_acc, CONFIDENCE_THRESHOLD
from config import DEVICE

# ── Blueprints ─────────────────────────────────────────
from routes.classify import classify_bp
from routes.documents import docs_bp
from routes.queue import queue_bp
from routes.types import types_bp
from routes.status import status_bp

# ── Initialisation des services ───────────────────────
print("\n" + "=" * 55)
print("  BankDoc - Initialisation des services")
print("=" * 55)

# Initialisation MongoDB
mongo_ok = init_mongodb()
print(f"  MongoDB  : {'✅ CONNECTÉ' if mongo_ok else '❌ INACTIF'}")

# Vérification Ollama
check_ollama_available()

# ── App Flask ──────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="")
CORS(app)

# ── Enregistrement des blueprints ──────────────────────
app.register_blueprint(classify_bp)
app.register_blueprint(docs_bp)
app.register_blueprint(queue_bp)
app.register_blueprint(types_bp)
app.register_blueprint(status_bp)

# ── Pages statiques ────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/library")
def library():
    return app.send_static_file("library.html")

@app.route("/batch")
@app.route("/queue")
def queue_page():
    return app.send_static_file("queue.html")

# ── Démarrage ──────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(f"  Device   : {DEVICE}")
    print(f"  Classes  : {len(DOC_CLASSES)}")
    print(f"  Val acc  : {val_acc:.4f}  |  Seuil : {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  MongoDB  : {'✅ CONNECTÉ' if is_mongo_available() else '❌ INACTIF'}")
    print("  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)