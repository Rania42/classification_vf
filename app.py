"""
BankDoc — Point d'entrée Flask.
Initialise les services et enregistre les blueprints.
"""
from flask import Flask
from flask_cors import CORS

from services.mongo import init_mongodb, is_mongo_available
from core.agents import check_ollama_available
from core.model import DOC_CLASSES, val_acc, CONFIDENCE_THRESHOLD
from config import DEVICE

from routes.classify import classify_bp
from routes.documents import docs_bp
from routes.queue import queue_bp
from routes.types import types_bp
from routes.status import status_bp
from routes.smart_search import smart_search_bp

print("\n" + "=" * 55)
print("  BankDoc - Initialisation des services")
print("=" * 55)

mongo_ok = init_mongodb()
print(f"  MongoDB  : {'✅ CONNECTÉ' if mongo_ok else '❌ INACTIF'}")

check_ollama_available()

app = Flask(__name__, template_folder="templates", static_folder="templates", static_url_path="")
CORS(app)

app.register_blueprint(classify_bp)
app.register_blueprint(docs_bp)
app.register_blueprint(queue_bp)
app.register_blueprint(types_bp)
app.register_blueprint(status_bp)
app.register_blueprint(smart_search_bp)

@app.route("/")
def index():
    return app.send_static_file("dashboard.html")

@app.route("/library")
def library():
    return app.send_static_file("library.html")

@app.route("/batch")
@app.route("/queue")
def queue_page():
    return app.send_static_file("queue.html")

@app.route("/search")
def search_page():
    return app.send_static_file("smart_search.html")

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print(f"  Device   : {DEVICE}")
    print(f"  Classes  : {len(DOC_CLASSES)}")
    print(f"  Val acc  : {val_acc:.4f}  |  Seuil : {CONFIDENCE_THRESHOLD:.0%}")
    print(f"  MongoDB  : {'✅ CONNECTÉ' if is_mongo_available() else '❌ INACTIF'}")
    print("  http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)