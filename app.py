import os
import io
import re
import time
import uuid
import base64
import shutil
import requests
import numpy as np
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections import Counter
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel

import easyocr
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# MongoDB
from pymongo import MongoClient, ASCENDING, DESCENDING
from bson import ObjectId
from bson.json_util import dumps as bson_dumps
import gridfs

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = "model/bank_doc_classifier_multimodal.pth"
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_CHAT  = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2"
QWEN_MODEL   = "qwen2.5vl:3b"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OLLAMA_TIMEOUT      = 10
OLLAMA_MAX_RETRIES  = 1
QWEN_TIMEOUT        = 120
USE_OLLAMA          = True
USE_QWEN            = True

# MongoDB config
MONGO_URI    = "mongodb://localhost:27017/"
MONGO_DB     = "bankdoc"
DOCS_FOLDER  = "stored_documents"   # dossier local pour copies physiques

os.makedirs(DOCS_FOLDER, exist_ok=True)

KEYWORD_RULES = {
    "acte_heredite": [
        "acte de notoriété", "succession", "héritier", "héritière", "défunt",
        "dévolution successorale", "notaire", "héritage", "ayant droit",
        "décès", "hoirie", "masse successorale",
    ],
    "rib": [
        "IBAN", "BIC", "RIB", "domiciliation bancaire", "titulaire du compte",
        "code banque", "code guichet", "numéro de compte", "clé RIB",
    ],
    "cheque": [
        "chèque", "payez contre ce chèque", "à l'ordre de", "chéquier",
        "tiré sur", "montant en lettres", "endossement", "barrement",
    ],
    "tableau_amortissement": [
        "tableau d'amortissement", "amortissement", "échéance", "mensualité",
        "capital restant dû", "intérêts", "taux d'intérêt", "durée du prêt",
        "remboursement", "annuité", "période", "capital amorti",
    ],
    "acte_naissance": [
        "acte de naissance", "né(e) le", "commune de naissance", "extrait",
        "officier d'état civil", "naissance", "registre des naissances",
        "filiation", "père", "mère", "lieu de naissance",
    ],
    "assurance": [
        "police d'assurance", "assuré", "assureur", "prime", "garantie",
        "sinistre", "contrat d'assurance", "couverture", "franchise",
        "bénéficiaire", "risque assuré", "cotisation", "résiliation",
    ],
    "attestation_solde": [
        "attestation de solde", "solde créditeur", "solde du compte",
        "certifions", "attestons", "arrêté au", "situation de compte",
        "disponibilité", "avoir en compte", "banque atteste",
    ],
    "carte_identite": [
        "carte nationale d'identité", "cni", "cin", "carte d'identité",
        "né le", "n° cin", "numéro cin", "national identity card",
        "بطاقة الوطنية", "identity card", "pièce d'identité",
    ],
}

# ─────────────────────────────────────────────
# MONGODB INIT
# ─────────────────────────────────────────────
mongo_client = None
mongo_db     = None
fs           = None

def init_mongodb():
    global mongo_client, mongo_db, fs
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        mongo_client.server_info()
        mongo_db = mongo_client[MONGO_DB]
        fs       = gridfs.GridFS(mongo_db)

        # Indexes
        mongo_db.documents.create_index("prediction")
        mongo_db.documents.create_index("original_filename")
        mongo_db.documents.create_index("uploaded_at")
        mongo_db.documents.create_index("confidence")
        mongo_db.documents.create_index([("ocr_text", "text"), ("original_filename", "text")])

        print("[MongoDB] ✅ Connecté — base : bankdoc")
        return True
    except Exception as e:
        print(f"[MongoDB] ❌ Erreur : {e}")
        return False

MONGO_AVAILABLE = init_mongodb()


def save_document_to_mongo(file_path: str, original_filename: str, result: dict) -> str | None:
    if not MONGO_AVAILABLE:
        return None
    try:
        ext      = original_filename.rsplit(".", 1)[-1].lower() if "." in original_filename else "jpg"
        doc_id   = str(uuid.uuid4())
        category = result.get("prediction", "inconnu")

        cat_folder = os.path.join(DOCS_FOLDER, category)
        os.makedirs(cat_folder, exist_ok=True)
        stored_filename = f"{doc_id}.{ext}"
        stored_path     = os.path.join(cat_folder, stored_filename)
        shutil.copy2(file_path, stored_path)

        with open(file_path, "rb") as f_bin:
            gridfs_id = fs.put(
                f_bin,
                filename=stored_filename,
                content_type=f"image/{ext}" if ext != "pdf" else "application/pdf",
                metadata={"category": category, "doc_id": doc_id},
            )

        doc = {
            "doc_id":            doc_id,
            "original_filename": original_filename,
            "stored_filename":   stored_filename,
            "stored_path":       stored_path,
            "gridfs_id":         gridfs_id,
            "prediction":        category,
            "confidence":        result.get("confidence", 0),
            "path":              result.get("path", ""),
            "ocr_text":          result.get("ocr_text", ""),
            "all_scores":        result.get("all_scores", []),
            "qwen":              result.get("qwen", {}),
            "votes":             result.get("votes", {}),
            "agreement":         result.get("agreement", True),
            "time_ms":           result.get("time_ms", 0),
            "uploaded_at":       datetime.utcnow(),
            "file_size_bytes":   os.path.getsize(file_path),
            "corrected":         False,
        }
        insert_result = mongo_db.documents.insert_one(doc)
        mongo_id = str(insert_result.inserted_id)
        print(f"[MongoDB] ✅ Document sauvegardé : {mongo_id} → {category}/{stored_filename}")
        return mongo_id

    except Exception as e:
        print(f"[MongoDB] ❌ Erreur sauvegarde : {e}")
        return None


# ─────────────────────────────────────────────
# OLLAMA / QWEN — UTILITAIRES
# ─────────────────────────────────────────────

def check_ollama_available():
    global USE_OLLAMA, USE_QWEN
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code != 200:
            USE_OLLAMA = USE_QWEN = False
            return False
        models = response.json().get("models", [])
        names  = [m.get("name", "") for m in models]
        USE_OLLAMA = any(OLLAMA_MODEL in n for n in names)
        USE_QWEN   = any(QWEN_MODEL   in n for n in names)
        print(f"[Ollama] llama3.2 : {'✅' if USE_OLLAMA else '❌'}  |  "
              f"qwen2.5vl : {'✅' if USE_QWEN else '❌'}")
        return True
    except Exception as e:
        print(f"[Ollama] ❌ Non disponible : {e}")
        USE_OLLAMA = USE_QWEN = False
        return False


def call_ollama_text(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str | None:
    if not USE_OLLAMA:
        return None
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                requests.post, OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 256}},
                timeout=timeout,
            )
            resp = fut.result(timeout=timeout)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
    except FuturesTimeoutError:
        print(f"[llama3.2] Timeout ({timeout}s)")
    except Exception as e:
        print(f"[llama3.2] Erreur : {e}")
    return None


def image_to_base64(img_path: str, max_size: int = 1024) -> str:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_qwen_vision(img_path: str, ocr_text: str, doc_classes: list[str],
                     timeout: int = QWEN_TIMEOUT) -> dict:
    if not USE_QWEN:
        return {"class": None, "confidence": 0.0, "reasoning": "Qwen non disponible"}

    classes_str = "\n".join(f"- {c}" for c in doc_classes)
    prompt = f"""Tu es un expert en classification de documents bancaires marocains.
Analyse attentivement cette image de document et le texte OCR fourni.

CATÉGORIES DISPONIBLES :
{classes_str}

TEXTE OCR EXTRAIT :
{ocr_text[:800]}

INSTRUCTIONS :
1. Regarde la mise en page, les tampons, les logos et le contenu visuel
2. Identifie le type de document bancaire
3. Réponds UNIQUEMENT avec ce JSON (sans markdown, sans backticks) :
{{"class": "nom_exact_de_la_categorie", "confidence": 0.95, "reasoning": "explication courte"}}

Si le document ne correspond à aucune catégorie, utilise "inconnu"."""

    img_b64 = image_to_base64(img_path)
    payload = {
        "model": QWEN_MODEL, "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    }

    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut  = ex.submit(requests.post, OLLAMA_CHAT, json=payload, timeout=timeout)
            resp = fut.result(timeout=timeout)

        if resp.status_code != 200:
            return {"class": None, "confidence": 0.0, "reasoning": f"HTTP {resp.status_code}"}

        raw = resp.json().get("message", {}).get("content", "").strip()
        return _parse_json_response(raw, doc_classes)

    except FuturesTimeoutError:
        return {"class": None, "confidence": 0.0, "reasoning": f"Timeout ({timeout}s)"}
    except Exception as e:
        return {"class": None, "confidence": 0.0, "reasoning": str(e)}


def _parse_json_response(raw: str, valid_classes: list[str]) -> dict:
    import json
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"class"[^{}]*\}', raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}

    cls        = data.get("class", "").strip().lower().replace(" ", "_")
    confidence = float(data.get("confidence", 0.5))
    reasoning  = data.get("reasoning", "")

    cls_valid = next(
        (c for c in valid_classes if c.lower() == cls or cls in c.lower()), None
    )
    if not cls_valid:
        for c in valid_classes:
            if c.lower() in raw.lower():
                cls_valid = c
                break

    return {
        "class":      cls_valid,
        "confidence": min(max(confidence, 0.0), 1.0),
        "reasoning":  reasoning or raw[:200],
    }


# ─────────────────────────────────────────────
# ARCHITECTURE DU MODÈLE
# ─────────────────────────────────────────────
class MultimodalBankClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vision_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)
        visual_dim = self.vision_model.num_features
        self.text_model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        text_dim = self.text_model.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + text_dim, 512),
            nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, input_ids, attention_mask):
        v = self.vision_model(image)
        t = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.fusion(torch.cat([v, t], dim=1))


# ─────────────────────────────────────────────
# CHARGEMENT MODÈLE
# ─────────────────────────────────────────────
print(f"\n[INIT] Device : {DEVICE}")
print("[INIT] Chargement du modèle...")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
le          = checkpoint["label_encoder"]
num_classes = checkpoint["num_classes"]
img_size    = checkpoint["img_size"]
max_text    = checkpoint["max_text_len"]
val_acc     = checkpoint.get("val_acc", 0.0)

_thr_raw = checkpoint.get("confidence_threshold", 70.0)
CONFIDENCE_THRESHOLD = _thr_raw / 100.0 if _thr_raw > 1.0 else _thr_raw

DOC_CLASSES = list(le.classes_) + ["carte_identite"]

model = MultimodalBankClassifier(num_classes=num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()
print(f"[INIT] Classes : {DOC_CLASSES}")
print(f"[INIT] Val acc : {val_acc:.4f}  |  Seuil : {CONFIDENCE_THRESHOLD:.0%}")

tokenizer  = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
ocr_latin  = easyocr.Reader(["fr", "en"], gpu=torch.cuda.is_available(), verbose=False)
ocr_arabic = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False)
print("[INIT] OCR prêt")

img_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

check_ollama_available()

print("\n" + "=" * 60)
print(f"  MongoDB : {'CONNECTÉ' if MONGO_AVAILABLE else 'INACTIF'}")
print(f"  Ollama llama3.2 : {'ACTIF' if USE_OLLAMA else 'INACTIF'}")
print(f"  Qwen2.5-VL      : {'ACTIF' if USE_QWEN   else 'INACTIF'}")
print("  http://localhost:5000")
print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# FONCTIONS PIPELINE
# ─────────────────────────────────────────────

def extract_text_ocr(img_path: str) -> str:
    """
    Extrait le texte OCR COMPLET de l image.
    La troncature a max_text se fait UNIQUEMENT dans classify_with_model()
    pour le tokenizer BERT. Le texte complet est conserve pour le LLM.
    """
    text_latin = " ".join(ocr_latin.readtext(img_path, detail=0, paragraph=True))
    if len(text_latin.strip()) < 30:
        text_arabic = " ".join(ocr_arabic.readtext(img_path, detail=0, paragraph=True))
        text = (text_latin + " " + text_arabic).strip()
    else:
        text = text_latin
    return text or "[NO_TEXT]"  # PAS de [:max_text] ici — texte complet pour le LLM


def classify_with_model(img_path: str, text: str):
    image      = Image.open(img_path).convert("RGB")
    img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)
    tok        = tokenizer(text, max_length=max_text,
                           padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(img_tensor, tok["input_ids"].to(DEVICE), tok["attention_mask"].to(DEVICE))
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_class = le.classes_[pred_idx]
    confidence = float(probs[pred_idx])
    all_scores = sorted(
        [{"class": le.classes_[i], "score": float(probs[i])} for i in range(len(le.classes_))],
        key=lambda x: x["score"], reverse=True,
    )
    return pred_class, confidence, all_scores


def agent_nettoyeur_rapide(raw_text: str) -> str:
    cleaned = re.sub(r"[^\w\s\-\.,;:éèêëàâäôöùûüçæœ]", " ", raw_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 50 and cleaned.count(" ") > 5:
        return cleaned
    if USE_OLLAMA:
        result = call_ollama_text(f"Nettoie ce texte OCR: {raw_text}\nTexte nettoyé:", timeout=5)
        if result and len(result) > 10:
            return result
    return cleaned


def agent_keywords(text: str) -> dict:
    text_lower = text.lower()
    scores, found = {}, {}
    for cls, keywords in KEYWORD_RULES.items():
        hits = [kw for kw in keywords if kw.lower() in text_lower]
        if hits:
            scores[cls] = len(hits)
            found[cls]  = hits

    if re.search(r"n[ée] le \d{1,2}[./-]\d{1,2}[./-]\d{4}", text_lower):
        scores["carte_identite"] = scores.get("carte_identite", 0) + 3
        found.setdefault("carte_identite", []).append("date_naissance")

    if not scores:
        return {"class": None, "keywords_found": {}, "score": 0}
    best = max(scores, key=scores.get)
    return {"class": best, "keywords_found": found, "score": scores[best]}


def agent_llm_texte(text: str, model_prediction: str) -> str:
    text_lower = text.lower()
    if any(w in text_lower for w in ["carte nationale", "cni", "cin"]):
        if re.search(r"n[ée] le \d{1,2}[./-]\d{1,2}[./-]\d{4}", text_lower):
            return "carte_identite"
    if any(w in text_lower for w in ["assurance", "police", "prime"]):
        if any(w in text_lower for w in ["contrat", "assuré"]):
            return "assurance"
    if any(w in text_lower for w in ["iban", "bic", "rib"]):
        return "rib"
    if any(w in text_lower for w in ["chèque", "cheque", "à l'ordre de"]):
        return "cheque"

    if USE_OLLAMA:
        prompt = (f"Catégories: {', '.join(DOC_CLASSES)}\n"
                  f"Prédiction: {model_prediction}\n"
                  f"Texte: {text[:500]}\n"
                  f"Réponds UNIQUEMENT par le nom de la catégorie:")
        result = call_ollama_text(prompt, timeout=5)
        if result:
            for cls in DOC_CLASSES:
                if cls.lower() in result.lower():
                    return cls
    return model_prediction


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

UPLOAD_TEMP = "uploads_temp"
os.makedirs(UPLOAD_TEMP, exist_ok=True)


# ── Serve static HTML pages ───────────────────────────────────────────────

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/library")
def library_page():
    return app.send_static_file("library.html")

@app.route("/batch")
def batch_page():
    return app.send_static_file("batch.html")


# ── Classification ────────────────────────────────────────────────────────

@app.route("/classify", methods=["POST"])
def classify():
    t0 = time.time()
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400

    file     = request.files["file"]
    ext      = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    tmp_path = os.path.join(UPLOAD_TEMP, f"{uuid.uuid4().hex}.{ext}")
    file.save(tmp_path)
    steps = []
    raw_text = ""

    try:
        # Étape 1 : OCR
        raw_text = extract_text_ocr(tmp_path)
        steps.append({"step": "OCR bilingue", "status": "ok",
                      "detail": raw_text[:300] + ("…" if len(raw_text) > 300 else "")})

        # Étape 2 : Modèle EfficientNet + mBERT
        pred_class, confidence, all_scores = classify_with_model(tmp_path, raw_text)
        steps.append({"step": "Modèle multimodal (EfficientNet+mBERT)", "status": "ok",
                      "detail": f"{pred_class} — confiance {confidence:.1%}"})

        if re.search(r"carte nationale|cni|cin", raw_text.lower()):
            if re.search(r"n[ée] le \d{1,2}[./-]\d{1,2}[./-]\d{4}", raw_text.lower()):
                if confidence < 0.8:
                    pred_class, confidence = "carte_identite", 0.85
                    steps.append({"step": "Règle métier CIN", "status": "warning",
                                  "detail": "Carte d'identité détectée — correction prioritaire"})

        # Étape 3 : Qwen2.5-VL
        qwen_result = {"class": None, "confidence": 0.0, "reasoning": "Non activé"}
        if USE_QWEN:
            steps.append({"step": "Qwen2.5-VL (analyse visuelle)", "status": "ok",
                          "detail": "Analyse en cours…"})
            qwen_result = call_qwen_vision(tmp_path, raw_text, DOC_CLASSES)
            status_qwen = "ok" if qwen_result["class"] else "warning"
            steps[-1] = {
                "step": "Qwen2.5-VL (analyse visuelle)", "status": status_qwen,
                "detail": (
                    f"Classe : {qwen_result['class'] or 'non déterminé'}  |  "
                    f"Confiance : {qwen_result['confidence']:.0%}  |  "
                    f"{qwen_result['reasoning'][:120]}"
                ),
            }
        else:
            steps.append({"step": "Qwen2.5-VL", "status": "warning",
                          "detail": "Modèle non disponible (ollama list)"})

        qwen_agrees = (
            qwen_result["class"] is not None
            and qwen_result["class"].lower() == pred_class.lower()
        )

        # Résolution finale
        final_prediction = pred_class
        final_confidence = confidence
        final_scores     = all_scores
        path             = "direct"
        votes            = {}
        agreement        = True

        if confidence >= CONFIDENCE_THRESHOLD and qwen_agrees:
            steps.append({
                "step": f"Validation croisée (seuil ≥ {CONFIDENCE_THRESHOLD:.0%} + Qwen ✓)",
                "status": "ok",
                "detail": f"Les deux modèles s'accordent sur « {pred_class} »",
            })
            path = "direct_qwen_validated"

        elif confidence >= CONFIDENCE_THRESHOLD and not qwen_agrees and qwen_result["class"]:
            steps.append({
                "step": "Désaccord EfficientNet ↔ Qwen", "status": "warning",
                "detail": (
                    f"EfficientNet → {pred_class} ({confidence:.0%})  |  "
                    f"Qwen → {qwen_result['class']} ({qwen_result['confidence']:.0%})"
                ),
            })
            if confidence >= 0.85:
                final_prediction = pred_class
                steps.append({"step": "Décision", "status": "ok",
                               "detail": f"EfficientNet prioritaire (confiance {confidence:.0%} > 85%)"})
            elif qwen_result["confidence"] >= 0.80:
                final_prediction = qwen_result["class"]
                final_confidence = qwen_result["confidence"]
                steps.append({"step": "Décision", "status": "ok",
                               "detail": f"Qwen prioritaire (confiance {qwen_result['confidence']:.0%})"})
            else:
                steps.append({"step": "Décision", "status": "warning",
                               "detail": "Désaccord non résolu — révision manuelle recommandée"})
            path = "qwen_disagreement"

        else:
            steps.append({"step": f"Confiance faible (< {CONFIDENCE_THRESHOLD:.0%})",
                          "status": "warning",
                          "detail": f"{confidence:.1%} — agents complémentaires activés"})

            clean_text = agent_nettoyeur_rapide(raw_text)
            steps.append({"step": "Agent Nettoyeur", "status": "ok",
                          "detail": "Texte nettoyé" if clean_text != raw_text else "Nettoyage non nécessaire"})

            pred2, conf2, scores2 = classify_with_model(tmp_path, clean_text)
            final_scores = scores2
            steps.append({"step": "Reclassification (texte propre)", "status": "ok",
                          "detail": f"{pred2} — {conf2:.1%}"})

            kw = agent_keywords(clean_text)
            steps.append({"step": "Agent Mots-clés", "status": "ok" if kw["class"] else "info",
                          "detail": f"Classe : {kw['class'] or 'aucune'}  |  Score : {kw['score']}"})

            llm_class = agent_llm_texte(clean_text, pred2)
            steps.append({"step": "Agent LLM texte (llama3.2)", "status": "ok",
                          "detail": f"→ {llm_class}" + (" (correction)" if llm_class != pred2 else "")})

            vote_list = [pred2, llm_class]
            if kw["class"]:
                vote_list.append(kw["class"])
            if qwen_result["class"]:
                vote_list.append(qwen_result["class"])
                vote_list.append(qwen_result["class"])

            vote_counts  = Counter(vote_list)
            final_prediction, top_votes = vote_counts.most_common(1)[0]
            agreement    = top_votes >= 2
            final_confidence = max(conf2, qwen_result.get("confidence", 0))
            votes = {"model": pred2, "keywords": kw["class"], "llm": llm_class, "qwen": qwen_result["class"]}
            path  = "agents"

            steps.append({
                "step": "Vote final (modèle + keywords + llama + Qwen×2)",
                "status": "ok" if agreement else "warning",
                "detail": (
                    f"Résultat : {final_prediction}  |  "
                    f"{top_votes}/{len(vote_list)} votes  |  "
                    f"{'Accord ✓' if agreement else 'Désaccord ⚠ — révision recommandée'}"
                ),
            })

        time_ms = int((time.time() - t0) * 1000)

        result_data = {
            "prediction": final_prediction,
            "confidence": round(final_confidence * 100, 2),
            "all_scores": final_scores,
            "path":       path,
            "steps":      steps,
            "qwen":       qwen_result,
            "votes":      votes,
            "agreement":  agreement,
            "time_ms":    time_ms,
            "ocr_text":   raw_text,
        }

        # Sauvegarde MongoDB
        mongo_id = save_document_to_mongo(
            tmp_path,
            file.filename,
            {**result_data, "ocr_text": raw_text},
        )
        if mongo_id:
            result_data["mongo_id"] = mongo_id
            steps.append({
                "step": "Sauvegarde MongoDB", "status": "ok",
                "detail": f"ID : {mongo_id} → catégorie : {final_prediction}",
            })
        else:
            steps.append({
                "step": "Sauvegarde MongoDB", "status": "warning",
                "detail": "MongoDB non disponible — document non sauvegardé",
            })

        return jsonify(result_data)

    except Exception as e:
        import traceback
        print(f"[ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e), "prediction": "erreur", "confidence": 0}), 500

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────
# ROUTES MONGODB — DOCUMENTS
# ─────────────────────────────────────────────

def _serialize_doc(doc: dict) -> dict:
    """Sérialise un document MongoDB en JSON-safe dict."""
    doc = dict(doc)
    doc["_id"]       = str(doc["_id"])
    doc["gridfs_id"] = str(doc.get("gridfs_id", ""))
    if "uploaded_at" in doc and doc["uploaded_at"]:
        doc["uploaded_at"] = doc["uploaded_at"].isoformat() + "Z"
    if "corrected_at" in doc and doc.get("corrected_at"):
        doc["corrected_at"] = doc["corrected_at"].isoformat() + "Z"
    return doc


@app.route("/documents", methods=["GET"])
def list_documents():
    """
    Liste les documents avec filtres, tri et pagination.
    Query params :
        category    – filtre par catégorie exacte
        q           – recherche texte libre (nom fichier, OCR, catégorie)
        page        – numéro de page (défaut 1)
        per_page    – docs par page (défaut 24, max 100)
        sort        – date_desc | date_asc | conf_desc | conf_asc | name_asc
        conf_min    – confiance minimum 0-100 (défaut 0)
    """
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible", "documents": [], "total": 0}), 200

    category = request.args.get("category", "")
    q        = request.args.get("q", "").strip()
    page     = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 24)), 100)
    skip     = (page - 1) * per_page
    sort     = request.args.get("sort", "date_desc")
    conf_min = float(request.args.get("conf_min", 0))

    # Build query
    query = {}
    if category:
        query["prediction"] = category
    if conf_min > 0:
        query["confidence"] = {"$gte": conf_min}
    if q:
        query["$or"] = [
            {"original_filename": {"$regex": q, "$options": "i"}},
            {"ocr_text":          {"$regex": q, "$options": "i"}},
            {"prediction":        {"$regex": q, "$options": "i"}},
        ]

    # Sort mapping
    sort_map = {
        "date_desc": [("uploaded_at", DESCENDING)],
        "date_asc":  [("uploaded_at", ASCENDING)],
        "conf_desc": [("confidence",  DESCENDING)],
        "conf_asc":  [("confidence",  ASCENDING)],
        "name_asc":  [("original_filename", ASCENDING)],
    }
    mongo_sort = sort_map.get(sort, [("uploaded_at", DESCENDING)])

    total = mongo_db.documents.count_documents(query)
    docs  = list(
        mongo_db.documents.find(query, {"ocr_text": 0})
        .sort(mongo_sort)
        .skip(skip)
        .limit(per_page)
    )

    return jsonify({
        "documents": [_serialize_doc(d) for d in docs],
        "total":     total,
        "page":      page,
        "per_page":  per_page,
        "pages":     max((total + per_page - 1) // per_page, 1),
    })


@app.route("/documents/stats", methods=["GET"])
def documents_stats():
    """Statistiques agrégées par catégorie."""
    if not MONGO_AVAILABLE:
        return jsonify({"stats": [], "total": 0, "mongodb": False})

    pipeline = [
        {"$group": {
            "_id":      "$prediction",
            "count":    {"$sum": 1},
            "avg_conf": {"$avg": "$confidence"},
            "last_at":  {"$max": "$uploaded_at"},
        }},
        {"$sort": {"count": -1}},
    ]
    stats = list(mongo_db.documents.aggregate(pipeline))
    total = mongo_db.documents.count_documents({})

    return jsonify({
        "stats": [
            {
                "category":       s["_id"],
                "count":          s["count"],
                "avg_confidence": round(s["avg_conf"] or 0, 1),
                "last_upload":    s["last_at"].isoformat() + "Z" if s["last_at"] else None,
            }
            for s in stats
        ],
        "total":   total,
        "mongodb": True,
    })


@app.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id: str):
    """Détail complet d'un document (incluant ocr_text et all_scores)."""
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404
        return jsonify(_serialize_doc(doc))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/documents/<doc_id>/correct", methods=["PATCH"])
def correct_document(doc_id: str):
    """
    Corrige la catégorie prédite d'un document.
    Body JSON : {"prediction": "nouvelle_categorie"}
    Déplace aussi le fichier physique dans le bon sous-dossier.
    """
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        data = request.json or {}
        new_prediction = data.get("prediction", "").strip()
        if not new_prediction:
            return jsonify({"error": "Catégorie manquante"}), 400

        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        old_prediction = doc.get("prediction", "inconnu")

        # Déplacer le fichier physique si nécessaire
        old_stored_path = doc.get("stored_path", "")
        new_stored_path = old_stored_path

        if old_stored_path and os.path.exists(old_stored_path) and old_prediction != new_prediction:
            stored_filename = doc.get("stored_filename", os.path.basename(old_stored_path))
            new_cat_folder  = os.path.join(DOCS_FOLDER, new_prediction)
            os.makedirs(new_cat_folder, exist_ok=True)
            new_stored_path = os.path.join(new_cat_folder, stored_filename)
            try:
                shutil.move(old_stored_path, new_stored_path)
                print(f"[Correct] Fichier déplacé : {old_stored_path} → {new_stored_path}")
            except Exception as e:
                print(f"[Correct] Avertissement déplacement fichier : {e}")
                new_stored_path = old_stored_path  # garder l'ancien chemin si échec

        # Mise à jour MongoDB
        update_fields = {
            "prediction":          new_prediction,
            "corrected":           True,
            "original_prediction": old_prediction,
            "corrected_at":        datetime.utcnow(),
        }
        if new_stored_path != old_stored_path:
            update_fields["stored_path"] = new_stored_path

        mongo_db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": update_fields}
        )

        print(f"[Correct] {doc_id} : {old_prediction} → {new_prediction}")
        return jsonify({
            "success":  True,
            "old":      old_prediction,
            "new":      new_prediction,
            "new_path": new_stored_path,
        })

    except Exception as e:
        import traceback
        print(f"[Correct ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 400


@app.route("/documents/<doc_id>/download", methods=["GET"])
def download_document(doc_id: str):
    """Télécharge le fichier depuis le disque ou GridFS."""
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        # Fichier physique en priorité
        stored_path = doc.get("stored_path", "")
        if stored_path and os.path.exists(stored_path):
            return send_file(
                stored_path,
                as_attachment=True,
                download_name=doc["original_filename"],
            )

        # Fallback GridFS
        gridfs_id = doc.get("gridfs_id")
        if gridfs_id:
            grid_file = fs.get(gridfs_id)
            return send_file(
                io.BytesIO(grid_file.read()),
                as_attachment=True,
                download_name=doc["original_filename"],
                mimetype=grid_file.content_type or "application/octet-stream",
            )

        return jsonify({"error": "Fichier physique introuvable"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id: str):
    """Supprime un document (MongoDB + fichier physique + GridFS)."""
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        # Supprimer fichier physique
        stored_path = doc.get("stored_path", "")
        if stored_path and os.path.exists(stored_path):
            os.remove(stored_path)

        # Supprimer GridFS
        gridfs_id = doc.get("gridfs_id")
        if gridfs_id:
            try:
                fs.delete(gridfs_id)
            except Exception:
                pass

        # Supprimer document MongoDB
        mongo_db.documents.delete_one({"_id": ObjectId(doc_id)})

        return jsonify({"success": True, "deleted_id": doc_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
# EXTRACTION LLM DES MÉTADONNÉES PAR TYPE
# ─────────────────────────────────────────────

# Schémas de champs attendus par type de document
DOC_FIELD_SCHEMAS = {
    "carte_identite": {
        "label": "Carte Nationale d'Identité (CIN)",
        "fields": {
            "nom":              "Nom de famille (NOM)",
            "prenom":           "Prénom(s)",
            "cin_number":       "Numéro CIN (ex: AB123456)",
            "date_naissance":   "Date de naissance (JJ/MM/AAAA)",
            "lieu_naissance":   "Lieu / ville de naissance",
            "date_delivrance":  "Date de délivrance de la carte",
            "date_expiration":  "Date d'expiration / validité",
            "adresse":          "Adresse complète",
            "sexe":             "Sexe (M/F)",
        }
    },
    "rib": {
        "label": "Relevé d'Identité Bancaire (RIB)",
        "fields": {
            "titulaire":        "Nom complet du titulaire du compte",
            "iban":             "IBAN complet",
            "bic":              "Code BIC / SWIFT",
            "code_banque":      "Code banque (3-5 chiffres)",
            "code_guichet":     "Code guichet (3-5 chiffres)",
            "num_compte":       "Numéro de compte",
            "cle_rib":          "Clé RIB (2 chiffres)",
            "domiciliation":    "Domiciliation / nom de l'agence",
            "banque":           "Nom de la banque",
        }
    },
    "cheque": {
        "label": "Chèque bancaire",
        "fields": {
            "beneficiaire":       "Nom du bénéficiaire (à l'ordre de)",
            "montant_chiffres":   "Montant en chiffres (avec devise)",
            "montant_lettres":    "Montant en lettres",
            "date_emission":      "Date d'émission du chèque",
            "num_cheque":         "Numéro du chèque",
            "tire_sur":           "Banque tirée (tiré sur)",
            "emetteur":           "Nom de l'émetteur / signataire",
            "endossable":         "Endossable ou Non Endossable",
        }
    },
    "tableau_amortissement": {
        "label": "Tableau d'amortissement",
        "fields": {
            "montant_pret":     "Montant total du prêt",
            "taux_interet":     "Taux d'intérêt (annuel ou mensuel)",
            "duree":            "Durée totale du prêt (mois ou années)",
            "mensualite":       "Montant de la mensualité",
            "capital_initial":  "Capital emprunté initial",
            "date_debut":       "Date de début / première échéance",
            "nombre_echeances": "Nombre total d'échéances",
            "banque":           "Nom de la banque prêteuse",
        }
    },
    "acte_naissance": {
        "label": "Acte de naissance",
        "fields": {
            "nom":              "Nom de famille de l'enfant",
            "prenom":           "Prénom(s) de l'enfant",
            "date_naissance":   "Date de naissance (JJ/MM/AAAA)",
            "heure_naissance":  "Heure de naissance",
            "lieu_naissance":   "Commune / ville de naissance",
            "sexe":             "Sexe (masculin / féminin)",
            "pere":             "Nom complet du père",
            "mere":             "Nom complet de la mère",
            "num_acte":         "Numéro de l'acte",
            "date_acte":        "Date de l'établissement de l'acte",
        }
    },
    "acte_heredite": {
        "label": "Acte d'hérédité / notoriété",
        "fields": {
            "defunt":           "Nom complet du défunt(e)",
            "date_deces":       "Date de décès",
            "lieu_deces":       "Lieu de décès",
            "heritiers":        "Liste des héritiers",
            "notaire":          "Nom du notaire",
            "date_acte":        "Date de l'acte",
            "lieu_acte":        "Lieu de l'acte",
        }
    },
    "assurance": {
        "label": "Contrat / attestation d'assurance",
        "fields": {
            "assure":           "Nom de l'assuré(e)",
            "num_contrat":      "Numéro de contrat / police",
            "type_garantie":    "Type de garantie / objet du contrat",
            "date_effet":       "Date d'effet du contrat",
            "date_echeance":    "Date d'échéance / expiration",
            "prime":            "Montant de la prime / cotisation",
            "assureur":         "Nom de la compagnie d'assurance",
            "agence":           "Agence / courtier",
        }
    },
    "attestation_solde": {
        "label": "Attestation de solde bancaire",
        "fields": {
            "titulaire":        "Nom complet du titulaire",
            "num_compte":       "Numéro de compte",
            "solde":            "Solde du compte (avec devise)",
            "arrete_au":        "Date d'arrêté du solde",
            "banque":           "Nom de la banque",
            "agence":           "Agence bancaire",
            "type_compte":      "Type de compte (courant, épargne…)",
        }
    },
}


def _clean_llm_value(v):
    """Nettoie une valeur retournée par le LLM."""
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("", "null", "none", "n/a", "na", "—", "-", "inconnu", "non disponible", "nd"):
        return None
    return s


def _parse_llm_json(raw: str, field_keys: list) -> dict | None:
    """
    Parse la réponse JSON du LLM de façon robuste.
    Gère : blocs markdown, JSON avec texte autour, JSON mal formé.
    """
    import json
    if not raw:
        return None

    # Enlever les blocs markdown ```json ... ```
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw).strip()

    # Stratégie 1 : trouver le bloc JSON le plus GRAND (greedy, de la première { à la dernière })
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        try:
            data   = json.loads(candidate)
            result = {k: _clean_llm_value(data.get(k)) for k in field_keys}
            if any(v for v in result.values()):
                return result
        except json.JSONDecodeError:
            pass

    # Stratégie 2 : json.loads direct sur tout le texte nettoyé
    try:
        data   = json.loads(raw)
        result = {k: _clean_llm_value(data.get(k)) for k in field_keys}
        return result
    except Exception:
        pass

    # Stratégie 3 : extraction champ par champ avec regex (LLM trop bavard)
    result = {}
    for k in field_keys:
        # Cherche "nom_champ": "valeur" ou "nom_champ": null
        m = re.search(rf'"{re.escape(k)}"\s*:\s*(?:"([^"]*?)"|null|None)', raw, re.IGNORECASE)
        if m:
            result[k] = _clean_llm_value(m.group(1)) if m.group(1) else None
        else:
            result[k] = None
    if any(v for v in result.values()):
        return result

    return None


def _call_ollama_streaming(prompt: str, model: str, timeout_no_token: int = 30) -> str:
    """
    Appelle Ollama en mode streaming pour éviter le timeout HTTP global.
    Lit les tokens au fur et à mesure — pas de timeout sur la durée totale,
    seulement un timeout si aucun token n'arrive pendant `timeout_no_token` secondes.
    Compatible CPU lent (llama3.2 peut prendre 5-10 min sur CPU).
    """
    import json as _json
    payload = {
        "model":   model,
        "prompt":  prompt,
        "stream":  True,
        "options": {"temperature": 0.05, "num_predict": 600},
    }
    chunks = []
    try:
        with requests.post(
            OLLAMA_URL, json=payload,
            stream=True,
            timeout=(10, timeout_no_token),   # (connect_timeout, read_timeout entre tokens)
        ) as resp:
            if resp.status_code != 200:
                print(f"[Extract] Ollama HTTP {resp.status_code}")
                return ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    token = obj.get("response", "")
                    chunks.append(token)
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except requests.exceptions.ReadTimeout:
        print(f"[Extract] Streaming interrompu — pas de token depuis {timeout_no_token}s")
    except Exception as e:
        print(f"[Extract] Erreur streaming : {e}")
    return "".join(chunks)


def _call_qwen_streaming(prompt: str, img_b64: str, timeout_no_token: int = 60) -> str:
    """Idem pour Qwen2.5-VL en mode vision."""
    import json as _json
    payload = {
        "model":    QWEN_MODEL,
        "stream":   True,
        "options":  {"temperature": 0.05, "num_predict": 600},
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    }
    chunks = []
    try:
        with requests.post(
            OLLAMA_CHAT, json=payload,
            stream=True,
            timeout=(10, timeout_no_token),
        ) as resp:
            if resp.status_code != 200:
                print(f"[Extract Qwen] HTTP {resp.status_code}")
                return ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj   = _json.loads(line)
                    token = obj.get("message", {}).get("content", "")
                    chunks.append(token)
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except requests.exceptions.ReadTimeout:
        print(f"[Extract Qwen] Streaming interrompu — pas de token depuis {timeout_no_token}s")
    except Exception as e:
        print(f"[Extract Qwen] Erreur streaming : {e}")
    return "".join(chunks)


def extract_metadata_with_llm(ocr_text: str, doc_type: str, stored_path: str = "") -> dict:
    """
    Extrait les métadonnées via LLM en utilisant le streaming Ollama.
    Le streaming évite les timeouts HTTP : on lit token par token,
    sans délai global, même sur CPU lent.

    Stratégie :
      1. llama3.2 en streaming sur le texte OCR (prompt court et ciblé)
      2. Qwen2.5-VL en streaming sur l'image si llama échoue
    """
    import json

    schema = DOC_FIELD_SCHEMAS.get(doc_type)
    if not schema:
        return {"error": f"Type '{doc_type}' non supporté"}

    field_keys   = list(schema["fields"].keys())
    empty_result = {k: None for k in field_keys}

    # Prompt minimal et ciblé — moins de tokens = plus rapide sur CPU
    # On liste seulement les clés JSON, pas les descriptions verbeuses
    keys_inline = ", ".join(f'"{k}"' for k in field_keys)
    ocr_short   = ocr_text[:2000]   # 2000 chars suffisent

    prompt = (
        f"Texte OCR d'un document '{doc_type}' (marocain, peut être bruité) :\n"
        f"{ocr_short}\n\n"
        f"Extrais en JSON les champs : {keys_inline}\n"
        f"null si absent. Corrige les erreurs OCR. JSON uniquement :"
    )

    # ── Tentative 1 : llama3.2 streaming ──────────────────────────────────
    if USE_OLLAMA:
        print(f"[Extract] llama3.2 streaming pour {doc_type} ({len(ocr_short)} chars)...")
        t0  = time.time()
        raw = _call_ollama_streaming(prompt, OLLAMA_MODEL, timeout_no_token=30)
        print(f"[Extract] llama3.2 → {len(raw)} chars en {time.time()-t0:.0f}s : {raw[:150]}")

        if raw:
            result = _parse_llm_json(raw, field_keys)
            if result and any(v for v in result.values()):
                n = sum(1 for v in result.values() if v)
                print(f"[Extract LLM] ✅ llama3.2 → {n}/{len(field_keys)} champs")
                result["_source"] = "llama3.2"
                return result
            else:
                print(f"[Extract LLM] ⚠ llama3.2 répondu mais 0 champ — raw: {raw[:200]}")

    # ── Tentative 2 : Qwen vision streaming ───────────────────────────────
    if USE_QWEN and stored_path and os.path.exists(stored_path):
        prompt_qwen = (
            f"Document type: {doc_type}. Extract JSON fields: {keys_inline}. "
            f"null if absent. JSON only:"
        )
        try:
            print(f"[Extract] Qwen streaming pour {doc_type}...")
            img_b64 = image_to_base64(stored_path)
            t0  = time.time()
            raw = _call_qwen_streaming(prompt_qwen, img_b64, timeout_no_token=60)
            print(f"[Extract] Qwen → {len(raw)} chars en {time.time()-t0:.0f}s : {raw[:150]}")

            if raw:
                result = _parse_llm_json(raw, field_keys)
                if result and any(v for v in result.values()):
                    n = sum(1 for v in result.values() if v)
                    print(f"[Extract Qwen] ✅ Qwen → {n}/{len(field_keys)} champs")
                    result["_source"] = "qwen"
                    return result
                else:
                    print(f"[Extract Qwen] ⚠ Qwen répondu mais 0 champ — raw: {raw[:200]}")
        except Exception as e:
            print(f"[Extract Qwen] ❌ {e}")

    print(f"[Extract LLM] ❌ ÉCHEC — retourne null")
    return empty_result


@app.route("/documents/<doc_id>/extract", methods=["POST"])
def extract_document_metadata(doc_id: str):
    """
    Extrait les métadonnées structurées d'un document via LLM (llama3.2 ou Qwen).
    Lance l'extraction et sauvegarde le résultat dans MongoDB.
    Retourne : {"fields": {...}, "source": "llama3.2"|"qwen"|"none", "doc_type": "..."}
    """
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        doc_type    = doc.get("prediction", "inconnu")
        ocr_text    = doc.get("ocr_text", "")
        stored_path = doc.get("stored_path", "")
        force = request.json.get("force", False) if request.json else False

        # ── Regénérer l OCR si tronqué (anciens docs avec max_text=128) ──
        OCR_MIN_LENGTH = 200
        if stored_path and os.path.exists(stored_path) and len(ocr_text) < OCR_MIN_LENGTH:
            print(f"[Extract] OCR trop court ({len(ocr_text)} chars) — re-OCR depuis {stored_path}")
            try:
                fresh_ocr = extract_text_ocr(stored_path)
                if len(fresh_ocr) > len(ocr_text):
                    ocr_text = fresh_ocr
                    # Mettre à jour l OCR dans MongoDB
                    mongo_db.documents.update_one(
                        {"_id": ObjectId(doc_id)},
                        {"$set": {"ocr_text": ocr_text}}
                    )
                    print(f"[Extract] OCR mis à jour : {len(ocr_text)} chars")
            except Exception as e:
                print(f"[Extract] Avertissement re-OCR : {e}")

        # ── Cache : si déjà extrait et non forcé, retourner le résultat ──
        if not force and doc.get("extracted_fields"):
            cached = doc["extracted_fields"]
            if any(v for v in cached.values()):
                return jsonify({
                    "fields":    cached,
                    "source":    doc.get("extraction_source", "cache"),
                    "doc_type":  doc_type,
                    "cached":    True,
                })

        if not ocr_text or ocr_text == "[NO_TEXT]":
            return jsonify({"error": "Aucun texte OCR disponible pour ce document"}), 400

        print(f"[Extract] Texte OCR disponible : {len(ocr_text)} chars")

        if doc_type not in DOC_FIELD_SCHEMAS:
            return jsonify({
                "fields":   {},
                "source":   "none",
                "doc_type": doc_type,
                "error":    f"Type '{doc_type}' non supporté pour l'extraction",
            })

        # Lancer l'extraction
        t0     = time.time()
        fields = extract_metadata_with_llm(ocr_text, doc_type, stored_path)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Récupérer la source réelle depuis le champ _source injecté par la fonction
        source = fields.pop("_source", "none")
        if source == "none" and any(v for v in fields.values() if v):
            source = "llama3.2"  # fallback

        # Sauvegarder dans MongoDB
        mongo_db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {
                "extracted_fields":   fields,
                "extraction_source":  source,
                "extracted_at":       datetime.utcnow(),
            }}
        )

        print(f"[Extract] {doc_id} ({doc_type}) → {source} → {elapsed_ms}ms")
        return jsonify({
            "fields":      fields,
            "source":      source,
            "doc_type":    doc_type,
            "elapsed_ms":  elapsed_ms,
            "cached":      False,
        })

    except Exception as e:
        import traceback
        print(f"[Extract ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500



# ─────────────────────────────────────────────
# GESTION DES TYPES PERSONNALISÉS
# ─────────────────────────────────────────────

TRAINING_THRESHOLD = 50   # Nombre d images pour déclencher l alerte de ré-entraînement

def _init_custom_types_collection():
    """Crée les index sur la collection doc_types si MongoDB est disponible."""
    if not MONGO_AVAILABLE:
        return
    try:
        mongo_db.doc_types.create_index("name", unique=True)
        mongo_db.doc_types.create_index("is_custom")
        mongo_db.doc_types.create_index("count")
        # Insérer les types natifs s ils n existent pas encore
        native_types = [
            "carte_identite","rib","cheque","tableau_amortissement",
            "acte_naissance","acte_heredite","assurance","attestation_solde",
        ]
        for t in native_types:
            mongo_db.doc_types.update_one(
                {"name": t},
                {"$setOnInsert": {
                    "name": t, "label": t.replace("_", " ").title(),
                    "is_custom": False, "count": 0,
                    "ready_for_training": False,
                    "created_at": datetime.utcnow(),
                }},
                upsert=True,
            )
    except Exception as e:
        print(f"[DocTypes] Init erreur : {e}")

_init_custom_types_collection()


@app.route("/types", methods=["GET"])
def list_types():
    """Liste tous les types de documents (natifs + personnalisés)."""
    if not MONGO_AVAILABLE:
        # Retourner les types natifs par défaut
        native = ["carte_identite","rib","cheque","tableau_amortissement",
                  "acte_naissance","acte_heredite","assurance","attestation_solde"]
        return jsonify({"types": [{"name": t, "label": t.replace("_"," ").title(),
                                   "is_custom": False, "count": 0} for t in native]})
    try:
        types = list(mongo_db.doc_types.find({}, {"_id": 0}).sort("name", 1))
        # Enrichir avec le count réel depuis la collection documents
        for t in types:
            real_count = mongo_db.documents.count_documents({"prediction": t["name"]})
            t["count"] = real_count
            t["ready_for_training"] = real_count >= TRAINING_THRESHOLD
            if "created_at" in t and t["created_at"]:
                t["created_at"] = t["created_at"].isoformat() + "Z"
        return jsonify({"types": types, "training_threshold": TRAINING_THRESHOLD})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/types", methods=["POST"])
def create_type():
    """
    Crée un nouveau type de document personnalisé.
    Body JSON : {"name": "certificat_vie", "label": "Certificat de vie"}
    """
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        data  = request.json or {}
        name  = data.get("name", "").strip().lower().replace(" ", "_").replace("-", "_")
        label = data.get("label", "").strip() or name.replace("_", " ").title()

        if not name:
            return jsonify({"error": "Le nom du type est requis"}), 400
        if not re.match(r"^[a-z][a-z0-9_]{1,49}$", name):
            return jsonify({"error": "Nom invalide (lettres minuscules, chiffres, underscores, 2-50 chars)"}), 400

        # Vérifier si existe déjà
        existing = mongo_db.doc_types.find_one({"name": name})
        if existing:
            return jsonify({"error": f"Le type '{name}' existe déjà"}), 409

        doc_type = {
            "name":               name,
            "label":              label,
            "is_custom":          True,
            "count":              0,
            "ready_for_training": False,
            "created_at":         datetime.utcnow(),
            "description":        data.get("description", ""),
        }
        mongo_db.doc_types.insert_one(doc_type)

        # Créer le dossier physique pour les images de ce type
        type_folder = os.path.join(DOCS_FOLDER, name)
        os.makedirs(type_folder, exist_ok=True)

        print(f"[DocTypes] ✅ Nouveau type créé : {name} ({label})")
        return jsonify({
            "success": True,
            "type": {
                "name": name, "label": label,
                "is_custom": True, "count": 0,
                "ready_for_training": False,
                "folder": os.path.abspath(type_folder),
            }
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/types/<type_name>", methods=["DELETE"])
def delete_type(type_name: str):
    """Supprime un type personnalisé (uniquement si is_custom=True et count=0)."""
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        doc_type = mongo_db.doc_types.find_one({"name": type_name})
        if not doc_type:
            return jsonify({"error": "Type introuvable"}), 404
        if not doc_type.get("is_custom"):
            return jsonify({"error": "Les types natifs ne peuvent pas être supprimés"}), 403
        count = mongo_db.documents.count_documents({"prediction": type_name})
        if count > 0:
            return jsonify({"error": f"Ce type contient {count} document(s) — videz-le d abord"}), 409
        mongo_db.doc_types.delete_one({"name": type_name})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/documents/<doc_id>/classify_manual", methods=["POST"])
def classify_manual(doc_id: str):
    """
    Classifie manuellement un document.
    Body JSON :
      {"type_name": "certificat_vie", "create_new": false}
      ou {"type_name": "nouveau_type", "label": "Nouveau Type", "create_new": true}

    Sauvegarde le document dans MongoDB avec le nouveau type,
    déplace le fichier physique dans le bon dossier,
    et met à jour le compteur du type.
    """
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    try:
        data       = request.json or {}
        type_name  = data.get("type_name", "").strip().lower().replace(" ", "_")
        create_new = data.get("create_new", False)
        label      = data.get("label", type_name.replace("_", " ").title())

        if not type_name:
            return jsonify({"error": "type_name requis"}), 400

        # Créer le type si demandé
        if create_new:
            existing = mongo_db.doc_types.find_one({"name": type_name})
            if not existing:
                if not re.match(r"^[a-z][a-z0-9_]{1,49}$", type_name):
                    return jsonify({"error": "Nom de type invalide"}), 400
                mongo_db.doc_types.insert_one({
                    "name": type_name, "label": label,
                    "is_custom": True, "count": 0,
                    "ready_for_training": False,
                    "created_at": datetime.utcnow(),
                    "description": data.get("description", ""),
                })
                os.makedirs(os.path.join(DOCS_FOLDER, type_name), exist_ok=True)
                print(f"[Manual] Nouveau type créé : {type_name}")
        else:
            # Vérifier que le type existe
            if not mongo_db.doc_types.find_one({"name": type_name}):
                return jsonify({"error": f"Type '{type_name}' introuvable — utilisez create_new=true pour le créer"}), 404

        # Récupérer le document
        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404

        old_type = doc.get("prediction", "inconnu")

        # Déplacer le fichier physique
        old_path = doc.get("stored_path", "")
        new_path = old_path
        if old_path and os.path.exists(old_path) and old_type != type_name:
            stored_filename = doc.get("stored_filename", os.path.basename(old_path))
            new_folder      = os.path.join(DOCS_FOLDER, type_name)
            os.makedirs(new_folder, exist_ok=True)
            new_path = os.path.join(new_folder, stored_filename)
            try:
                shutil.move(old_path, new_path)
            except Exception as e:
                print(f"[Manual] Avertissement déplacement : {e}")
                new_path = old_path

        # Mettre à jour MongoDB
        update = {
            "prediction":            type_name,
            "manually_classified":   True,
            "manual_classifier":     data.get("user", "human"),
            "classified_at":         datetime.utcnow(),
            "original_prediction":   old_type,
            "original_confidence":   doc.get("confidence", 0),
        }
        if new_path != old_path:
            update["stored_path"] = new_path

        mongo_db.documents.update_one({"_id": ObjectId(doc_id)}, {"$set": update})

        # Vérifier le seuil de ré-entraînement
        new_count = mongo_db.documents.count_documents({"prediction": type_name})
        ready     = new_count >= TRAINING_THRESHOLD

        print(f"[Manual] {doc_id}: {old_type} → {type_name} | count={new_count}")
        return jsonify({
            "success":          True,
            "old_type":         old_type,
            "new_type":         type_name,
            "doc_count":        new_count,
            "ready_for_training": ready,
            "training_threshold": TRAINING_THRESHOLD,
        })

    except Exception as e:
        import traceback
        print(f"[Manual ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/types/stats", methods=["GET"])
def types_stats():
    """Statistiques sur les types custom : compteurs et alertes de ré-entraînement."""
    if not MONGO_AVAILABLE:
        return jsonify({"custom_types": [], "alerts": []})
    try:
        custom_types = list(mongo_db.doc_types.find(
            {"is_custom": True}, {"_id": 0}
        ))
        alerts = []
        for t in custom_types:
            count = mongo_db.documents.count_documents({"prediction": t["name"]})
            t["count"] = count
            t["ready_for_training"] = count >= TRAINING_THRESHOLD
            if "created_at" in t and t["created_at"]:
                t["created_at"] = t["created_at"].isoformat() + "Z"
            if count >= TRAINING_THRESHOLD:
                alerts.append({
                    "type":    t["name"],
                    "label":   t["label"],
                    "count":   count,
                    "message": f"✅ {t['label']} a {count} images — prêt pour le ré-entraînement !",
                })
        return jsonify({
            "custom_types": custom_types,
            "alerts":       alerts,
            "threshold":    TRAINING_THRESHOLD,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# ROUTES UTILITAIRES
# ─────────────────────────────────────────────

@app.route("/status", methods=["GET"])
def status():
    check_ollama_available()
    mongo_ok = False
    try:
        mongo_client.server_info()
        mongo_ok = True
    except Exception:
        pass

    return jsonify({
        "model_loaded":  True,
        "device":        str(DEVICE),
        "classes":       DOC_CLASSES,
        "val_acc":       round(val_acc * 100, 2),
        "threshold":     round(CONFIDENCE_THRESHOLD * 100, 1),
        "ollama":        USE_OLLAMA,
        "ollama_model":  OLLAMA_MODEL,
        "qwen":          USE_QWEN,
        "qwen_model":    QWEN_MODEL,
        "mongodb":       mongo_ok,
        "docs_folder":   os.path.abspath(DOCS_FOLDER),
    })


@app.route("/toggle_ollama", methods=["POST"])
def toggle_ollama():
    global USE_OLLAMA
    USE_OLLAMA = request.json.get("enabled", USE_OLLAMA)
    return jsonify({"ollama_enabled": USE_OLLAMA})


@app.route("/toggle_qwen", methods=["POST"])
def toggle_qwen():
    global USE_QWEN
    USE_QWEN = request.json.get("enabled", USE_QWEN)
    return jsonify({"qwen_enabled": USE_QWEN})


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)