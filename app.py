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

OLLAMA_TIMEOUT      = 60      # Augmenté de 10 à 60 secondes
OLLAMA_MAX_RETRIES  = 2
QWEN_TIMEOUT        = 180     # Augmenté de 120 à 180 secondes
USE_OLLAMA          = True
USE_QWEN            = True

# MongoDB config
MONGO_URI    = "mongodb://localhost:27017/"
MONGO_DB     = "bankdoc"
DOCS_FOLDER  = "stored_documents"

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
    text_latin = " ".join(ocr_latin.readtext(img_path, detail=0, paragraph=True))
    if len(text_latin.strip()) < 30:
        text_arabic = " ".join(ocr_arabic.readtext(img_path, detail=0, paragraph=True))
        text = (text_latin + " " + text_arabic).strip()
    else:
        text = text_latin
    return text[:max_text] or "[NO_TEXT]"


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
# SCHÉMAS ET EXTRACTION LLM CORRIGÉS
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


def extract_metadata_with_llm(ocr_text: str, doc_type: str, stored_path: str = "") -> dict:
    """
    Utilise llama3.2 pour extraire les métadonnées structurées d'un document
    """
    import json
    
    print(f"\n[Extract LLM] === DÉBUT EXTRACTION ===")
    print(f"[Extract LLM] Type: {doc_type}")
    print(f"[Extract LLM] OCR length: {len(ocr_text)}")
    
    schema = DOC_FIELD_SCHEMAS.get(doc_type)
    if not schema:
        print(f"[Extract LLM] ❌ Schéma non trouvé pour {doc_type}")
        return {"error": f"Type '{doc_type}' non supporté pour l'extraction"}
    
    field_keys = list(schema["fields"].keys())
    
    # RÉDUIRE LE TEXTE OCR POUR ACCÉLÉRER
    max_chars = 1500
    truncated_ocr = ocr_text[:max_chars]
    if len(ocr_text) > max_chars:
        truncated_ocr += "\n[...]\n" + ocr_text[-500:]
    
    # PROMPT SIMPLIFIÉ ET DIRECT
    prompt = f"""Extrais les informations de ce document {schema['label']}.

Texte OCR (peut contenir des erreurs):
\"\"\"
{truncated_ocr}
\"\"\"

Champs à extraire (réponds UNIQUEMENT en JSON, null si absent):
{', '.join(field_keys)}

Format JSON attendu:
{{{', '.join([f'"{k}": null' for k in field_keys])}}}"""

    print(f"[Extract LLM] Prompt length: {len(prompt)}")
    
    # Tentative avec llama3.2
    if USE_OLLAMA:
        try:
            print(f"[Extract LLM] 📤 Envoi à llama3.2...")
            
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.05,
                        "num_predict": 500,
                    },
                },
                timeout=OLLAMA_TIMEOUT
            )
            
            print(f"[Extract LLM] Response status: {response.status_code}")
            
            if response.status_code == 200:
                raw = response.json().get("response", "").strip()
                print(f"[Extract LLM] Réponse reçue ({len(raw)} chars)")
                print(f"[Extract LLM] Réponse (500 premiers chars):\n{raw[:500]}")
                
                # Nettoyer la réponse
                raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw)
                
                # Chercher un bloc JSON
                m = re.search(r'\{[\s\S]*\}', raw)
                if m:
                    json_str = m.group()
                    print(f"[Extract LLM] JSON trouvé")
                    
                    try:
                        data = json.loads(json_str)
                        result = {}
                        for k in field_keys:
                            val = data.get(k)
                            if val and str(val).strip() not in ("", "null", "N/A", "None", "—"):
                                result[k] = str(val).strip()
                            else:
                                result[k] = None
                        
                        filled = sum(1 for v in result.values() if v)
                        print(f"[Extract LLM] ✅ SUCCÈS: {filled}/{len(field_keys)} champs extraits")
                        return result
                        
                    except json.JSONDecodeError as e:
                        print(f"[Extract LLM] ❌ JSON invalide: {e}")
                        print(f"[Extract LLM] JSON string: {json_str[:200]}")
                else:
                    print(f"[Extract LLM] ❌ Aucun JSON trouvé")
            else:
                print(f"[Extract LLM] ❌ Erreur HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"[Extract LLM] ⏱️ TIMEOUT après {OLLAMA_TIMEOUT}s")
        except Exception as e:
            print(f"[Extract LLM] ❌ Exception: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[Extract LLM] ❌ Ollama désactivé")
    
    # Tentative avec Qwen si disponible
    if USE_QWEN and stored_path and os.path.exists(stored_path):
        print(f"[Extract LLM] Tentative avec Qwen...")
        try:
            prompt_qwen = f"""Extrais ces champs du document: {', '.join(field_keys)}.
Réponds UNIQUEMENT avec ce JSON (null si absent):
{{{', '.join([f'"{k}": null' for k in field_keys])}}}"""
            
            img_b64 = image_to_base64(stored_path)
            payload = {
                "model": QWEN_MODEL,
                "stream": False,
                "options": {"temperature": 0.05, "num_predict": 500},
                "messages": [{"role": "user", "content": prompt_qwen, "images": [img_b64]}],
            }
            
            response = requests.post(OLLAMA_CHAT, json=payload, timeout=QWEN_TIMEOUT)
            
            if response.status_code == 200:
                raw = response.json().get("message", {}).get("content", "").strip()
                raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw)
                m = re.search(r'\{[\s\S]*\}', raw)
                if m:
                    data = json.loads(m.group())
                    result = {}
                    for k in field_keys:
                        val = data.get(k)
                        if val and str(val).strip() not in ("", "null", "N/A", "None", "—"):
                            result[k] = str(val).strip()
                        else:
                            result[k] = None
                    filled = sum(1 for v in result.values() if v)
                    print(f"[Extract LLM] ✅ Qwen: {filled} champs extraits")
                    return result
        except Exception as e:
            print(f"[Extract LLM] ❌ Qwen erreur: {e}")
    
    print(f"[Extract LLM] ❌ ÉCHEC, retourne valeurs null")
    return {k: None for k in field_keys}


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
                new_stored_path = old_stored_path

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

        stored_path = doc.get("stored_path", "")
        if stored_path and os.path.exists(stored_path):
            return send_file(
                stored_path,
                as_attachment=True,
                download_name=doc["original_filename"],
            )

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

        stored_path = doc.get("stored_path", "")
        if stored_path and os.path.exists(stored_path):
            os.remove(stored_path)

        gridfs_id = doc.get("gridfs_id")
        if gridfs_id:
            try:
                fs.delete(gridfs_id)
            except Exception:
                pass

        mongo_db.documents.delete_one({"_id": ObjectId(doc_id)})

        return jsonify({"success": True, "deleted_id": doc_id})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ─────────────────────────────────────────────
# ROUTE D'EXTRACTION CORRIGÉE
# ─────────────────────────────────────────────

@app.route("/documents/<doc_id>/extract", methods=["POST"])
def extract_document_metadata(doc_id: str):
    """
    Extrait les métadonnées structurées d'un document via LLM
    """
    print(f"\n[Extract Route] === DÉBUT pour {doc_id} ===")
    
    if not MONGO_AVAILABLE:
        return jsonify({"error": "MongoDB non disponible"}), 503
    
    try:
        doc = mongo_db.documents.find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return jsonify({"error": "Document introuvable"}), 404
        
        doc_type = doc.get("prediction", "inconnu")
        ocr_text = doc.get("ocr_text", "")
        stored_path = doc.get("stored_path", "")
        
        print(f"[Extract Route] Type: {doc_type}")
        print(f"[Extract Route] OCR length: {len(ocr_text)}")
        
        # Vérifier si déjà extrait
        force = request.json.get("force", False) if request.json else False
        if not force and doc.get("extracted_fields"):
            print("[Extract Route] Utilisation du cache")
            return jsonify({
                "fields": doc["extracted_fields"],
                "source": doc.get("extraction_source", "cache"),
                "doc_type": doc_type,
                "cached": True,
            })
        
        if not ocr_text or ocr_text == "[NO_TEXT]":
            return jsonify({"error": "Aucun texte OCR disponible pour ce document"}), 400
        
        if doc_type not in DOC_FIELD_SCHEMAS:
            return jsonify({
                "fields": {},
                "source": "none",
                "doc_type": doc_type,
                "error": f"Type '{doc_type}' non supporté pour l'extraction",
            })
        
        # Lancer l'extraction
        t0 = time.time()
        fields = extract_metadata_with_llm(ocr_text, doc_type, stored_path)
        elapsed_ms = int((time.time() - t0) * 1000)
        
        # Déterminer la source
        filled = sum(1 for v in fields.values() if v)
        source = "llama3.2" if filled > 0 else "none"
        
        print(f"[Extract Route] Terminé en {elapsed_ms}ms, {filled} champs extraits")
        
        # Sauvegarder dans MongoDB
        mongo_db.documents.update_one(
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
            "elapsed_ms": elapsed_ms,
            "cached": False,
        })
        
    except Exception as e:
        import traceback
        print(f"[Extract Route] ERREUR: {traceback.format_exc()}")
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