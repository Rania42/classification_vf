"""
Microservice d'extraction structurée — port 5001
Indépendant du monolithe Flask principal (port 5000).

Endpoints :
  POST /extract          → extraction complète
  POST /extract/batch    → extraction batch (liste de documents)
  GET  /health           → statut du service
  GET  /schemas          → liste des schemas disponibles
  GET  /schemas/<type>   → schema d'un type spécifique
"""

import os
import re
import time
import json
import base64
import io
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Config inline (pas de dépendance sur config.py du monolithe) ───────────
OLLAMA_URL    = os.environ.get("OLLAMA_URL",   "http://localhost:11434/api/generate")
OLLAMA_CHAT   = os.environ.get("OLLAMA_CHAT",  "http://localhost:11434/api/chat")
GEMMA_MODEL   = os.environ.get("GEMMA_MODEL",  "gemma2:2b")
QWEN_MODEL    = os.environ.get("QWEN_MODEL",   "qwen2.5vl:3b")
SERVICE_PORT  = int(os.environ.get("EXTRACTION_PORT", 5001))
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", 120))

# ── Schemas des champs par type de document ────────────────────────────────
DOC_FIELD_SCHEMAS = {
    "rib": {
        "label": "Relevé d'Identité Bancaire",
        "fields": {
            "titulaire":     "Nom du titulaire",
            "iban":          "IBAN complet",
            "bic":           "Code BIC/SWIFT",
            "code_banque":   "Code banque",
            "code_guichet":  "Code guichet",
            "num_compte":    "Numéro de compte",
            "cle_rib":       "Clé RIB",
            "domiciliation": "Domiciliation/agence",
            "banque":        "Nom de la banque",
        }
    },
    "cheque": {
        "label": "Chèque bancaire",
        "fields": {
            "beneficiaire":     "Bénéficiaire",
            "montant_chiffres": "Montant en chiffres",
            "montant_lettres":  "Montant en lettres",
            "date_emission":    "Date d'émission",
            "num_cheque":       "Numéro du chèque",
            "tire_sur":         "Banque tirée",
            "emetteur":         "Émetteur/signataire",
            "endossable":       "Endossable (oui/non)",
        }
    },
    "tableau_amortissement": {
        "label": "Tableau d'amortissement",
        "fields": {
            "montant_pret":      "Montant du prêt",
            "taux_interet":      "Taux d'intérêt",
            "duree":             "Durée du prêt",
            "mensualite":        "Mensualité",
            "capital_initial":   "Capital initial",
            "date_debut":        "Date de début",
            "nombre_echeances":  "Nombre d'échéances",
            "banque":            "Banque prêteuse",
        }
    },
    "acte_naissance": {
        "label": "Acte de naissance",
        "fields": {
            "nom":            "Nom",
            "prenom":         "Prénom(s)",
            "date_naissance": "Date de naissance",
            "heure_naissance":"Heure de naissance",
            "lieu_naissance": "Lieu de naissance",
            "sexe":           "Sexe",
            "pere":           "Nom du père",
            "mere":           "Nom de la mère",
            "num_acte":       "Numéro de l'acte",
            "date_acte":      "Date de l'acte",
        }
    },
    "acte_heredite": {
        "label": "Acte d'hérédité",
        "fields": {
            "defunt":     "Nom du défunt(e)",
            "date_deces": "Date de décès",
            "lieu_deces": "Lieu de décès",
            "heritiers":  "Liste des héritiers",
            "notaire":    "Notaire",
            "date_acte":  "Date de l'acte",
            "lieu_acte":  "Lieu de l'acte",
        }
    },
    "assurance": {
        "label": "Contrat d'assurance",
        "fields": {
            "assure":        "Assuré(e)",
            "num_contrat":   "N° contrat",
            "type_garantie": "Type de garantie",
            "date_effet":    "Date d'effet",
            "date_echeance": "Date d'échéance",
            "prime":         "Prime/cotisation",
            "assureur":      "Compagnie d'assurance",
            "agence":        "Agence",
        }
    },
    "attestation_solde": {
        "label": "Attestation de solde",
        "fields": {
            "titulaire":   "Titulaire",
            "num_compte":  "N° compte",
            "solde":       "Solde",
            "arrete_au":   "Arrêté au",
            "banque":      "Banque",
            "agence":      "Agence",
            "type_compte": "Type de compte",
        }
    },
    "cin": {
        "label": "Carte d'Identité Nationale (CIN)",
        "fields": {
            "nom":              "Nom de famille",
            "prenom":           "Prénom(s)",
            "cin_number":       "Numéro CIN",
            "date_naissance":   "Date de naissance",
            "lieu_naissance":   "Lieu de naissance",
            "date_delivrance":  "Date de délivrance",
            "date_expiration":  "Date d'expiration",
            "adresse":          "Adresse complète",
            "sexe":             "Sexe (M/F)",
        }
    },
    "lettre_de_change": {
        "label": "Lettre de change",
        "fields": {
            "tireur":          "Tireur (émetteur)",
            "tire":            "Tiré (débiteur)",
            "beneficiaire":    "Bénéficiaire",
            "montant":         "Montant",
            "montant_lettres": "Montant en lettres",
            "date_echeance":   "Date d'échéance",
            "lieu_paiement":   "Lieu de paiement",
            "date_emission":   "Date d'émission",
            "lieu_creation":   "Lieu de création",
            "valeur_recue":    "Valeur reçue (nature)",
            "domiciliation":   "Domiciliation bancaire",
            "num_effet":       "Numéro d'effet",
        }
    },
    "certificat_medical": {
        "label": "Certificat médical",
        "fields": {
            "patient":               "Nom du patient",
            "date_naissance_patient":"Date de naissance du patient",
            "medecin":               "Nom du médecin",
            "specialite":            "Spécialité",
            "etablissement":         "Établissement/Cabinet",
            "date_consultation":     "Date de consultation",
            "diagnostic":            "Diagnostic / Motif",
            "duree_repos":           "Durée d'arrêt/repos",
            "date_debut_arret":      "Date de début d'arrêt",
            "date_fin_arret":        "Date de fin d'arrêt",
            "aptitude":              "Aptitude (apte/inapte/arrêt)",
        }
    },
    "contrat_garantie": {
        "label": "Contrat de garantie",
        "fields": {
            "garant":               "Garant (banque ou personne)",
            "beneficiaire":         "Bénéficiaire de la garantie",
            "debiteur_principal":   "Débiteur principal",
            "montant_garanti":      "Montant garanti",
            "type_garantie":        "Type de garantie",
            "date_effet":           "Date d'effet",
            "date_expiration":      "Date d'expiration",
            "num_contrat":          "Numéro de contrat/référence",
            "conditions_appel":     "Conditions d'appel",
            "banque":               "Banque émettrice",
        }
    },
    "bon_a_ordre": {
        "label": "Bon à ordre / Billet à ordre",
        "fields": {
            "souscripteur":    "Souscripteur (émetteur)",
            "beneficiaire":    "Bénéficiaire",
            "montant":         "Montant",
            "montant_lettres": "Montant en lettres",
            "date_echeance":   "Date d'échéance",
            "lieu_paiement":   "Lieu de paiement",
            "date_emission":   "Date d'émission",
            "lieu_creation":   "Lieu de création",
            "valeur_recue":    "Valeur reçue",
            "num_bon":         "Numéro du bon",
            "domiciliation":   "Domiciliation bancaire",
        }
    },
}

# ── État Ollama ────────────────────────────────────────────────────────────
_use_gemma = True
_use_qwen  = True


def _check_ollama() -> dict:
    global _use_gemma, _use_qwen
    import requests as _req
    try:
        r = _req.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            _use_gemma = _use_qwen = False
            return {"gemma": False, "qwen": False}
        names = [m.get("name", "") for m in r.json().get("models", [])]
        _use_qwen  = any(QWEN_MODEL  in n for n in names)
        _use_gemma = any(GEMMA_MODEL in n for n in names)
        return {"gemma": _use_gemma, "qwen": _use_qwen, "models": names}
    except Exception as e:
        _use_gemma = _use_qwen = False
        return {"gemma": False, "qwen": False, "error": str(e)}


# ── Helpers LLM ────────────────────────────────────────────────────────────
def _call_gemma(prompt: str) -> str:
    import requests as _req
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 400},
        "keep_alive": -1,
    }
    chunks = []
    try:
        with _req.post(OLLAMA_URL, json=payload, stream=True,
                       timeout=(10, OLLAMA_TIMEOUT)) as resp:
            if resp.status_code != 200:
                return ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunks.append(obj.get("response", ""))
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"[Gemma] Erreur : {e}")
    return "".join(chunks)


def _call_qwen(prompt: str, img_b64: str) -> str:
    import requests as _req
    payload = {
        "model": QWEN_MODEL,
        "stream": True,
        "options": {"temperature": 0.1, "num_predict": 400},
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "keep_alive": -1,
    }
    chunks = []
    try:
        with _req.post(OLLAMA_CHAT, json=payload, stream=True,
                       timeout=(10, OLLAMA_TIMEOUT)) as resp:
            if resp.status_code != 200:
                return ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    chunks.append(obj.get("message", {}).get("content", ""))
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"[Qwen] Erreur : {e}")
    return "".join(chunks)


def _img_to_b64(path: str, max_size: int = 1024) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _clean_val(v):
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("", "null", "none", "n/a", "na", "—", "-",
                     "inconnu", "non disponible", "nd", "non renseigné"):
        return None
    return s


def _parse_json(raw: str, field_keys: list) -> dict | None:
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(raw[start:end + 1])
            result = {k: _clean_val(data.get(k)) for k in field_keys}
            if any(v for v in result.values()):
                return result
        except Exception:
            pass
    # fallback regex
    result = {}
    for k in field_keys:
        m = re.search(rf'"{re.escape(k)}"\s*:\s*(?:"([^"]*?)"|null)',
                      raw, re.IGNORECASE)
        result[k] = _clean_val(m.group(1)) if m and m.group(1) else None
    return result if any(v for v in result.values()) else None


# ── Logique d'extraction principale ───────────────────────────────────────
def extract_structured(
    ocr_text: str,
    doc_type: str,
    img_path: str | None = None,
    img_b64: str | None = None,
    force: bool = False,
) -> dict:
    """
    Retourne un JSON structuré :
    {
      "doc_type": str,
      "label": str,
      "fields": { field_key: value_or_null, ... },
      "filled_count": int,
      "total_fields": int,
      "source": "gemma2" | "qwen" | "none",
      "elapsed_ms": int,
      "error": str | null
    }
    """
    t0 = time.time()

    schema = DOC_FIELD_SCHEMAS.get(doc_type)
    if not schema:
        return {
            "doc_type": doc_type,
            "label": doc_type,
            "fields": {},
            "filled_count": 0,
            "total_fields": 0,
            "source": "none",
            "elapsed_ms": 0,
            "error": f"Type '{doc_type}' non supporté. Types disponibles : {list(DOC_FIELD_SCHEMAS.keys())}",
        }

    field_keys  = list(schema["fields"].keys())
    keys_inline = ", ".join(f'"{k}"' for k in field_keys)
    empty       = {k: None for k in field_keys}
    ocr_short   = (ocr_text or "")[:2000]

    result_fields = None
    source = "none"

    # ── Tentative 1 : Gemma2 (texte) ──────────────────────────────────────
    if _use_gemma and ocr_short:
        prompt = (
            f"Tu es un expert en documents bancaires et administratifs marocains. "
            f"Voici le texte OCR d'un document de type '{doc_type}' ({schema['label']}) "
            f"— le texte peut être bruité :\n{ocr_short}\n\n"
            f"Extrais en JSON les champs suivants : {keys_inline}\n"
            f"Mets null si le champ est absent ou illisible. "
            f"Corrige les erreurs OCR évidentes. "
            f"Réponds UNIQUEMENT avec le JSON, sans texte autour :"
        )
        raw = _call_gemma(prompt)
        if raw:
            parsed = _parse_json(raw, field_keys)
            if parsed and any(v for v in parsed.values()):
                result_fields = parsed
                source = "gemma2"

    # ── Tentative 2 : Qwen (vision) si Gemma insuffisant ──────────────────
    if result_fields is None and _use_qwen:
        b64 = img_b64
        if not b64 and img_path and os.path.exists(img_path):
            try:
                b64 = _img_to_b64(img_path)
            except Exception as e:
                print(f"[Extract] img→b64 échoué : {e}")

        if b64:
            prompt_q = (
                f"Document type: {doc_type} ({schema['label']}) — Moroccan banking document. "
                f"Extract these JSON fields: {keys_inline}. "
                f"null if absent. Fix OCR errors. JSON only, no extra text. "
                f"OCR hint: {ocr_short[:400]}"
            )
            raw = _call_qwen(prompt_q, b64)
            if raw:
                parsed = _parse_json(raw, field_keys)
                if parsed and any(v for v in parsed.values()):
                    result_fields = parsed
                    source = "qwen"

    # ── Résultat final ─────────────────────────────────────────────────────
    fields = empty.copy()
    if result_fields:
        fields.update(result_fields)

    filled = sum(1 for v in fields.values() if v)
    elapsed = int((time.time() - t0) * 1000)

    return {
        "doc_type":     doc_type,
        "label":        schema["label"],
        "fields":       fields,
        "filled_count": filled,
        "total_fields": len(field_keys),
        "source":       source,
        "elapsed_ms":   elapsed,
        "error":        None if source != "none" else "Aucun LLM disponible ou extraction échouée",
    }


# ── Alias rétrocompatibilité : carte_identite → cin ───────────────────────
DOC_FIELD_SCHEMAS["carte_identite"] = DOC_FIELD_SCHEMAS["cin"]

# ── App Flask ──────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

_check_ollama()
print(f"[ExtractionService] Gemma: {_use_gemma} | Qwen: {_use_qwen}")


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":  "BankDoc — Extraction Microservice",
        "version":  "1.0",
        "status":   "running",
        "port":     SERVICE_PORT,
        "llm": {
            "gemma_available": _use_gemma,
            "qwen_available":  _use_qwen,
        },
        "endpoints": {
            "GET  /":                "Cette page",
            "GET  /health":          "Statut détaillé du service et des LLMs",
            "GET  /schemas":         "Tous les schémas de champs disponibles",
            "GET  /schemas/<type>":  "Schéma d un type spécifique",
            "POST /extract":         "Extraire les champs structurés d un document",
            "POST /extract/batch":   "Extraction batch (max 50 documents)",
        },
        "supported_types": [t for t in DOC_FIELD_SCHEMAS.keys() if t != "carte_identite"],
        "note": "carte_identite est un alias de cin",
    })


@app.route("/health", methods=["GET"])
def health():
    ollama_status = _check_ollama()
    return jsonify({
        "status":          "ok",
        "service":         "extraction-microservice",
        "port":            SERVICE_PORT,
        "gemma_available": _use_gemma,
        "qwen_available":  _use_qwen,
        "ollama":          ollama_status,
        "supported_types": [t for t in DOC_FIELD_SCHEMAS.keys() if t != "carte_identite"],
    })


@app.route("/schemas", methods=["GET"])
def list_schemas():
    return jsonify({
        "schemas": {
            k: {
                "label":        v["label"],
                "field_count":  len(v["fields"]),
                "fields":       v["fields"],
            }
            for k, v in DOC_FIELD_SCHEMAS.items()
            if k != "carte_identite"
        }
    })


@app.route("/schemas/<doc_type>", methods=["GET"])
def get_schema(doc_type):
    schema = DOC_FIELD_SCHEMAS.get(doc_type)
    if not schema:
        return jsonify({"error": f"Type '{doc_type}' non supporté"}), 404
    return jsonify({
        "doc_type": doc_type,
        "label":    schema["label"],
        "fields":   schema["fields"],
    })


@app.route("/extract", methods=["POST"])
def extract():
    """
    Body JSON :
    {
      "ocr_text":  str          (requis),
      "doc_type":  str          (requis),
      "img_path":  str          (optionnel — chemin local accessible),
      "img_b64":   str          (optionnel — image base64 JPEG/PNG),
      "force":     bool         (optionnel, ignoré ici — pas de cache)
    }

    Réponse :
    {
      "doc_type":     str,
      "label":        str,
      "fields":       { field_key: value_or_null },
      "filled_count": int,
      "total_fields": int,
      "source":       "gemma2" | "qwen" | "none",
      "elapsed_ms":   int,
      "error":        str | null
    }
    """
    data = request.get_json(silent=True) or {}

    ocr_text = data.get("ocr_text", "").strip()
    doc_type = data.get("doc_type", "").strip().lower()
    img_path = data.get("img_path")
    img_b64  = data.get("img_b64")
    force    = data.get("force", False)

    if not doc_type:
        return jsonify({"error": "doc_type est requis"}), 400
    if not ocr_text and not img_b64 and not img_path:
        return jsonify({"error": "Au moins ocr_text, img_path ou img_b64 est requis"}), 400

    try:
        result = extract_structured(
            ocr_text=ocr_text,
            doc_type=doc_type,
            img_path=img_path,
            img_b64=img_b64,
            force=force,
        )
        status = 200 if result["source"] != "none" else 206  # 206 = partial
        return jsonify(result), status
    except Exception as e:
        print(f"[/extract ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e), "doc_type": doc_type, "fields": {}}), 500


@app.route("/extract/batch", methods=["POST"])
def extract_batch():
    """
    Body JSON :
    {
      "documents": [
        { "id": str, "ocr_text": str, "doc_type": str, "img_path": str? },
        ...
      ]
    }

    Réponse :
    {
      "results": [
        { "id": str, ...extraction_result },
        ...
      ],
      "total": int,
      "success_count": int,
      "elapsed_ms": int
    }
    """
    data = request.get_json(silent=True) or {}
    documents = data.get("documents", [])

    if not documents:
        return jsonify({"error": "Liste de documents vide"}), 400
    if len(documents) > 50:
        return jsonify({"error": "Maximum 50 documents par batch"}), 400

    t0 = time.time()
    results = []
    success = 0

    for doc in documents:
        doc_id   = doc.get("id", "")
        ocr_text = doc.get("ocr_text", "").strip()
        doc_type = doc.get("doc_type", "").strip().lower()
        img_path = doc.get("img_path")
        img_b64  = doc.get("img_b64")

        if not doc_type:
            results.append({"id": doc_id, "error": "doc_type manquant", "fields": {}})
            continue

        try:
            result = extract_structured(
                ocr_text=ocr_text,
                doc_type=doc_type,
                img_path=img_path,
                img_b64=img_b64,
            )
            result["id"] = doc_id
            results.append(result)
            if result["source"] != "none":
                success += 1
        except Exception as e:
            results.append({
                "id":       doc_id,
                "doc_type": doc_type,
                "fields":   {},
                "error":    str(e),
                "source":   "none",
            })

    return jsonify({
        "results":       results,
        "total":         len(results),
        "success_count": success,
        "elapsed_ms":    int((time.time() - t0) * 1000),
    })


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Extraction Microservice — port {SERVICE_PORT}")
    print(f"  Gemma : {'✅' if _use_gemma else '❌'}  |  Qwen : {'✅' if _use_qwen else '❌'}")
    print(f"  Types supportés : {len(DOC_FIELD_SCHEMAS)}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=False, threaded=True)