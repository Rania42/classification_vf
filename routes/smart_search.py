"""
Route POST /smart_search — recherche intelligente via LLM (Gemma2).
"""
import re
import json
from flask import Blueprint, request, jsonify

from services.mongo import get_mongo_db, is_mongo_available, _serialize_doc
from core.agents import _call_ollama_streaming, USE_OLLAMA

smart_search_bp = Blueprint("smart_search", __name__)

GEMMA_MODEL = "gemma2:2b"

DOC_TYPES = [
    "rib", "cheque", "tableau_amortissement", "acte_naissance",
    "acte_heredite", "assurance", "attestation_solde", "carte_identite",
    "cin", "lettre_de_change", "certificat_medical", "contrat_garantie", "bon_a_ordre",
]

DOC_TYPES_LABELS = {
    "rib": "RIB / Relevé d'Identité Bancaire",
    "cheque": "Chèque bancaire",
    "tableau_amortissement": "Tableau d'amortissement",
    "acte_naissance": "Acte de naissance",
    "acte_heredite": "Acte d'hérédité",
    "assurance": "Contrat d'assurance",
    "attestation_solde": "Attestation de solde",
    "carte_identite": "Carte d'Identité Nationale",
    "cin": "Carte d'Identité Nationale (CIN)",
    "lettre_de_change": "Lettre de change / Traite",
    "certificat_medical": "Certificat médical",
    "contrat_garantie": "Contrat de garantie",
    "bon_a_ordre": "Bon à ordre / Billet à ordre",
}


def _llm_interpret_query(user_prompt: str) -> dict:
    if not USE_OLLAMA:
        return {
            "doc_types": [],
            "keywords": user_prompt.split(),
            "extracted_fields": {},
            "text_query": user_prompt,
            "explanation": "Recherche textuelle directe (LLM indisponible)"
        }

    types_with_labels = "\n".join(f"  - {k}: {v}" for k, v in DOC_TYPES_LABELS.items())
    prompt = f"""Tu es un assistant expert en documents bancaires et administratifs marocains.
L'utilisateur veut rechercher des documents dans une base de données.

Types de documents disponibles :
{types_with_labels}

Prompt de l'utilisateur : "{user_prompt}"

Analyse ce prompt et retourne UNIQUEMENT un JSON avec ces champs :
{{
  "doc_types": ["liste des types de documents concernés"],
  "keywords": ["mots-clés importants à chercher dans le texte OCR"],
  "extracted_fields": {{"champ": "valeur"}},
  "text_query": "requête de recherche textuelle optimisée",
  "explanation": "explication courte de la recherche effectuée en français"
}}

Exemples :
- "le chèque de monsieur Alami" → doc_types: ["cheque"], keywords: ["Alami"]
- "les lettres de change de 2024" → doc_types: ["lettre_de_change"], keywords: ["2024"]
- "certificats médicaux d'arrêt de travail" → doc_types: ["certificat_medical"], keywords: ["arrêt de travail"]
- "contrats de garantie bancaire" → doc_types: ["contrat_garantie"], keywords: ["garantie"]
- "bons à ordre" → doc_types: ["bon_a_ordre"], keywords: []
- "cartes d'identité expirées" → doc_types: ["cin"], keywords: ["expir"]

Réponds UNIQUEMENT avec le JSON, aucun autre texte."""

    raw = _call_ollama_streaming(prompt, GEMMA_MODEL, timeout_no_token=45)
    if not raw:
        return {
            "doc_types": [],
            "keywords": user_prompt.split(),
            "extracted_fields": {},
            "text_query": user_prompt,
            "explanation": "LLM n'a pas répondu — recherche textuelle directe"
        }

    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(raw[start:end + 1])
            return {
                "doc_types": data.get("doc_types", []),
                "keywords": data.get("keywords", []),
                "extracted_fields": data.get("extracted_fields", {}),
                "text_query": data.get("text_query", user_prompt),
                "explanation": data.get("explanation", "Recherche effectuée")
            }
        except Exception:
            pass

    return {
        "doc_types": [],
        "keywords": user_prompt.split(),
        "extracted_fields": {},
        "text_query": user_prompt,
        "explanation": "Interprétation partielle — recherche textuelle"
    }


def _build_mongo_query(interpretation: dict) -> dict:
    conditions = []

    if interpretation.get("doc_types"):
        conditions.append({"prediction": {"$in": interpretation["doc_types"]}})

    all_keywords = []
    if interpretation.get("keywords"):
        all_keywords.extend(interpretation["keywords"])
    if interpretation.get("text_query"):
        all_keywords.append(interpretation["text_query"])

    if all_keywords:
        text_conditions = []
        for kw in all_keywords:
            kw = kw.strip()
            if len(kw) >= 2:
                text_conditions.append({"ocr_text": {"$regex": kw, "$options": "i"}})
                text_conditions.append({"original_filename": {"$regex": kw, "$options": "i"}})

        for field_key, field_val in interpretation.get("extracted_fields", {}).items():
            if field_val:
                text_conditions.append({
                    f"extracted_fields.{field_key}": {"$regex": str(field_val), "$options": "i"}
                })

        if text_conditions:
            conditions.append({"$or": text_conditions})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@smart_search_bp.route("/smart_search", methods=["POST"])
def smart_search():
    if not is_mongo_available():
        return jsonify({"error": "MongoDB non disponible", "results": []}), 503

    db = get_mongo_db()
    if db is None:
        return jsonify({"error": "MongoDB non disponible", "results": []}), 503

    data = request.json or {}
    user_prompt = data.get("prompt", "").strip()
    limit = min(int(data.get("limit", 20)), 50)

    if not user_prompt:
        return jsonify({"error": "Prompt vide", "results": []}), 400

    interpretation = _llm_interpret_query(user_prompt)
    mongo_query = _build_mongo_query(interpretation)

    try:
        docs = list(
            db.documents.find(mongo_query)
            .sort("uploaded_at", -1)
            .limit(limit)
        )
        serialized = [_serialize_doc(d) for d in docs]

        return jsonify({
            "results": serialized,
            "total": len(serialized),
            "interpretation": interpretation,
            "query_used": str(mongo_query)[:500],
            "llm_available": USE_OLLAMA
        })
    except Exception as e:
        import traceback
        print(f"[SmartSearch ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e), "results": []}), 500