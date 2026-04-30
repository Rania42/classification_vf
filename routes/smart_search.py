"""
Route POST /smart_search — recherche intelligente via LLM (Gemma2).
Le LLM interprète le prompt en langage naturel et génère des filtres MongoDB.
"""
import re
import json
from flask import Blueprint, request, jsonify

from services.mongo import get_mongo_db, is_mongo_available, _serialize_doc
from core.agents import _call_ollama_streaming, USE_OLLAMA

smart_search_bp = Blueprint("smart_search", __name__)

GEMMA_MODEL = "gemma2:9b"

DOC_TYPES = [
    "rib", "cheque", "tableau_amortissement", "acte_naissance",
    "acte_heredite", "assurance", "attestation_solde", "carte_identite"
]


def _llm_interpret_query(user_prompt: str) -> dict:
    """
    Demande à Gemma2 d'interpréter le prompt et de retourner des filtres structurés.
    Retourne un dict avec: doc_types, keywords, extracted_fields, text_query, explanation
    """
    if not USE_OLLAMA:
        # Fallback: recherche textuelle brute
        return {
            "doc_types": [],
            "keywords": user_prompt.split(),
            "extracted_fields": {},
            "text_query": user_prompt,
            "explanation": "Recherche textuelle directe (LLM indisponible)"
        }

    prompt = f"""Tu es un assistant expert en documents bancaires marocains.
L'utilisateur veut rechercher des documents dans une base de données.

Types de documents disponibles : {', '.join(DOC_TYPES)}

Prompt de l'utilisateur : "{user_prompt}"

Analyse ce prompt et retourne UNIQUEMENT un JSON avec ces champs :
{{
  "doc_types": ["liste des types de documents concernés, ex: cheque, rib"],
  "keywords": ["mots-clés importants à chercher dans le texte OCR"],
  "extracted_fields": {{"champ": "valeur"}},
  "text_query": "requête de recherche textuelle optimisée",
  "explanation": "explication courte de la recherche effectuée en français"
}}

Exemples :
- "le chèque de monsieur Alami" → doc_types: ["cheque"], keywords: ["Alami"], extracted_fields: {{"beneficiaire": "Alami"}}
- "les RIB de la banque Attijariwafa" → doc_types: ["rib"], keywords: ["Attijariwafa"], extracted_fields: {{"banque": "Attijariwafa"}}
- "actes de naissance de 2023" → doc_types: ["acte_naissance"], keywords: ["2023"]
- "tous les documents de Ahmed Benali" → doc_types: [], keywords: ["Ahmed", "Benali"]

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

    # Parse JSON
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
    """Construit la requête MongoDB depuis l'interprétation LLM."""
    conditions = []

    # Filtre par type(s) de document
    if interpretation.get("doc_types"):
        conditions.append({"prediction": {"$in": interpretation["doc_types"]}})

    # Recherche textuelle dans OCR + filename
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

        # Recherche dans les champs extraits
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

    # 1. Interprétation LLM
    interpretation = _llm_interpret_query(user_prompt)

    # 2. Construction requête MongoDB
    mongo_query = _build_mongo_query(interpretation)

    # 3. Exécution
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