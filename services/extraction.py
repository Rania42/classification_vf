"""
Extraction des métadonnées structurées via LLM.
"""
import re
import time
import os

from config import QWEN_MODEL
from core.agents import (
    _call_ollama_streaming,
    _call_qwen_streaming,
    image_to_base64,
)

GEMMA_MODEL = "gemma2:9b"

DOC_FIELD_SCHEMAS = {
    "carte_identite": {
        "label": "Carte Nationale d'Identité (CIN)",
        "fields": {
            "nom": "Nom de famille", "prenom": "Prénom(s)",
            "cin_number": "Numéro CIN", "date_naissance": "Date de naissance",
            "lieu_naissance": "Lieu de naissance", "date_delivrance": "Date de délivrance",
            "date_expiration": "Date d'expiration", "adresse": "Adresse complète",
            "sexe": "Sexe (M/F)",
        }
    },
    "cin": {
        "label": "Carte d'Identité Nationale (CIN)",
        "fields": {
            "nom": "Nom de famille", "prenom": "Prénom(s)",
            "cin_number": "Numéro CIN", "date_naissance": "Date de naissance",
            "lieu_naissance": "Lieu de naissance", "date_delivrance": "Date de délivrance",
            "date_expiration": "Date d'expiration", "adresse": "Adresse complète",
            "sexe": "Sexe (M/F)",
        }
    },
    "rib": {
        "label": "Relevé d'Identité Bancaire",
        "fields": {
            "titulaire": "Nom du titulaire", "iban": "IBAN complet", "bic": "Code BIC/SWIFT",
            "code_banque": "Code banque", "code_guichet": "Code guichet",
            "num_compte": "Numéro de compte", "cle_rib": "Clé RIB",
            "domiciliation": "Domiciliation/agence", "banque": "Nom de la banque",
        }
    },
    "cheque": {
        "label": "Chèque bancaire",
        "fields": {
            "beneficiaire": "Bénéficiaire", "montant_chiffres": "Montant en chiffres",
            "montant_lettres": "Montant en lettres", "date_emission": "Date d'émission",
            "num_cheque": "Numéro du chèque", "tire_sur": "Banque tirée",
            "emetteur": "Émetteur/signataire", "endossable": "Endossable (oui/non)",
        }
    },
    "tableau_amortissement": {
        "label": "Tableau d'amortissement",
        "fields": {
            "montant_pret": "Montant du prêt", "taux_interet": "Taux d'intérêt",
            "duree": "Durée du prêt", "mensualite": "Mensualité",
            "capital_initial": "Capital initial", "date_debut": "Date de début",
            "nombre_echeances": "Nombre d'échéances", "banque": "Banque prêteuse",
        }
    },
    "acte_naissance": {
        "label": "Acte de naissance",
        "fields": {
            "nom": "Nom", "prenom": "Prénom(s)", "date_naissance": "Date de naissance",
            "heure_naissance": "Heure de naissance", "lieu_naissance": "Lieu de naissance",
            "sexe": "Sexe", "pere": "Nom du père", "mere": "Nom de la mère",
            "num_acte": "Numéro de l'acte", "date_acte": "Date de l'acte",
        }
    },
    "acte_heredite": {
        "label": "Acte d'hérédité",
        "fields": {
            "defunt": "Nom du défunt(e)", "date_deces": "Date de décès",
            "lieu_deces": "Lieu de décès", "heritiers": "Liste des héritiers",
            "notaire": "Notaire", "date_acte": "Date de l'acte",
            "lieu_acte": "Lieu de l'acte",
        }
    },
    "assurance": {
        "label": "Contrat d'assurance",
        "fields": {
            "assure": "Assuré(e)", "num_contrat": "N° contrat",
            "type_garantie": "Type de garantie", "date_effet": "Date d'effet",
            "date_echeance": "Date d'échéance", "prime": "Prime/cotisation",
            "assureur": "Compagnie d'assurance", "agence": "Agence",
        }
    },
    "attestation_solde": {
        "label": "Attestation de solde",
        "fields": {
            "titulaire": "Titulaire", "num_compte": "N° compte", "solde": "Solde",
            "arrete_au": "Arrêté au", "banque": "Banque", "agence": "Agence",
            "type_compte": "Type de compte",
        }
    },
    # ── Nouveaux types ──────────────────────────────────────────────────────
    "lettre_de_change": {
        "label": "Lettre de change",
        "fields": {
            "tireur": "Tireur (émetteur)", "tire": "Tiré (débiteur)",
            "beneficiaire": "Bénéficiaire", "montant": "Montant",
            "montant_lettres": "Montant en lettres", "date_echeance": "Date d'échéance",
            "lieu_paiement": "Lieu de paiement", "date_emission": "Date d'émission",
            "lieu_creation": "Lieu de création", "valeur_recue": "Valeur reçue (nature)",
            "domiciliation": "Domiciliation bancaire", "num_effet": "Numéro d'effet",
        }
    },
    "certificat_medical": {
        "label": "Certificat médical",
        "fields": {
            "patient": "Nom du patient", "date_naissance_patient": "Date de naissance du patient",
            "medecin": "Nom du médecin", "specialite": "Spécialité",
            "etablissement": "Établissement/Cabinet", "date_consultation": "Date de consultation",
            "diagnostic": "Diagnostic / Motif", "duree_repos": "Durée d'arrêt/repos",
            "date_debut_arret": "Date de début d'arrêt", "date_fin_arret": "Date de fin d'arrêt",
            "aptitude": "Aptitude (apte/inapte/arrêt)",
        }
    },
    "contrat_garantie": {
        "label": "Contrat de garantie",
        "fields": {
            "garant": "Garant (banque ou personne)", "beneficiaire": "Bénéficiaire de la garantie",
            "debiteur_principal": "Débiteur principal", "montant_garanti": "Montant garanti",
            "type_garantie": "Type de garantie (hypothèque, caution, nantissement…)",
            "date_effet": "Date d'effet", "date_expiration": "Date d'expiration",
            "num_contrat": "Numéro de contrat/référence", "conditions_appel": "Conditions d'appel",
            "banque": "Banque émettrice",
        }
    },
    "bon_a_ordre": {
        "label": "Bon à ordre / Billet à ordre",
        "fields": {
            "souscripteur": "Souscripteur (émetteur)", "beneficiaire": "Bénéficiaire",
            "montant": "Montant", "montant_lettres": "Montant en lettres",
            "date_echeance": "Date d'échéance", "lieu_paiement": "Lieu de paiement",
            "date_emission": "Date d'émission", "lieu_creation": "Lieu de création",
            "valeur_recue": "Valeur reçue", "num_bon": "Numéro du bon",
            "domiciliation": "Domiciliation bancaire",
        }
    },
}


def _clean_val(v):
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("", "null", "none", "n/a", "na", "—", "-", "inconnu",
                     "non disponible", "nd", "non renseigné"):
        return None
    return s


def _parse_llm_json(raw: str, field_keys: list) -> dict | None:
    import json
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```", "", raw).strip()

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data   = json.loads(raw[start:end + 1])
            result = {k: _clean_val(data.get(k)) for k in field_keys}
            if any(v for v in result.values()):
                return result
        except Exception:
            pass

    try:
        data = json.loads(raw)
        return {k: _clean_val(data.get(k)) for k in field_keys}
    except Exception:
        pass

    result = {}
    for k in field_keys:
        m = re.search(rf'"{re.escape(k)}"\s*:\s*(?:"([^"]*?)"|null|None)',
                      raw, re.IGNORECASE)
        result[k] = _clean_val(m.group(1)) if m and m.group(1) else None
    if any(v for v in result.values()):
        return result
    return None


def extract_metadata_with_llm(ocr_text: str, doc_type: str, stored_path: str = "") -> dict:
    import core.agents as _agents_module

    schema = DOC_FIELD_SCHEMAS.get(doc_type)
    if not schema:
        return {"error": f"Type '{doc_type}' non supporté pour l'extraction", "_error": True}

    field_keys  = list(schema["fields"].keys())
    empty       = {k: None for k in field_keys}
    keys_inline = ", ".join(f'"{k}"' for k in field_keys)
    ocr_short   = ocr_text[:2000]

    prompt = (
        f"Tu es un expert en documents bancaires et administratifs marocains. "
        f"Voici le texte OCR d'un document de type '{doc_type}' ({schema['label']}) — le texte peut être bruité :\n"
        f"{ocr_short}\n\n"
        f"Extrais en JSON les champs suivants : {keys_inline}\n"
        f"Mets null si le champ est absent ou illisible. Corrige les erreurs OCR évidentes. "
        f"Réponds UNIQUEMENT avec le JSON, sans texte autour :"
    )

    if _agents_module.USE_OLLAMA:
        print(f"[Extract] Gemma2:9b pour '{doc_type}'...")
        t0  = time.time()
        raw = _call_ollama_streaming(prompt, GEMMA_MODEL, timeout_no_token=60)
        elapsed = time.time() - t0
        print(f"[Extract] Gemma2 → {len(raw)} chars en {elapsed:.1f}s")
        if raw:
            result = _parse_llm_json(raw, field_keys)
            if result and any(v for v in result.values()):
                full = empty.copy()
                full.update(result)
                full["_source"] = "gemma2"
                full["_error"]  = False
                return full
        print("[Extract] Gemma2 : réponse vide ou JSON invalide")

    if _agents_module.USE_QWEN and stored_path and os.path.exists(stored_path):
        prompt_q = (
            f"Document type: {doc_type} ({schema['label']}) — Moroccan banking/administrative document. "
            f"Extract these JSON fields: {keys_inline}. "
            f"null if absent. Fix OCR errors. JSON only, no extra text:"
        )
        try:
            print(f"[Extract] Qwen2.5-VL fallback pour '{doc_type}'...")
            img_b64 = image_to_base64(stored_path)
            t0  = time.time()
            raw = _call_qwen_streaming(prompt_q, img_b64, timeout_no_token=60)
            elapsed = time.time() - t0
            print(f"[Extract] Qwen → {len(raw)} chars en {elapsed:.1f}s")
            if raw:
                result = _parse_llm_json(raw, field_keys)
                if result and any(v for v in result.values()):
                    full = empty.copy()
                    full.update(result)
                    full["_source"] = "qwen"
                    full["_error"]  = False
                    return full
            print("[Extract] Qwen : réponse vide ou JSON invalide")
        except Exception as e:
            print(f"[Extract Qwen] ❌ {e}")

    print(f"[Extract] Échec total pour '{doc_type}'")
    empty["_source"]    = "none"
    empty["_error"]     = True
    empty["_error_msg"] = "Aucun LLM disponible ou extraction échouée"
    return empty