"""
Client HTTP pour le microservice d'extraction (port 5001).
Remplace les appels directs à extract_metadata_with_llm() dans documents.py.

Si le microservice est indisponible, fallback automatique sur l'extraction locale.
"""

import os
import requests
from typing import Optional

EXTRACTION_SERVICE_URL = os.environ.get(
    "EXTRACTION_SERVICE_URL", "http://localhost:5001"
)
_SERVICE_TIMEOUT = int(os.environ.get("EXTRACTION_SERVICE_TIMEOUT", 90))
_service_available: bool | None = None  # None = pas encore testé


def _check_service() -> bool:
    global _service_available
    try:
        r = requests.get(
            f"{EXTRACTION_SERVICE_URL}/health",
            timeout=3,
        )
        _service_available = r.status_code == 200
    except Exception:
        _service_available = False
    return _service_available


def is_service_available() -> bool:
    global _service_available
    if _service_available is None:
        return _check_service()
    return _service_available


def extract_via_service(
    ocr_text: str,
    doc_type: str,
    img_path: Optional[str] = None,
    force: bool = False,
) -> dict:
    """
    Appelle le microservice d'extraction.

    Retourne le JSON structuré du microservice :
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

    En cas d'erreur, retourne un dict avec "_error": True pour que
    documents.py puisse basculer sur le fallback local.
    """
    global _service_available

    payload = {
        "ocr_text": ocr_text,
        "doc_type": doc_type,
        "force":    force,
    }
    # On passe le chemin si accessible depuis le microservice (même machine)
    if img_path and os.path.exists(img_path):
        payload["img_path"] = img_path

    try:
        resp = requests.post(
            f"{EXTRACTION_SERVICE_URL}/extract",
            json=payload,
            timeout=_SERVICE_TIMEOUT,
        )
        _service_available = True

        data = resp.json()

        # Normalise la réponse pour compatibilité avec l'ancien format
        if resp.status_code in (200, 206):
            fields = data.get("fields", {})
            # Nettoie les clés internes éventuelles
            fields.pop("_source", None)
            fields.pop("_error", None)
            fields.pop("_error_msg", None)

            return {
                **fields,                                    # champs métier à plat
                "_fields":       fields,                     # champs structurés
                "_source":       data.get("source", "none"),
                "_error":        data.get("error") is not None and data.get("source") == "none",
                "_error_msg":    data.get("error"),
                "_filled_count": data.get("filled_count", 0),
                "_total_fields": data.get("total_fields", 0),
                "_elapsed_ms":   data.get("elapsed_ms", 0),
                "_via_service":  True,
            }

        # Erreur HTTP
        return {
            "_source":    "none",
            "_error":     True,
            "_error_msg": data.get("error", f"HTTP {resp.status_code}"),
            "_via_service": True,
        }

    except requests.exceptions.ConnectionError:
        _service_available = False
        return {
            "_source":    "none",
            "_error":     True,
            "_error_msg": "Microservice d'extraction non joignable",
            "_via_service": False,
        }
    except requests.exceptions.Timeout:
        return {
            "_source":    "none",
            "_error":     True,
            "_error_msg": f"Timeout ({_SERVICE_TIMEOUT}s) du microservice",
            "_via_service": True,
        }
    except Exception as e:
        return {
            "_source":    "none",
            "_error":     True,
            "_error_msg": str(e),
            "_via_service": True,
        }


def extract_with_fallback(
    ocr_text: str,
    doc_type: str,
    stored_path: Optional[str] = None,
    force: bool = False,
) -> dict:
    """
    Tente d'abord le microservice, puis bascule sur l'extraction locale
    (extract_metadata_with_llm) si indisponible.

    Utilisé par documents.py à la place de l'appel direct.
    """
    # ── Tentative microservice ─────────────────────────────────────────────
    if is_service_available():
        result = extract_via_service(ocr_text, doc_type, stored_path, force)
        if not result.get("_error"):
            print(f"[ExtractionClient] ✅ via microservice → {result.get('_source')} "
                  f"({result.get('_filled_count', 0)}/{result.get('_total_fields', 0)} champs)")
            return result
        print(f"[ExtractionClient] ⚠ microservice KO : {result.get('_error_msg')} "
              f"— fallback local")

    # ── Fallback : extraction locale ───────────────────────────────────────
    print(f"[ExtractionClient] 🔄 Fallback extraction locale pour '{doc_type}'")
    try:
        from services.extraction import extract_metadata_with_llm
        return extract_metadata_with_llm(ocr_text, doc_type, stored_path or "")
    except Exception as e:
        print(f"[ExtractionClient] ❌ Fallback local échoué : {e}")
        return {
            "_source":    "none",
            "_error":     True,
            "_error_msg": str(e),
        }


def get_service_status() -> dict:
    """Retourne le statut complet du microservice (pour /status)."""
    try:
        r = requests.get(f"{EXTRACTION_SERVICE_URL}/health", timeout=3)
        if r.status_code == 200:
            return {"available": True, "url": EXTRACTION_SERVICE_URL, **r.json()}
    except Exception as e:
        return {"available": False, "url": EXTRACTION_SERVICE_URL, "error": str(e)}
    return {"available": False, "url": EXTRACTION_SERVICE_URL}


def get_supported_schemas() -> dict:
    """Récupère les schémas depuis le microservice."""
    try:
        r = requests.get(f"{EXTRACTION_SERVICE_URL}/schemas", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}