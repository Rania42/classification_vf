"""
Agents d'analyse : llama3.2 (Ollama streaming), Qwen2.5-VL (streaming), Nettoyeur regex.
"""
import re
import io
import base64
import requests
from PIL import Image

from config import (
    OLLAMA_URL, OLLAMA_CHAT, OLLAMA_MODEL, QWEN_MODEL,
    OLLAMA_TIMEOUT, QWEN_TIMEOUT, KEYWORD_RULES,
)

# ── État des agents (modifiable via /toggle_*) ─────────
USE_OLLAMA = True
USE_QWEN   = True


def check_ollama_available() -> dict:
    """Vérifie quels modèles Ollama sont disponibles."""
    global USE_OLLAMA, USE_QWEN
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            USE_OLLAMA = USE_QWEN = False
            return {"ollama": False, "qwen": False}
        names = [m.get("name", "") for m in r.json().get("models", [])]
        USE_OLLAMA = any(OLLAMA_MODEL in n for n in names)
        USE_QWEN   = any(QWEN_MODEL   in n for n in names)
        print(f"[Ollama] llama3.2 : {'✅' if USE_OLLAMA else '❌'}  |  qwen2.5vl : {'✅' if USE_QWEN else '❌'}")
    except Exception as e:
        print(f"[Ollama] ❌ Non disponible : {e}")
        USE_OLLAMA = USE_QWEN = False
    return {"ollama": USE_OLLAMA, "qwen": USE_QWEN}


# ── Streaming Ollama ───────────────────────────────────
def _call_ollama_streaming(prompt: str, model: str, timeout_no_token: int = 30) -> str:
    """
    Appel streaming Ollama /api/generate.
    timeout_no_token : secondes sans recevoir de token avant d'abandonner.
    """
    import json as _json
    payload = {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": 0.1, "num_predict": 300},
    }
    chunks = []
    try:
        with requests.post(
            OLLAMA_URL, json=payload, stream=True,
            timeout=(10, timeout_no_token),  # (connect, read)
        ) as resp:
            if resp.status_code != 200:
                print(f"[llama3.2] HTTP {resp.status_code}")
                return ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    chunks.append(obj.get("response", ""))
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except requests.exceptions.ConnectTimeout:
        print("[llama3.2] Timeout connexion")
    except requests.exceptions.ReadTimeout:
        print(f"[llama3.2] Timeout lecture ({timeout_no_token}s sans token)")
    except requests.exceptions.ConnectionError:
        print("[llama3.2] Ollama non joignable (service arrêté ?)")
    except Exception as e:
        print(f"[llama3.2] Erreur inattendue : {e}")
    return "".join(chunks)


def _call_qwen_streaming(prompt: str, img_b64: str, timeout_no_token: int = 60) -> str:
    """
    Appel streaming Qwen2.5-VL via /api/chat.
    timeout_no_token plus long car modèle vision plus lent.
    """
    import json as _json
    payload = {
        "model": QWEN_MODEL, "stream": True,
        "options": {"temperature": 0.1, "num_predict": 300},
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    }
    chunks = []
    try:
        with requests.post(
            OLLAMA_CHAT, json=payload, stream=True,
            timeout=(10, timeout_no_token),
        ) as resp:
            if resp.status_code != 200:
                print(f"[Qwen] HTTP {resp.status_code}")
                return ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    obj = _json.loads(line)
                    chunks.append(obj.get("message", {}).get("content", ""))
                    if obj.get("done"):
                        break
                except Exception:
                    continue
    except requests.exceptions.ConnectTimeout:
        print("[Qwen] Timeout connexion")
    except requests.exceptions.ReadTimeout:
        print(f"[Qwen] Timeout lecture ({timeout_no_token}s sans token)")
    except requests.exceptions.ConnectionError:
        print("[Qwen] Ollama non joignable")
    except Exception as e:
        print(f"[Qwen] Erreur inattendue : {e}")
    return "".join(chunks)


# ── Image → base64 ─────────────────────────────────────
def image_to_base64(img_path: str, max_size: int = 1024) -> str:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Parse JSON réponse LLM ─────────────────────────────
def _parse_classification_json(raw: str, valid_classes: list) -> dict:
    import json
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip()
    data = {}
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r'\{[^{}]*"class"[^{}]*\}', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except Exception:
                pass

    cls        = data.get("class", "").strip().lower().replace(" ", "_")
    confidence = float(data.get("confidence", 0.5))
    reasoning  = data.get("reasoning", "")

    cls_valid = next((c for c in valid_classes if c.lower() == cls or cls in c.lower()), None)
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


# ═══════════════════════════════════════════════════════
# AGENT LLAMA3.2 — Classification via texte OCR
# ═══════════════════════════════════════════════════════
def agent_llm_classification(ocr_text: str, doc_classes: list, context: str = "") -> str | None:
    """
    Appelle llama3.2 pour classifier le document via son texte OCR.
    Retourne la classe prédite ou None si échec/timeout/indisponible.
    """
    if not USE_OLLAMA:
        return None

    classes_str = ", ".join(doc_classes)
    prompt = (
        f"Tu es un expert en documents bancaires marocains.\n"
        f"Catégories disponibles : {classes_str}\n"
        f"{'Contexte : ' + context + chr(10) if context else ''}"
        f"Texte OCR du document :\n{ocr_text[:1500]}\n\n"
        f"Analyse le texte et réponds UNIQUEMENT avec le nom exact de la catégorie "
        f"(parmi les catégories listées). Si aucune ne correspond : 'inconnu'.\n"
        f"Catégorie :"
    )

    raw = _call_ollama_streaming(prompt, OLLAMA_MODEL, timeout_no_token=30)
    if not raw:
        return None

    raw_lower = raw.lower().strip()
    for cls in doc_classes:
        if cls.lower() in raw_lower:
            return cls
    return None


# ═══════════════════════════════════════════════════════
# AGENT QWEN2.5-VL — Confirmation vision multimodale
# ═══════════════════════════════════════════════════════
def agent_qwen_vision(img_path: str, ocr_text: str, doc_classes: list,
                      timeout: int = QWEN_TIMEOUT, retry: bool = True) -> dict:
    """
    Appelle Qwen2.5-VL en streaming pour analyser l'image.
    Retourne {"class": ..., "confidence": ..., "reasoning": ..., "available": bool}
    """
    if not USE_QWEN:
        return {"class": None, "confidence": 0.0, "reasoning": "Qwen désactivé", "available": False}

    classes_str = "\n".join(f"- {c}" for c in doc_classes)
    prompt = (
        f"Tu es un expert en documents bancaires marocains.\n"
        f"Catégories disponibles :\n{classes_str}\n"
        f"Texte OCR (peut être bruité) : {ocr_text[:600]}\n\n"
        f"Analyse l'image et réponds UNIQUEMENT en JSON :\n"
        f'{{ "class": "nom_categorie", "confidence": 0.95, "reasoning": "explication courte" }}\n'
        f"Si le document ne correspond à aucune catégorie : class = 'inconnu'."
    )

    try:
        img_b64 = image_to_base64(img_path)
        raw = _call_qwen_streaming(prompt, img_b64, timeout_no_token=timeout)
        if raw:
            result = _parse_classification_json(raw, doc_classes)
            result["available"] = True
            return result

        if retry:
            print("[Qwen] Retry 1x après échec...")
            raw = _call_qwen_streaming(prompt, img_b64, timeout_no_token=timeout)
            if raw:
                result = _parse_classification_json(raw, doc_classes)
                result["available"] = True
                return result

    except Exception as e:
        print(f"[Qwen] Erreur : {e}")

    return {"class": None, "confidence": 0.0, "reasoning": "Timeout ou erreur", "available": False}


# ═══════════════════════════════════════════════════════
# AGENT KEYWORDS — Détection rapide par règles lexicales
# (utilisé uniquement pour information / debug, non décisionnel)
# ═══════════════════════════════════════════════════════
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
        found.setdefault("carte_identite", []).append("date_naissance_pattern")

    if not scores:
        return {"class": None, "keywords_found": {}, "score": 0}
    best = max(scores, key=scores.get)
    return {"class": best, "keywords_found": found, "score": scores[best]}


# ═══════════════════════════════════════════════════════
# AGENT NETTOYEUR — Regex uniquement (pas de LLM)
# Appel LLM supprimé : trop coûteux en latence pour ce rôle
# ═══════════════════════════════════════════════════════
def agent_nettoyeur(raw_text: str) -> str:
    """
    Nettoie le texte OCR par regex.
    Pas d'appel LLM ici — la latence doit rester minimale.
    """
    cleaned = re.sub(r"[^\w\s\-\.,;:éèêëàâäôöùûüçæœ]", " ", raw_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else raw_text