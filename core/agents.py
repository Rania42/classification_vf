"""
Agents d'analyse : Qwen2.5-VL (principal), Gemma2:9b (vérificateur), regex nettoyeur.
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

GEMMA_MODEL = "gemma2:2b"

USE_OLLAMA = True
USE_QWEN   = True

# ── Descriptions détaillées des classes pour les LLMs ──
DOC_CLASS_DESCRIPTIONS = {
    "rib": "Relevé d'Identité Bancaire — contient IBAN, BIC, code banque, code guichet, numéro de compte, clé RIB, titulaire, domiciliation bancaire.",
    "cheque": "Chèque bancaire — contient 'payez contre ce chèque', montant en chiffres et en lettres, bénéficiaire, date, signature, numéro de chèque, banque tirée.",
    "tableau_amortissement": "Tableau d'amortissement de prêt — tableau avec colonnes échéance, capital, intérêts, assurance, mensualité, capital restant dû.",
    "acte_naissance": "Acte de naissance — document d'état civil avec nom, prénom, date et lieu de naissance, filiation (père/mère), officier d'état civil.",
    "acte_heredite": "Acte d'hérédité / Acte de notoriété successoral — liste des héritiers, défunt, succession, notaire, dévolution successorale.",
    "assurance": "Contrat ou attestation d'assurance — police d'assurance, assuré, assureur, prime, garanties, sinistre, date d'effet et d'échéance.",
    "attestation_solde": "Attestation de solde bancaire — la banque certifie le solde créditeur d'un compte à une date donnée, arrêté au, disponibilité.",
    "cin": "Carte d'Identité Nationale marocaine (CIN/CNI) — carte plastifiée avec photo, numéro CIN, nom, prénom, date/lieu de naissance, adresse, date d'expiration, المملكة المغربية.",
    "lettre_de_change": "Lettre de change (traite) — effet de commerce avec tireur, tiré, bénéficiaire, montant, date d'échéance, lieu de paiement, mention 'valeur reçue', acceptation, aval.",
    "certificat_medical": "Certificat médical — document signé par un médecin certifiant l'état de santé, diagnostic, aptitude/inaptitude, arrêt de travail, repos médical, cachet du médecin.",
    "contrat_garantie": "Contrat ou lettre de garantie — engagement de garantie bancaire, cautionnement, hypothèque, nantissement, garant, montant garanti, durée, appel en garantie.",
    "bon_a_ordre": "Bon à ordre / Billet à ordre — titre négociable, promesse de payer, souscripteur, bénéficiaire, montant, échéance, mentions 'je paierai' ou 'nous paierons', endossement.",
}


def check_ollama_available() -> dict:
    global USE_OLLAMA, USE_QWEN
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            USE_OLLAMA = USE_QWEN = False
            return {"ollama": False, "qwen": False, "gemma": False}
        names = [m.get("name", "") for m in r.json().get("models", [])]
        USE_QWEN   = any(QWEN_MODEL   in n for n in names)
        USE_OLLAMA = any(GEMMA_MODEL  in n for n in names)
        gemma_ok   = USE_OLLAMA
        llama_ok   = any("llama3.2" in n for n in names)
        print(f"[Ollama] qwen2.5vl : {'✅' if USE_QWEN else '❌'}  |  gemma2:9b : {'✅' if gemma_ok else '❌'}  |  llama3.2 : {'✅' if llama_ok else '❌'}")
    except Exception as e:
        print(f"[Ollama] ❌ Non disponible : {e}")
        USE_OLLAMA = USE_QWEN = False
    return {"ollama": USE_OLLAMA, "qwen": USE_QWEN, "gemma": USE_OLLAMA}


def _call_ollama_streaming(prompt: str, model: str, timeout_no_token: int = 30) -> str:
    import json as _json
    payload = {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": 0.1, "num_predict": 300},
        "keep_alive": -1,
    }
    chunks = []
    try:
        with requests.post(
            OLLAMA_URL, json=payload, stream=True,
            timeout=(10, timeout_no_token),
        ) as resp:
            if resp.status_code != 200:
                print(f"[{model}] HTTP {resp.status_code}")
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
        print(f"[{model}] Timeout connexion")
    except requests.exceptions.ReadTimeout:
        print(f"[{model}] Timeout lecture ({timeout_no_token}s)")
    except requests.exceptions.ConnectionError:
        print(f"[{model}] Ollama non joignable")
    except Exception as e:
        print(f"[{model}] Erreur : {e}")
    return "".join(chunks)


def _call_qwen_streaming(prompt: str, img_b64: str, timeout_no_token: int = 60) -> str:
    import json as _json
    payload = {
        "model": QWEN_MODEL, "stream": True,
        "options": {"temperature": 0.1, "num_predict": 300},
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "keep_alive": -1,
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
        print(f"[Qwen] Timeout lecture ({timeout_no_token}s)")
    except requests.exceptions.ConnectionError:
        print("[Qwen] Ollama non joignable")
    except Exception as e:
        print(f"[Qwen] Erreur : {e}")
    return "".join(chunks)


def image_to_base64(img_path: str, max_size: int = 1024) -> str:
    if img_path.lower().endswith(".pdf"):
        import tempfile as _tf
        from core.ocr import pdf_first_page_to_image
        tmp = pdf_first_page_to_image(img_path)
        try:
            return image_to_base64(tmp, max_size)
        finally:
            import os as _os
            if _os.path.exists(tmp):
                _os.remove(tmp)

    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        r = max_size / max(w, h)
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_classes_description(doc_classes: list) -> str:
    """Construit une description détaillée de chaque classe pour les prompts LLM."""
    lines = []
    for cls in doc_classes:
        desc = DOC_CLASS_DESCRIPTIONS.get(cls, cls.replace("_", " ").title())
        lines.append(f"- {cls}: {desc}")
    return "\n".join(lines)


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


def agent_llm_classification(ocr_text: str, doc_classes: list, context: str = "",
                               model: str = None) -> str | None:
    if not USE_OLLAMA:
        return None

    target_model = model or GEMMA_MODEL
    classes_desc = _build_classes_description(doc_classes)
    prompt = (
        f"Tu es un expert en documents bancaires et administratifs marocains.\n"
        f"Voici les catégories disponibles avec leur description :\n{classes_desc}\n\n"
        f"{'Contexte : ' + context + chr(10) if context else ''}"
        f"Texte OCR du document :\n{ocr_text[:1500]}\n\n"
        f"Analyse le texte et réponds UNIQUEMENT avec le nom exact de la catégorie. "
        f"Si aucune ne correspond : 'inconnu'.\nCatégorie :"
    )

    raw = _call_ollama_streaming(prompt, target_model, timeout_no_token=45)
    if not raw:
        return None

    raw_lower = raw.lower().strip()
    for cls in doc_classes:
        if cls.lower() in raw_lower:
            return cls
    return None


def agent_qwen_vision(img_path: str, ocr_text: str, doc_classes: list,
                      timeout: int = QWEN_TIMEOUT, retry: bool = True) -> dict:
    if not USE_QWEN:
        return {"class": None, "confidence": 0.0, "reasoning": "Qwen désactivé", "available": False}

    classes_desc = _build_classes_description(doc_classes)
    prompt = (
        f"Tu es un expert en documents bancaires et administratifs marocains.\n"
        f"Voici les catégories disponibles avec leur description :\n{classes_desc}\n\n"
        f"Texte OCR (peut être bruité) : {ocr_text[:600]}\n\n"
        f"Analyse attentivement l'image et le texte OCR. "
        f"Réponds UNIQUEMENT en JSON :\n"
        f'{{ "class": "nom_categorie_exact", "confidence": 0.95, "reasoning": "explication courte" }}\n'
        f"Utilise EXACTEMENT l'un des noms de catégorie listés ci-dessus. "
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


def agent_keywords(text: str) -> dict:
    text_lower = text.lower()
    scores, found = {}, {}
    for cls, keywords in KEYWORD_RULES.items():
        hits = [kw for kw in keywords if kw.lower() in text_lower]
        if hits:
            scores[cls] = len(hits)
            found[cls]  = hits

    if re.search(r"n[ée] le \d{1,2}[./-]\d{1,2}[./-]\d{4}", text_lower):
        scores["cin"] = scores.get("cin", 0) + 3
        found.setdefault("cin", []).append("date_naissance_pattern")

    if not scores:
        return {"class": None, "keywords_found": {}, "score": 0}
    best = max(scores, key=scores.get)
    return {"class": best, "keywords_found": found, "score": scores[best]}


def agent_nettoyeur(raw_text: str) -> str:
    cleaned = re.sub(r"[^\w\s\-\.,;:éèêëàâäôöùûüçæœ]", " ", raw_text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else raw_text