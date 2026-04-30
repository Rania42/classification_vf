"""
Pipeline de classification bancaire.
Qwen2.5-VL = modèle principal (vision)
Gemma2:9b  = vérificateur (texte OCR)
EfficientNet+mBERT = fallback dernier recours → toujours révision humaine
"""
import os
import time

from core.ocr import extract_text_ocr, pdf_first_page_to_image
from core.model import classify_with_model, DOC_CLASSES, CONFIDENCE_THRESHOLD
from core.agents import (
    agent_qwen_vision,
    agent_nettoyeur,
    USE_OLLAMA, USE_QWEN,
)

GEMMA_MODEL = "gemma2:9b"


def _ensure_image_path(img_path: str) -> tuple[str, str | None]:
    if img_path.lower().endswith(".pdf"):
        try:
            tmp = pdf_first_page_to_image(img_path)
            return tmp, tmp
        except Exception as e:
            print(f"[Pipeline] PDF→image échoué : {e}")
            return img_path, None
    return img_path, None


def _step(steps: list, name: str, status: str, detail: str):
    steps.append({"step": name, "status": status, "detail": detail})


def _call_gemma_verify(ocr_text: str, predicted_class: str, doc_classes: list) -> dict:
    """Gemma2:9b vérifie la prédiction Qwen via texte OCR."""
    from core.agents import _call_ollama_streaming
    import core.agents as _agents_module

    if not _agents_module.USE_OLLAMA:
        return {"agrees": False, "class": None, "available": False}

    classes_str = ", ".join(doc_classes)
    prompt = (
        f"Tu es un expert en documents bancaires marocains.\n"
        f"Un système d'analyse visuelle a classifié ce document comme : '{predicted_class}'\n"
        f"Catégories possibles : {classes_str}\n\n"
        f"Texte OCR extrait :\n{ocr_text[:1800]}\n\n"
        f"En te basant UNIQUEMENT sur le texte OCR, quelle est la catégorie correcte ?\n"
        f"Réponds UNIQUEMENT avec le nom exact de la catégorie. Si incertain : 'inconnu'.\n"
        f"Catégorie :"
    )

    try:
        raw = _call_ollama_streaming(prompt, GEMMA_MODEL, timeout_no_token=45)
        if not raw:
            return {"agrees": False, "class": None, "available": False}
        raw_lower = raw.lower().strip()
        gemma_class = None
        for cls in doc_classes:
            if cls.lower() in raw_lower:
                gemma_class = cls
                break
        if not gemma_class:
            return {"agrees": False, "class": None, "available": False}
        return {
            "agrees": gemma_class.lower() == predicted_class.lower(),
            "class": gemma_class,
            "available": True,
        }
    except Exception as e:
        print(f"[Gemma2] Erreur : {e}")
        return {"agrees": False, "class": None, "available": False}


def _call_gemma_classify(ocr_text: str, doc_classes: list) -> dict:
    """Gemma2:9b classifie seul quand Qwen est absent."""
    from core.agents import _call_ollama_streaming
    import core.agents as _agents_module

    if not _agents_module.USE_OLLAMA:
        return {"class": None, "available": False}

    classes_str = ", ".join(doc_classes)
    prompt = (
        f"Tu es un expert en documents bancaires marocains.\n"
        f"Catégories disponibles : {classes_str}\n"
        f"Texte OCR du document :\n{ocr_text[:1800]}\n\n"
        f"Identifie la catégorie de ce document. Réponds UNIQUEMENT avec le nom exact. "
        f"Si incertain : 'inconnu'.\nCatégorie :"
    )
    try:
        raw = _call_ollama_streaming(prompt, GEMMA_MODEL, timeout_no_token=45)
        if not raw:
            return {"class": None, "available": False}
        raw_lower = raw.lower().strip()
        for cls in doc_classes:
            if cls.lower() in raw_lower:
                return {"class": cls, "available": True}
        return {"class": None, "available": False}
    except Exception as e:
        print(f"[Gemma2 classify] Erreur : {e}")
        return {"class": None, "available": False}


def run_pipeline(img_path: str, original_filename: str = "") -> dict:
    t0 = time.time()
    steps: list = []
    agents_used: list = []

    # ── ÉTAPE 0 : PDF → image ─────────────────────────────────────────────
    img_path_for_vision, pdf_tmp = _ensure_image_path(img_path)
    if pdf_tmp:
        _step(steps, "Conversion PDF", "ok",
              f"Première page extraite → {os.path.basename(pdf_tmp)}")

    # ── ÉTAPE 1 : OCR ─────────────────────────────────────────────────────
    ocr_text, degraded = extract_text_ocr(img_path)
    _step(steps, "Extraction de texte (OCR)", "warning" if degraded else "ok",
          f"{'Image de qualité réduite — ' if degraded else ''}{len(ocr_text)} caractères extraits")

    clean_text = agent_nettoyeur(ocr_text)

    # ── ÉTAPE 2 : Qwen2.5-VL — analyse visuelle principale ───────────────
    qwen_result = {"available": False, "class": None, "confidence": 0.0, "reasoning": "Non appelé"}
    qwen_class: str | None = None
    qwen_conf = 0.0
    qwen_available = False

    if USE_QWEN:
        qwen_result = agent_qwen_vision(img_path_for_vision, clean_text, DOC_CLASSES)
        if qwen_result.get("available") and qwen_result.get("class"):
            qwen_available = True
            qwen_class = qwen_result["class"]
            qwen_conf = qwen_result["confidence"]
            agents_used.append("qwen")
            _step(steps, "Analyse visuelle (Qwen2.5-VL)", "ok",
                  f"Identifié : {qwen_class} — confiance {qwen_conf:.0%}")
        else:
            _step(steps, "Analyse visuelle (Qwen2.5-VL)", "warning",
                  "Indisponible — passage en mode dégradé")
    else:
        _step(steps, "Analyse visuelle (Qwen2.5-VL)", "warning", "Désactivé")

    # ── ÉTAPE 3 : Gemma2:9b — vérification ou classification de secours ───
    gemma_result = {"agrees": False, "class": None, "available": False}
    gemma_available = False
    gemma_solo = False

    if qwen_available:
        # Gemma vérifie la prédiction Qwen
        gemma_result = _call_gemma_verify(clean_text, qwen_class, DOC_CLASSES)
        if gemma_result["available"]:
            gemma_available = True
            agents_used.append("gemma2")
            if gemma_result["agrees"]:
                _step(steps, "Vérification textuelle (Gemma2:9b)", "ok",
                      f"Confirme : {qwen_class}")
            else:
                _step(steps, "Vérification textuelle (Gemma2:9b)", "warning",
                      f"Désaccord — Gemma identifie : {gemma_result['class'] or 'inconnu'}")
        else:
            _step(steps, "Vérification textuelle (Gemma2:9b)", "warning",
                  "Indisponible")
    else:
        # Qwen absent → Gemma classifie seul
        if USE_OLLAMA:
            gr = _call_gemma_classify(clean_text, DOC_CLASSES)
            if gr["available"]:
                gemma_available = True
                gemma_solo = True
                qwen_class = gr["class"]
                qwen_conf = 0.0
                gemma_result = {"agrees": True, "class": gr["class"], "available": True}
                agents_used.append("gemma2")
                _step(steps, "Classification textuelle (Gemma2:9b)", "warning",
                      f"Analyse visuelle absente — Gemma identifie : {gr['class']}")
            else:
                _step(steps, "Classification textuelle (Gemma2:9b)", "warning",
                      "Indisponible — passage au modèle de secours")

    # ── ÉTAPE 4 : EfficientNet+mBERT — FALLBACK uniquement ───────────────
    model_pred = None
    model_conf = 0.0
    all_scores = []
    used_efficientnet = False

    if not qwen_available and not gemma_available:
        model_pred, model_conf, all_scores = classify_with_model(img_path_for_vision, clean_text)
        used_efficientnet = True
        _step(steps, "Modèle de secours (EfficientNet+mBERT)", "warning",
              f"Résultat : {model_pred} ({model_conf:.0%}) — révision humaine obligatoire")
    else:
        try:
            _, _, all_scores = classify_with_model(img_path_for_vision, clean_text)
        except Exception:
            all_scores = []

    # ── DÉCISION FINALE ────────────────────────────────────────────────────
    final_prediction = qwen_class or model_pred or "inconnu"
    final_confidence = qwen_conf if qwen_available else (model_conf if used_efficientnet else 0.5)
    path = ""
    agreement = False
    needs_manual = True

    if qwen_available and gemma_available and gemma_result["agrees"]:
        agreement = True
        needs_manual = False
        path = "qwen_gemma_confirmed"
        final_confidence = min(qwen_conf + 0.05, 1.0)
        _step(steps, "✅ Résultat validé automatiquement", "ok",
              f"Qwen2.5-VL et Gemma2 confirment : {final_prediction}")

    elif qwen_available and gemma_available and not gemma_result["agrees"]:
        path = "qwen_gemma_disagreement"
        _step(steps, "⚠ Vérification humaine requise", "warning",
              f"Vision ({qwen_class}) et texte ({gemma_result['class']}) ne concordent pas")

    elif qwen_available and not gemma_available:
        path = "qwen_only"
        _step(steps, "⚠ Vérification recommandée", "warning",
              "Analyse visuelle seule — vérificateur textuel indisponible")

    elif not qwen_available and gemma_available and gemma_solo:
        path = "gemma_only"
        final_confidence = 0.5
        _step(steps, "⚠ Vérification requise", "warning",
              "Résultat basé sur le texte uniquement — analyse visuelle indisponible")

    else:
        path = "efficientnet_fallback"
        _step(steps, "⚠ Révision obligatoire", "warning",
              "Mode dégradé — aucun LLM disponible — vérification humaine obligatoire")

    time_ms = int((time.time() - t0) * 1000)

    if pdf_tmp and os.path.exists(pdf_tmp):
        try:
            os.remove(pdf_tmp)
        except Exception:
            pass

    return {
        "prediction": final_prediction,
        "confidence": round(final_confidence * 100, 2),
        "all_scores": all_scores,
        "path": path,
        "steps": steps,
        "qwen": qwen_result,
        "gemma": gemma_result,
        "votes": {
            "qwen": qwen_class,
            "gemma": gemma_result.get("class"),
            "efficientnet": model_pred if used_efficientnet else None,
        },
        "agreement": agreement,
        "needs_manual": needs_manual,
        "degraded_image": degraded,
        "agents_used": agents_used,
        "used_efficientnet": used_efficientnet,
        # Compatibilité avec l'ancienne API (llama_result → gemma)
        "llama_result": {
            "available": gemma_available,
            "class": gemma_result.get("class"),
            "agrees": gemma_result.get("agrees") if gemma_available else None,
        },
        "time_ms": time_ms,
        "ocr_text": ocr_text,
    }