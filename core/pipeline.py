"""
Pipeline de classification bancaire — Confirmation double LLM obligatoire.

Logique :
  1. OCR + nettoyage
  2. EfficientNet+mBERT → prédiction initiale
  4. llama3.2 → confirmation ou rejet (OBLIGATOIRE)
  5. Qwen2.5-VL → confirmation ou rejet (OBLIGATOIRE)
  
  Décision finale :
  - Les DEUX confirment → validé automatiquement (needs_manual=False)
  - UN SEUL désaccord ou indisponible → révision humaine obligatoire
  - Les DEUX indisponibles → révision humaine obligatoire

  Principe bancaire : jamais de confiance aveugle dans un seul système.
"""
import re
import time

from core.ocr import extract_text_ocr
from core.model import classify_with_model, DOC_CLASSES, CONFIDENCE_THRESHOLD
from core.agents import (
    agent_llm_classification,
    agent_qwen_vision,
    agent_nettoyeur,
    USE_OLLAMA, USE_QWEN,
)


def _step(steps: list, name: str, status: str, detail: str):
    steps.append({"step": name, "status": status, "detail": detail})


def run_pipeline(img_path: str, original_filename: str = "") -> dict:
    """
    Exécute le pipeline de classification avec double confirmation LLM.
    
    Retourne un dict complet avec prediction, confidence, votes, steps, etc.
    """
    t0 = time.time()
    steps: list = []
    agents_used: list = []

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 1 — OCR bilingue (fr/ar)
    # ══════════════════════════════════════════════════════════════════════
    ocr_text, degraded = extract_text_ocr(img_path)
    _step(steps, "OCR bilingue",
          "warning" if degraded else "ok",
          f"{'⚠ Image dégradée — ' if degraded else ''}{ocr_text[:300]}…")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 2 — Nettoyage texte OCR (regex uniquement, pas de LLM)
    # ══════════════════════════════════════════════════════════════════════
    clean_text = agent_nettoyeur(ocr_text)
    if clean_text != ocr_text:
        _step(steps, "Nettoyage OCR", "ok", "Texte normalisé par regex")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 3 — Modèle EfficientNet+mBERT (prédiction initiale)
    # ══════════════════════════════════════════════════════════════════════
    model_pred, model_conf, all_scores = classify_with_model(img_path, clean_text)
    _step(steps, "Modèle EfficientNet+mBERT", "ok",
          f"→ {model_pred} ({model_conf:.1%})")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 4 — Agent llama3.2 (confirmation LLM texte)
    # ══════════════════════════════════════════════════════════════════════
    llama_class: str | None = None
    llama_available = False

    if USE_OLLAMA:
        llama_class = agent_llm_classification(clean_text, DOC_CLASSES)
        if llama_class:
            llama_available = True
            agents_used.append("llama")
            llama_agrees = llama_class.lower() == model_pred.lower()
            _step(steps, "Agent llama3.2", "ok",
                  f"→ {llama_class} | {'✅ Accord' if llama_agrees else '❌ Désaccord'} avec le modèle")
        else:
            _step(steps, "Agent llama3.2", "warning",
                  "Timeout ou indisponible — confirmation impossible")
    else:
        _step(steps, "Agent llama3.2", "warning",
              "Désactivé — confirmation impossible")

    # ══════════════════════════════════════════════════════════════════════
    # ÉTAPE 5 — Agent Qwen2.5-VL (confirmation vision)
    # Toujours appelé — indépendant de llama (double vérification)
    # ══════════════════════════════════════════════════════════════════════
    qwen_result = {"available": False, "class": None, "confidence": 0.0,
                   "reasoning": "Non appelé"}
    qwen_class: str | None = None
    qwen_conf = 0.0
    qwen_available = False

    if USE_QWEN:
        qwen_result = agent_qwen_vision(img_path, clean_text, DOC_CLASSES)
        if qwen_result.get("available") and qwen_result.get("class"):
            qwen_available = True
            qwen_class = qwen_result["class"]
            qwen_conf = qwen_result["confidence"]
            agents_used.append("qwen")
            qwen_agrees = qwen_class.lower() == model_pred.lower()
            _step(steps, "Agent Qwen2.5-VL", "ok",
                  f"→ {qwen_class} ({qwen_conf:.0%}) | {'✅ Accord' if qwen_agrees else '❌ Désaccord'} avec le modèle")
        else:
            _step(steps, "Agent Qwen2.5-VL", "warning",
                  "Timeout ou indisponible — confirmation impossible")
    else:
        _step(steps, "Agent Qwen2.5-VL", "warning",
              "Désactivé — confirmation impossible")

    # ══════════════════════════════════════════════════════════════════════
    # DÉCISION FINALE — Principe bancaire : double confirmation obligatoire
    # ══════════════════════════════════════════════════════════════════════
    votes = {
        "model": model_pred,
        "llm": llama_class,
        "qwen": qwen_class,
    }

    llama_agrees = llama_available and llama_class.lower() == model_pred.lower()
    qwen_agrees  = qwen_available  and qwen_class.lower()  == model_pred.lower()

    final_prediction = model_pred
    final_confidence = model_conf
    path = ""
    agreement = False
    needs_manual = True  # Par défaut : révision requise (principe de prudence)

    # ─────────────────────────────────────────────────────────────────────
    # CAS A : Les deux confirment → validation automatique
    # ─────────────────────────────────────────────────────────────────────
    if llama_agrees and qwen_agrees:
        agreement = True
        needs_manual = False
        path = "dual_llm_confirmed"
        # Bonus de confiance : accord unanime
        final_confidence = min(model_conf + 0.05, 1.0)
        _step(steps, "✅ Double confirmation LLM", "ok",
              f"llama3.2 + Qwen2.5-VL confirment → {model_pred} ({final_confidence:.0%})")

    # ─────────────────────────────────────────────────────────────────────
    # CAS B : llama confirme mais Qwen désaccorde → révision humaine
    # ─────────────────────────────────────────────────────────────────────
    elif llama_agrees and not qwen_agrees:
        agreement = False
        needs_manual = True
        path = "qwen_disagrees"
        _step(steps, "⚠️ Désaccord Qwen", "warning",
              f"llama3.2 confirme ({model_pred}) mais Qwen → {qwen_class or 'indisponible'} — RÉVISION HUMAINE")

    # ─────────────────────────────────────────────────────────────────────
    # CAS C : Qwen confirme mais llama désaccorde → révision humaine
    # ─────────────────────────────────────────────────────────────────────
    elif qwen_agrees and not llama_agrees:
        agreement = False
        needs_manual = True
        path = "llama_disagrees"
        _step(steps, "⚠️ Désaccord llama3.2", "warning",
              f"Qwen confirme ({model_pred}) mais llama → {llama_class or 'indisponible'} — RÉVISION HUMAINE")

    # ─────────────────────────────────────────────────────────────────────
    # CAS D : Les deux désaccordent → révision humaine + signal fort
    # ─────────────────────────────────────────────────────────────────────
    elif llama_available and qwen_available and not llama_agrees and not qwen_agrees:
        agreement = False
        needs_manual = True
        path = "dual_llm_rejected"
        # Les deux proposent peut-être la même alternative
        if llama_class and qwen_class and llama_class.lower() == qwen_class.lower():
            _step(steps, "🔴 Double rejet — alternative commune", "warning",
                  f"Modèle → {model_pred} | llama + Qwen → {llama_class} — RÉVISION HUMAINE URGENTE")
            # On propose l'alternative comme suggestion mais ne l'impose pas
            votes["suggested_alternative"] = llama_class
        else:
            _step(steps, "🔴 Double rejet — désaccord total", "warning",
                  f"Modèle → {model_pred} | llama → {llama_class} | Qwen → {qwen_class} — RÉVISION HUMAINE URGENTE")

    # ─────────────────────────────────────────────────────────────────────
    # CAS E : Un ou les deux LLM indisponibles → révision humaine
    # ─────────────────────────────────────────────────────────────────────
    else:
        agreement = False
        needs_manual = True
        missing = []
        if not llama_available:
            missing.append("llama3.2")
        if not qwen_available:
            missing.append("Qwen2.5-VL")

        if not missing:
            # Ne devrait pas arriver, mais sécurité
            path = "undetermined"
            _step(steps, "⚠️ Cas non déterminé", "warning",
                  "Vérification manuelle recommandée")
        elif len(missing) == 2:
            path = "no_llm_available"
            _step(steps, "🔴 Aucun LLM disponible", "warning",
                  f"Impossible de confirmer → {model_pred} — RÉVISION HUMAINE OBLIGATOIRE")
        else:
            path = f"{missing[0].replace('.','').replace('-','_')}_unavailable"
            _step(steps, f"⚠️ {missing[0]} indisponible", "warning",
                  f"Confirmation partielle impossible — RÉVISION HUMAINE OBLIGATOIRE")

    # ══════════════════════════════════════════════════════════════════════
    # RÉSUMÉ DÉCISION
    # ══════════════════════════════════════════════════════════════════════
    _step(steps, "Résultat final", "ok" if not needs_manual else "warning",
          f"{'✅ Validé automatiquement' if not needs_manual else '⚠️ Révision humaine requise'} → {final_prediction} ({final_confidence:.1%})")

    time_ms = int((time.time() - t0) * 1000)

    return {
        "prediction": final_prediction,
        "confidence": round(final_confidence * 100, 2),
        "all_scores": all_scores,
        "path": path,
        "steps": steps,
        "qwen": qwen_result,
        "votes": votes,
        "agreement": agreement,
        "needs_manual": needs_manual,
        "degraded_image": degraded,
        "agents_used": agents_used,
        "llama_result": {
            "available": llama_available,
            "class": llama_class,
            "agrees": llama_agrees if llama_available else None,
        },
        "time_ms": time_ms,
        "ocr_text": ocr_text,
    }