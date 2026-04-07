"""
Pipeline de classification intelligent.
Gère tous les cas : confiance haute/moyenne/faible, agents indisponibles,
désaccords, image dégradée, intervention humaine.
"""
import re
import time
from collections import Counter

from config import CONF_HIGH, CONF_MEDIUM
from core.ocr import extract_text_ocr
from core.model import classify_with_model, DOC_CLASSES, CONFIDENCE_THRESHOLD
from core.agents import (
    agent_llm_classification,
    agent_qwen_vision,
    agent_keywords,
    agent_nettoyeur,
    USE_OLLAMA, USE_QWEN,
)


def _step(steps: list, name: str, status: str, detail: str):
    steps.append({"step": name, "status": status, "detail": detail})


def _cin_override(text: str, pred: str, conf: float) -> tuple[str, float, bool]:
    """Règle métier prioritaire : CIN détecté via patterns."""
    t = text.lower()
    if re.search(r"carte nationale|cni|cin", t):
        if re.search(r"n[ée] le \d{1,2}[./-]\d{1,2}[./-]\d{4}", t):
            if conf < 0.8:
                return "carte_identite", 0.85, True
    return pred, conf, False


def run_pipeline(img_path: str, original_filename: str = "") -> dict:
    """
    Exécute le pipeline complet sur une image.
    Retourne le dictionnaire résultat prêt pour JSON.
    """
    t0    = time.time()
    steps = []
    agents_used = []

    # ══════════════════════════════════════════════════
    # ÉTAPE 1 — OCR bilingue + détection image dégradée
    # ══════════════════════════════════════════════════
    ocr_text, degraded = extract_text_ocr(img_path)
    ocr_status = "warning" if degraded else "ok"
    _step(steps, "OCR bilingue",
          ocr_status,
          f"{'⚠ Image dégradée — prétraitement appliqué — ' if degraded else ''}"
          f"{ocr_text[:300]}{'…' if len(ocr_text) > 300 else ''}")

    # ══════════════════════════════════════════════════
    # ÉTAPE 2 — Nettoyage OCR
    # ══════════════════════════════════════════════════
    clean_text = agent_nettoyeur(ocr_text)
    if clean_text != ocr_text:
        _step(steps, "Agent Nettoyeur", "ok", "Texte normalisé")

    # ══════════════════════════════════════════════════
    # ÉTAPE 3 — Modèle EfficientNet+mBERT (TOUJOURS appelé)
    # ══════════════════════════════════════════════════
    pred_class, confidence, all_scores = classify_with_model(img_path, clean_text)
    _step(steps, "Modèle multimodal (EfficientNet+mBERT)", "ok",
          f"{pred_class} — confiance {confidence:.1%}")

    # Règle métier CIN
    pred_class, confidence, cin_overridden = _cin_override(clean_text, pred_class, confidence)
    if cin_overridden:
        _step(steps, "Règle métier CIN", "warning",
              "Carte d'identité détectée par pattern — correction prioritaire")

    # ══════════════════════════════════════════════════
    # ÉTAPE 4 — Appel Qwen2.5-VL (si disponible)
    # ══════════════════════════════════════════════════
    qwen_result = agent_qwen_vision(img_path, clean_text, DOC_CLASSES)
    if qwen_result["available"]:
        agents_used.append("qwen")
        status_q = "ok" if qwen_result["class"] else "warning"
        _step(steps, "Qwen2.5-VL (vision multimodale)", status_q,
              f"Classe : {qwen_result['class'] or 'non déterminé'}  |  "
              f"Confiance : {qwen_result['confidence']:.0%}  |  "
              f"{qwen_result['reasoning'][:120]}")
    else:
        _step(steps, "Qwen2.5-VL", "warning",
              qwen_result.get("reasoning", "Non disponible"))

    qwen_class   = qwen_result.get("class")
    qwen_conf    = qwen_result.get("confidence", 0.0)
    qwen_agrees  = qwen_class and qwen_class.lower() == pred_class.lower()
    qwen_avail   = qwen_result["available"]

    # ══════════════════════════════════════════════════
    # DÉCISION SELON NIVEAU DE CONFIANCE MODÈLE
    # ══════════════════════════════════════════════════

    final_prediction = pred_class
    final_confidence = confidence
    final_scores     = all_scores
    path             = "direct"
    votes            = {}
    agreement        = True
    needs_manual     = False

    # ──────────────────────────────────────────────────
    # CAS A : HAUTE CONFIANCE (≥ CONF_HIGH = 80%)
    # ──────────────────────────────────────────────────
    if confidence >= CONF_HIGH:

        if qwen_avail and qwen_agrees:
            # A1 : Accord Modèle + Qwen → DIRECT VALIDÉ ✅
            _step(steps, "Validation croisée ✅", "ok",
                  f"Accord Modèle ({confidence:.0%}) + Qwen ({qwen_conf:.0%}) → {pred_class}")
            path = "direct_qwen_validated"

        elif qwen_avail and not qwen_agrees and qwen_class:
            # A2 : Désaccord Modèle ↔ Qwen → ARBITRAGE RENFORCÉ
            _step(steps, "Désaccord Modèle ↔ Qwen ⚠", "warning",
                  f"Modèle → {pred_class} ({confidence:.0%})  |  Qwen → {qwen_class} ({qwen_conf:.0%})")

            # Appel llama3.2 comme arbitre (VRAI appel Ollama)
            llama_class = agent_llm_classification(
                clean_text, DOC_CLASSES,
                context=f"Le modèle dit '{pred_class}', Qwen dit '{qwen_class}' — arbitre."
            )
            if llama_class:
                agents_used.append("llama")

            kw = agent_keywords(clean_text)
            agents_used.append("keywords")

            votes = {"model": pred_class, "qwen": qwen_class,
                     "llm": llama_class, "keywords": kw["class"]}

            if llama_class == pred_class:
                final_prediction = pred_class
                final_confidence = max(confidence, qwen_conf)
                _step(steps, "Arbitrage llama3.2", "ok",
                      f"llama confirme le modèle → {pred_class}")
                path = "arbitrage_model_wins"
            elif llama_class == qwen_class:
                final_prediction = qwen_class
                final_confidence = max(confidence, qwen_conf)
                _step(steps, "Arbitrage llama3.2", "ok",
                      f"llama confirme Qwen → {qwen_class}")
                path = "arbitrage_qwen_wins"
            elif confidence >= 0.85:
                # Modèle très confiant → priorité modèle
                final_prediction = pred_class
                _step(steps, "Décision confiance élevée", "ok",
                      f"Modèle prioritaire (conf {confidence:.0%} ≥ 85%)")
                path = "high_conf_model_priority"
            elif qwen_conf >= 0.80:
                final_prediction = qwen_class
                final_confidence = qwen_conf
                _step(steps, "Décision confiance Qwen", "ok",
                      f"Qwen prioritaire (conf {qwen_conf:.0%} ≥ 80%)")
                path = "high_conf_qwen_priority"
            else:
                # llama donne une 3e classe ou est indispo → MANUEL
                final_prediction = pred_class  # meilleure supposition
                needs_manual = True
                _step(steps, "Désaccord non résolu ⚠", "warning",
                      f"llama → {llama_class or 'indispo'} | RÉVISION MANUELLE")
                path = "qwen_disagreement"

        elif not qwen_avail:
            # A3 : Qwen indisponible — modèle seul à haute confiance
            if confidence >= 0.85:
                _step(steps, "Voie directe (sans Qwen)", "ok",
                      f"Modèle confiant à {confidence:.0%} — Qwen indisponible")
                path = "direct_no_qwen"
            else:
                # 80-85% sans Qwen → on consulte llama pour sécurité
                llama_class = agent_llm_classification(clean_text, DOC_CLASSES)
                if llama_class:
                    agents_used.append("llama")
                kw = agent_keywords(clean_text)
                agents_used.append("keywords")
                votes = {"model": pred_class, "llm": llama_class, "keywords": kw["class"]}

                if llama_class and llama_class != pred_class:
                    vote_list = [pred_class, pred_class, llama_class]
                    if kw["class"]:
                        vote_list.append(kw["class"])
                    cnt = Counter(vote_list)
                    final_prediction, top = cnt.most_common(1)[0]
                    agreement = top >= 2
                    if not agreement:
                        needs_manual = True
                    _step(steps, "Vote Modèle+llama (Qwen absent)", "ok" if agreement else "warning",
                          f"→ {final_prediction} | accord={agreement}")
                else:
                    _step(steps, "llama confirme le modèle", "ok",
                          f"→ {pred_class}")
                path = "medium_agents"

    # ──────────────────────────────────────────────────
    # CAS B : CONFIANCE MOYENNE (60-80%)
    # ──────────────────────────────────────────────────
    elif confidence >= CONF_MEDIUM:
        _step(steps, "Confiance moyenne — agents activés", "warning",
              f"{confidence:.1%} — consultation llama3.2 + keywords")

        # Toujours appeler llama (pas juste keywords)
        llama_class = agent_llm_classification(clean_text, DOC_CLASSES)
        if llama_class:
            agents_used.append("llama")
        kw = agent_keywords(clean_text)
        agents_used.append("keywords")

        votes = {"model": pred_class, "qwen": qwen_class,
                 "llm": llama_class, "keywords": kw["class"]}

        # Vote pondéré : Qwen×2, Modèle×1.5→2 votes, llama×1, kw×1
        vote_list = [pred_class, pred_class]          # modèle ×2
        if qwen_class:
            vote_list += [qwen_class, qwen_class]     # qwen ×2
        if llama_class:
            vote_list.append(llama_class)             # llama ×1
        if kw["class"]:
            vote_list.append(kw["class"])             # kw ×1

        cnt = Counter(vote_list)
        final_prediction, top_votes = cnt.most_common(1)[0]
        agreement = top_votes >= 3   # majorité sur ≥3 votes (plus strict)
        final_confidence = max(confidence, qwen_conf)

        _step(steps, "Vote pondéré (Modèle×2 + Qwen×2 + llama + kw)", 
              "ok" if agreement else "warning",
              f"→ {final_prediction} | {top_votes}/{len(vote_list)} votes | accord={agreement}")

        if not agreement:
            needs_manual = True
            _step(steps, "Désaccord agents ⚠", "warning",
                  "Révision manuelle recommandée (accord insuffisant)")
        path = "medium_agents"

    # ──────────────────────────────────────────────────
    # CAS C : FAIBLE CONFIANCE (< 60%)
    # ──────────────────────────────────────────────────
    else:
        _step(steps, f"Confiance faible ({confidence:.1%}) — pipeline complet", "warning",
              "Tous les agents activés")

        # Si image dégradée → re-run modèle sur texte brut (avant nettoyage)
        if degraded and ocr_text != clean_text:
            pred2, conf2, scores2 = classify_with_model(img_path, ocr_text)
            if conf2 > confidence:
                pred_class, confidence, all_scores = pred2, conf2, scores2
                _step(steps, "Reclassification (OCR brut)", "ok",
                      f"Meilleur résultat avec OCR brut : {pred2} ({conf2:.1%})")

        # Appel llama (VRAI appel Ollama)
        llama_class = agent_llm_classification(clean_text, DOC_CLASSES)
        if llama_class:
            agents_used.append("llama")
        _step(steps, "Agent llama3.2 (classification)", 
              "ok" if llama_class else "warning",
              f"→ {llama_class or 'indisponible / timeout'}")

        # Keywords
        kw = agent_keywords(clean_text)
        agents_used.append("keywords")
        _step(steps, "Agent mots-clés", "ok" if kw["class"] else "info",
              f"Classe : {kw['class'] or 'aucune'}  |  Score : {kw['score']}")

        votes = {"model": pred_class, "qwen": qwen_class,
                 "llm": llama_class, "keywords": kw["class"]}

        # Vote pondéré complet (Qwen×2, reste ×1 chacun)
        vote_list = [pred_class]
        if qwen_class:
            vote_list += [qwen_class, qwen_class]
        if llama_class:
            vote_list.append(llama_class)
        if kw["class"]:
            vote_list.append(kw["class"])

        if not vote_list:
            # Aucun agent n'a répondu
            final_prediction = "inconnu"
            needs_manual = True
            _step(steps, "Aucun agent disponible ⚠", "warning",
                  "Tous les agents sont indisponibles — classification manuelle obligatoire")
            path = "no_agents"
        else:
            cnt = Counter(vote_list)
            final_prediction, top_votes = cnt.most_common(1)[0]
            agreement = top_votes >= 2
            final_confidence = max(confidence, qwen_conf)
            final_scores = all_scores

            _step(steps, "Vote final (Modèle + Qwen×2 + llama + kw)",
                  "ok" if agreement else "warning",
                  f"→ {final_prediction} | {top_votes}/{len(vote_list)} votes | accord={agreement}")

            if not agreement:
                needs_manual = True
                _step(steps, "Désaccord fort ⚠", "warning",
                      "Classification manuelle obligatoire")
            path = "agents_full"

    # ══════════════════════════════════════════════════
    # RÉSULTAT FINAL
    # ══════════════════════════════════════════════════
    time_ms = int((time.time() - t0) * 1000)

    return {
        "prediction":    final_prediction,
        "confidence":    round(final_confidence * 100, 2),
        "all_scores":    final_scores,
        "path":          path,
        "steps":         steps,
        "qwen":          qwen_result,
        "votes":         votes,
        "agreement":     agreement,
        "needs_manual":  needs_manual,
        "degraded_image": degraded,
        "agents_used":   agents_used,
        "time_ms":       time_ms,
        "ocr_text":      ocr_text,
    }