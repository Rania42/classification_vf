import sys
from PIL import Image
import io
import base64
import requests
import re
import json

OLLAMA_CHAT = "http://localhost:11434/api/chat"
QWEN_MODEL  = "qwen2.5vl:3b"
QWEN_TIMEOUT = 120

DOC_CLASSES = [
    "acte_heredite", "rib", "cheque", "tableau_amortissement",
    "acte_naissance", "assurance", "attestation_solde", "carte_identite"
]

def image_to_base64(img_path, max_size=1024):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def test_qwen(img_path):
    print(f"\n[Qwen] Analyse de : {img_path}")
    print("[Qwen] Encodage image...")
    img_b64 = image_to_base64(img_path)

    classes_str = "\n".join(f"- {c}" for c in DOC_CLASSES)
    prompt = f"""Tu es un expert en classification de documents bancaires marocains.
Analyse cette image et réponds UNIQUEMENT avec ce JSON :
{{"class": "nom_categorie", "confidence": 0.95, "reasoning": "explication courte"}}

CATÉGORIES :
{classes_str}

Si aucune catégorie ne correspond, utilise "inconnu"."""

    payload = {
        "model": QWEN_MODEL,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
    }

    print("[Qwen] Envoi à Ollama... (patience, jusqu'à 2 min sur CPU)")
    try:
        resp = requests.post(OLLAMA_CHAT, json=payload, timeout=QWEN_TIMEOUT)
        if resp.status_code != 200:
            print(f"[Qwen] Erreur HTTP {resp.status_code}")
            return
        raw = resp.json().get("message", {}).get("content", "").strip()
        print(f"\n[Qwen] Réponse brute :\n{raw}")

        # Parser JSON
        raw_clean = re.sub(r"```(?:json)?\s*", "", raw).strip()
        try:
            data = json.loads(raw_clean)
        except:
            m = re.search(r'\{[^{}]*"class"[^{}]*\}', raw_clean, re.DOTALL)
            data = json.loads(m.group()) if m else {}

        print(f"\n{'='*50}")
        print(f"  Classe      : {data.get('class', '—')}")
        print(f"  Confiance   : {float(data.get('confidence', 0))*100:.0f}%")
        print(f"  Raisonnement: {data.get('reasoning', '—')}")
        print(f"{'='*50}\n")

    except requests.exceptions.Timeout:
        print(f"[Qwen] Timeout ({QWEN_TIMEOUT}s) — modèle trop lent sur CPU")
    except Exception as e:
        print(f"[Qwen] Erreur : {e}")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else None
    if not img:
        print("Usage : python test_qwen_only.py chemin\\image.jpg")
        sys.exit(1)
    test_qwen(img)
