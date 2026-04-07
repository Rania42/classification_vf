"""
OCR bilingue + évaluation qualité image + prétraitement si dégradée.
"""
import re
import torch
import easyocr
from PIL import Image, ImageFilter, ImageEnhance

from config import DEVICE, OCR_QUALITY_THRESHOLD

# Init readers une seule fois
ocr_latin  = easyocr.Reader(["fr", "en"], gpu=torch.cuda.is_available(), verbose=False)
ocr_arabic = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False)
print("[INIT] OCR prêt")


def _ocr_quality_score(text: str) -> float:
    """
    Ratio caractères alphanumériques / longueur totale.
    Score < OCR_QUALITY_THRESHOLD → image considérée dégradée.
    """
    if not text or len(text) < 5:
        return 0.0
    alphanum = sum(1 for c in text if c.isalnum() or c.isspace())
    return alphanum / len(text)


def _preprocess_degraded(img_path: str) -> str:
    """
    Prétraitement PIL pour images dégradées :
    contrast, sharpen, binarisation.
    Sauvegarde une version temporaire et retourne son chemin.
    """
    import tempfile, os
    img = Image.open(img_path).convert("L")          # Niveaux de gris
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.MedianFilter(3))
    # Binarisation simple (Otsu approché)
    img = img.point(lambda x: 255 if x > 128 else 0, '1').convert("RGB")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    return tmp.name


def extract_text_ocr(img_path: str, force_preprocess: bool = False) -> tuple[str, bool]:
    """
    Extrait le texte OCR complet.
    Retourne (texte, image_degradee).
    
    - Tente d'abord le latin+fr+en
    - Si < 30 chars → ajoute arabe
    - Évalue la qualité → si dégradée, prétraite et re-OCR
    - Retourne texte complet (PAS tronqué — troncature faite dans model.py)
    """
    import os

    tmp_path = None
    degraded = False

    def _run_ocr(path: str) -> str:
        text_latin = " ".join(ocr_latin.readtext(path, detail=0, paragraph=True))
        if len(text_latin.strip()) < 30:
            text_arabic = " ".join(ocr_arabic.readtext(path, detail=0, paragraph=True))
            return (text_latin + " " + text_arabic).strip()
        return text_latin

    text = _run_ocr(img_path)
    quality = _ocr_quality_score(text)

    if quality < OCR_QUALITY_THRESHOLD or force_preprocess or len(text.strip()) < 20:
        degraded = True
        try:
            tmp_path = _preprocess_degraded(img_path)
            text2    = _run_ocr(tmp_path)
            # Garder le meilleur des deux
            if len(text2.strip()) > len(text.strip()):
                text = text2
        except Exception as e:
            print(f"[OCR] Prétraitement échoué : {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    return (text or "[NO_TEXT]"), degraded