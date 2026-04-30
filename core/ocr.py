"""
OCR bilingue + évaluation qualité image + prétraitement si dégradée.
Support PDF : première page convertie en image avant traitement.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import re
import os
import tempfile
import torch
import easyocr
from PIL import Image, ImageFilter, ImageEnhance

from config import DEVICE, OCR_QUALITY_THRESHOLD


def pdf_first_page_to_image(pdf_path: str) -> str:
    """
    Convertit la première page d'un PDF en image PNG temporaire.
    Retourne le chemin de l'image temporaire.
    Essaie pdf2image (poppler) puis pymupdf (fitz) en fallback.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Tentative 1 : pdf2image (poppler)
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
        if pages:
            pages[0].save(tmp_path, "PNG")
            return tmp_path
    except Exception as e:
        print(f"[PDF] pdf2image échoué : {e} — essai pymupdf")

    # Tentative 2 : pymupdf (fitz)
    try:
        import fitz
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom ≈ 144 dpi
        pix = page.get_pixmap(matrix=mat)
        pix.save(tmp_path)
        doc.close()
        return tmp_path
    except Exception as e:
        print(f"[PDF] pymupdf échoué : {e}")

    raise RuntimeError("Impossible de convertir le PDF (installer pdf2image+poppler ou pymupdf)")

# Init readers une seule fois
try:
    ocr_latin  = easyocr.Reader(["fr", "en"], gpu=torch.cuda.is_available(), verbose=False)
    ocr_arabic = easyocr.Reader(["ar", "en"], gpu=torch.cuda.is_available(), verbose=False)
    print("[INIT] OCR prêt")
except Exception as e:
    print(f"[INIT] Erreur initialisation OCR: {e}")
    print("[INIT] Vérifiez votre connexion internet ou les modèles EasyOCR")
    raise


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

    Si img_path est un PDF, la première page est convertie en image avant traitement.
    - Tente d'abord le latin+fr+en
    - Si < 30 chars → ajoute arabe
    - Évalue la qualité → si dégradée, prétraite et re-OCR
    - Retourne texte complet (PAS tronqué — troncature faite dans model.py)
    """
    pdf_tmp = None
    degraded = False

    # Conversion PDF → image (première page)
    if img_path.lower().endswith(".pdf"):
        try:
            pdf_tmp = pdf_first_page_to_image(img_path)
            img_path = pdf_tmp
        except Exception as e:
            print(f"[OCR] Conversion PDF échouée : {e}")
            return "[NO_TEXT]", True

    tmp_path = None

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

    # Nettoyage image temporaire PDF
    if pdf_tmp and os.path.exists(pdf_tmp):
        try:
            os.remove(pdf_tmp)
        except Exception:
            pass

    return (text or "[NO_TEXT]"), degraded