"""
OCR bilingue + évaluation qualité image + prétraitement si dégradée.
Support PDF : première page convertie en image avant traitement.
"""
import re
import os
import tempfile
import torch
import numpy as np
import easyocr
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

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


def _otsu_threshold(gray_np: np.ndarray) -> int:
    """Calcule le seuil de binarisation Otsu réel."""
    hist, _ = np.histogram(gray_np.flatten(), bins=256, range=(0, 256))
    total = gray_np.size
    sum_total = np.dot(np.arange(256), hist)
    sum_bg, w_bg, mean_max, thresh = 0.0, 0, 0.0, 128
    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > mean_max:
            mean_max = var
            thresh = t
    return thresh


def _upscale_if_small(img: Image.Image, min_dim: int = 1200) -> Image.Image:
    """Upscale si l'image est trop petite pour un bon OCR."""
    w, h = img.size
    if min(w, h) < min_dim:
        scale = min_dim / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def _deskew(img: Image.Image) -> Image.Image:
    """Correction de l'inclinaison basique via détection de l'angle dominant."""
    try:
        gray = np.array(img.convert("L"))
        # Utiliser les colonnes de pixels sombres pour estimer l'angle
        # Approche simple : projections horizontales pour détecter rotation
        # On tente des angles de -5 à +5 degrés et on garde la variance max
        best_angle, best_var = 0.0, 0.0
        for angle in np.arange(-5, 5.5, 0.5):
            rotated = img.rotate(angle, expand=False, fillcolor=255)
            arr = np.array(rotated.convert("L"))
            row_sums = arr.sum(axis=1).astype(float)
            var = float(np.var(row_sums))
            if var > best_var:
                best_var = var
                best_angle = angle
        if abs(best_angle) > 0.3:
            img = img.rotate(best_angle, expand=True, fillcolor=255)
    except Exception:
        pass
    return img


def _preprocess_degraded(img_path: str) -> str:
    """
    Pipeline de prétraitement robuste pour images dégradées :
    1. Upscale si trop petite
    2. Conversion niveaux de gris
    3. Correction gamma adaptative
    4. Débruitage (MedianFilter)
    5. Amélioration contraste (CLAHE approché via Equalize)
    6. Sharpen
    7. Binarisation Otsu réelle
    8. Deskew
    Retourne le chemin d'une image temporaire.
    """
    img = Image.open(img_path).convert("RGB")

    # 1. Upscale si petite résolution
    img = _upscale_if_small(img, min_dim=1400)

    # 2. Niveaux de gris
    gray = img.convert("L")

    # 3. Correction gamma — éclaircit les images sous-exposées
    arr = np.array(gray, dtype=np.float32)
    mean_lum = arr.mean()
    if mean_lum < 100:          # image sombre → gamma < 1 pour éclaircir
        gamma = max(0.4, mean_lum / 128)
        arr = 255 * (arr / 255) ** gamma
        gray = Image.fromarray(arr.astype(np.uint8))
    elif mean_lum > 200:        # image surexposée → gamma > 1 pour assombrir
        gamma = 1.5
        arr = 255 * (arr / 255) ** gamma
        gray = Image.fromarray(arr.astype(np.uint8))

    # 4. Débruitage médian
    gray = gray.filter(ImageFilter.MedianFilter(3))

    # 5. Égalisation histogramme (CLAHE approché)
    gray = ImageOps.equalize(gray)

    # 6. Boost contraste
    gray = ImageEnhance.Contrast(gray).enhance(2.0)

    # 7. Sharpen
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))

    # 8. Binarisation Otsu réelle
    arr_np = np.array(gray)
    thresh = _otsu_threshold(arr_np)
    binary = gray.point(lambda x: 255 if x > thresh else 0, '1').convert("RGB")

    # 9. Deskew sur l'image binaire
    binary = _deskew(binary)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    binary.save(tmp.name, dpi=(300, 300))
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