import io
import os
from typing import List

from PIL import Image

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None

try:
    import docx
except Exception:
    docx = None

from .utils import clean_text


def extract_text_from_image_bytes(img_bytes: bytes) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is required for image OCR. Install pytesseract and Tesseract binary.")
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    text = pytesseract.image_to_string(image)
    return clean_text(text)


def extract_text_from_docx_bytes(b: bytes) -> str:
    if docx is None:
        raise RuntimeError("python-docx is required for DOCX parsing")
    from io import BytesIO

    doc = docx.Document(BytesIO(b))
    paragraphs = [p.text for p in doc.paragraphs]
    return clean_text('\n'.join(paragraphs))


def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_chunks: List[str] = []
    # try pdfplumber first for text-based PDFs
    if pdfplumber is not None:
        try:
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_chunks.append(t)
        except Exception:
            pass

    # if no text obtained and pdf2image available, fallback to OCR
    if not text_chunks and convert_from_bytes is not None and pytesseract is not None:
        images = convert_from_bytes(b)
        for im in images:
            t = pytesseract.image_to_string(im)
            if t:
                text_chunks.append(t)

    if not text_chunks:
        # last resort: return empty string
        return ""

    return clean_text('\n'.join(text_chunks))


def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        s = b.decode('utf-8')
    except Exception:
        s = b.decode('latin-1')
    return clean_text(s)


def extract_text_from_uploaded_file(uploaded) -> str:
    """uploaded is a Streamlit UploadedFile-like object with .name and .getvalue()."""
    name = uploaded.name.lower()
    b = uploaded.getvalue()
    if name.endswith('.pdf'):
        return extract_text_from_pdf_bytes(b)
    if name.endswith('.docx') or name.endswith('.doc'):
        return extract_text_from_docx_bytes(b)
    if name.endswith('.txt'):
        return extract_text_from_txt_bytes(b)
    # try images
    if any(name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']):
        return extract_text_from_image_bytes(b)
    # fallback: try to decode as text
    return extract_text_from_txt_bytes(b)
