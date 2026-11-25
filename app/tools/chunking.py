from io import BytesIO
from pathlib import Path
import re

from pypdf import PdfReader


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """
    Extracts clean text from a PDF file path or raw bytes (for FastAPI uploads),
    inserting [Page N] markers so quotes can be traced back to exact pages.
    """
    if isinstance(source, bytes):
        reader = PdfReader(BytesIO(source))
    else:
        reader = PdfReader(str(source))

    pages: list[str] = []
    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_page = raw_text.replace("\x00", "").strip()
        if cleaned_page:
            pages.append(f"[Page {page_num}]\n{cleaned_page}")

    return "\n\n".join(pages)


def _normalize_text(text: str) -> str:
    """Cleans null bytes and excessive blank lines while keeping paragraph breaks."""
    cleaned = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse spaces/tabs inside lines, but preserve newlines
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    # Collapse 3+ blank lines into a clean paragraph break
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Splits text into overlapping chunks without slicing words or numbers in half.
    Prefers splitting on paragraph breaks, then sentence endings, then spaces.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = _normalize_text(text)
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(cleaned)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # If we aren't at the very end, snap backwards to a natural text boundary
        if end < text_len:
            min_break = start + (chunk_size // 2)
            break_pos = -1
            for sep in ("\n\n", ". ", "? ", "! ", "\n", " "):
                pos = cleaned.rfind(sep, min_break, end)
                if pos != -1:
                    break_pos = pos + len(sep)
                    break
            if break_pos != -1:
                end = break_pos

        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= text_len:
            break

        next_start = end - chunk_overlap
        # Guarantee forward progress so the loop can never get stuck
        start = next_start if next_start > start else end

    return chunks