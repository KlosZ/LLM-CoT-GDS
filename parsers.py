"""
Текстовые парсеры для входных материалов (преподаватель) и работ студентов.

Поддерживаемые форматы (без OCR):
- PDF (через PyMuPDF/fitz или pypdf - что установлено; нет ничего - исключение)
- DOCX (python-docx)
- TXT/MD/CSV/TSV/JSON/PY и прочие "текстовые" расширения
- IPYNB (извлекает markdown + code cells)
- ZIP (распаковывает и извлекает текст из поддерживаемых файлов внутри)
"""

from __future__ import annotations

import io
import json
import mimetypes
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

JsonDict = Dict[str, Any]


# Data classes


@dataclass(frozen=True)
class ExtractResult:
    text: str
    meta: JsonDict


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    meta: JsonDict


# Public API


def extract_text_from_path(path: str) -> ExtractResult:
    """Extract text from a filesystem path."""
    with open(path, "rb") as f:
        data = f.read()
    filename = os.path.basename(path)
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return extract_text_from_bytes(filename, data, mime=mime)


def extract_text_from_streamlit(uploaded_file: Any) -> ExtractResult:
    """
    Streamlit UploadedFile wrapper.

    uploaded_file expected to have:
    - uploaded_file.name (str)
    - uploaded_file.type (str mime) (optional)
    - uploaded_file.getvalue() or uploaded_file.read()
    """
    name = getattr(uploaded_file, "name", "upload.bin")
    mime = getattr(uploaded_file, "type", None) or mimetypes.guess_type(name)[0] or "application/octet-stream"
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    else:
        data = uploaded_file.read()
    return extract_text_from_bytes(name, data, mime=mime)


def extract_text_from_bytes(filename: str, data: bytes, *, mime: Optional[str] = None) -> ExtractResult:
    """
    Main dispatcher: extract text from raw bytes based on extension/mime.
    Returns ExtractResult(text, meta).
    """
    mime = mime or mimetypes.guess_type(filename)[0] or "application/octet-stream"
    ext = (os.path.splitext(filename)[1] or "").lower()

    # ZIP: many files
    if ext == ".zip" or mime in ("application/zip", "application/x-zip-compressed"):
        return _extract_from_zip(filename, data)

    # PDF
    if ext == ".pdf" or mime == "application/pdf":
        text, meta = _extract_pdf(data)
        return ExtractResult(text=_normalize_text(text), meta={**meta, "source_filename": filename, "mime": mime})

    # DOCX
    if ext == ".docx" or mime in (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
    ):
        text = _extract_docx(data)
        return ExtractResult(text=_normalize_text(text), meta={"source_filename": filename, "mime": mime})

    # IPYNB
    if ext == ".ipynb" or mime in ("application/x-ipynb+json", "application/json"):
        # JSON может быть и не ноутбуком - проверим по структуре
        try:
            nb = json.loads(data.decode("utf-8", errors="replace"))
            if isinstance(nb, dict) and "cells" in nb and "nbformat" in nb:
                text = _extract_ipynb(nb)
                return ExtractResult(text=_normalize_text(text), meta={"source_filename": filename, "mime": mime})
        except Exception:
            pass  # продолжим как обычный текст/JSON

    # "Текстовые" файлы по расширению или mime
    if _is_likely_text(ext, mime):
        text = _decode_text(data)
        # Если JSON - можно красиво распечатать
        if ext == ".json":
            text = _pretty_json(text)
        return ExtractResult(text=_normalize_text(text), meta={"source_filename": filename, "mime": mime})

    # Неподдерживаемое
    raise ValueError(
        f"Unsupported file type for text extraction: filename={filename!r}, ext={ext!r}, mime={mime!r}. "
        "Supported: pdf, docx, txt/md/csv/tsv/json/py, ipynb, zip."
    )


def split_to_chunks(
        text: str,
        *,
        chunk_size: int = 1400,
        overlap: int = 200,
        source_meta: Optional[JsonDict] = None,
        chunk_id_prefix: str = "chunk"
) -> List[Chunk]:
    """
    Split text into overlapping chunks for RAG/indexing.

    - chunk_size: target size in characters
    - overlap: characters to overlap between chunks
    """
    src = source_meta or {}
    clean = _normalize_text(text)

    if not clean.strip():
        return []

    # First split by paragraphs (double newline), then fallback to sentences/spaces if needed
    paras = [p.strip() for p in clean.split("\n\n") if p.strip()]
    pieces: List[str] = []
    for p in paras:
        if len(p) <= chunk_size:
            pieces.append(p)
        else:
            pieces.extend(_split_long_text(p, chunk_size))

    chunks: List[Chunk] = []
    buf = ""
    idx = 0

    def _flush(b: str) -> None:
        nonlocal idx
        b = b.strip()
        if not b:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{chunk_id_prefix}_{idx}",
                text=b,
                meta={**src, "chunk_index": idx, "char_len": len(b)},
            )
        )
        idx += 1

    for piece in pieces:
        if not buf:
            buf = piece
            continue

        if len(buf) + 2 + len(piece) <= chunk_size:
            buf = buf + "\n\n" + piece
        else:
            _flush(buf)
            # overlap: keep tail of previous buffer
            if overlap > 0:
                tail = buf[-overlap:]
                buf = (tail + "\n\n" + piece).strip()
            else:
                buf = piece

    _flush(buf)
    return chunks


# ZIP extraction


def _extract_from_zip(zip_name: str, data: bytes) -> ExtractResult:
    """
    Extract text from many supported files inside ZIP.
    Safety limits:
    - max_files
    - max_total_uncompressed_bytes
    - per_file_limit
    """
    MAX_FILES = 60
    MAX_TOTAL = 20 * 1024 * 1024  # 20MB total extracted bytes
    PER_FILE = 5 * 1024 * 1024  # 5MB per file

    total = 0
    extracted_texts: List[str] = []
    manifest: List[JsonDict] = []

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        names = names[:MAX_FILES]

        for n in names:
            info = zf.getinfo(n)
            # basic zip bomb mitigation
            if info.file_size > PER_FILE:
                manifest.append({"file": n, "skipped": True, "reason": "file_too_large"})
                continue
            if total + info.file_size > MAX_TOTAL:
                manifest.append({"file": n, "skipped": True, "reason": "total_limit_reached"})
                continue

            total += info.file_size
            raw = zf.read(n)
            base = os.path.basename(n)
            ext = (os.path.splitext(base)[1] or "").lower()
            mime = mimetypes.guess_type(base)[0] or "application/octet-stream"

            # only attempt supported-ish
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                manifest.append({"file": n, "skipped": True, "reason": "image_not_supported"})
                continue

            try:
                res = extract_text_from_bytes(base, raw, mime=mime)
                if res.text.strip():
                    extracted_texts.append(f"===== FILE: {n} =====\n{res.text}")
                    manifest.append({"file": n, "skipped": False, "char_len": len(res.text)})
                else:
                    manifest.append({"file": n, "skipped": True, "reason": "empty_text"})
            except Exception as e:
                manifest.append({"file": n, "skipped": True, "reason": f"parse_error: {type(e).__name__}"})

    final_text = "\n\n".join(extracted_texts)
    meta = {
        "source_filename": zip_name,
        "mime": "application/zip",
        "zip_manifest": manifest,
        "zip_files_considered": len(manifest),
    }
    return ExtractResult(text=_normalize_text(final_text), meta=meta)


# PDF


def _extract_pdf(data: bytes) -> Tuple[str, JsonDict]:
    """
    Extract text from PDF using the best available backend.
    Returns (text, meta).
    """
    # 1) PyMuPDF (fitz)
    try:
        import fitz  # type: ignore
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pages.append(page.get_text("text"))
        doc.close()
        text = "\n".join(pages)
        return text, {"pdf_backend": "pymupdf", "pdf_pages": len(pages)}
    except Exception:
        pass

    # 2) pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages)
        return text, {"pdf_backend": "pypdf", "pdf_pages": len(reader.pages)}
    except Exception:
        pass

    raise RuntimeError(
        "PDF extraction backend not available. Install one of: PyMuPDF (fitz) or pypdf.\n"
        "Example:\n"
        "  pip install pymupdf\n"
        "or\n"
        "  pip install pypdf"
    )


# DOCX


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document  # type: ignore
    except Exception as e:
        raise RuntimeError("python-docx is required to parse .docx files. Install: pip install python-docx") from e

    doc = Document(io.BytesIO(data))
    parts: List[str] = []

    # paragraphs
    for p in doc.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    # tables (optional but useful)
    for table in doc.tables:
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            line = " | ".join([c for c in cells if c])
            if line.strip():
                parts.append(line)

    return "\n".join(parts)


# IPYNB


def _extract_ipynb(nb: JsonDict) -> str:
    """
    Extract markdown + code from Jupyter notebook structure.
    """
    cells = nb.get("cells", [])
    out: List[str] = []
    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type", "")
        src = cell.get("source", "")
        if isinstance(src, list):
            src_text = "".join(src)
        else:
            src_text = str(src)

        src_text = src_text.strip("\n")

        if not src_text.strip():
            continue

        if ctype == "markdown":
            out.append(src_text)
        elif ctype == "code":
            # Вставляем как код-блок, чтобы LLM видел структуру
            out.append(f"[code cell {i}]\n```python\n{src_text}\n```")
        else:
            out.append(src_text)

    return "\n\n".join(out)


# Text helpers


def _decode_text(data: bytes) -> str:
    """
    Decode bytes into text with reasonable fallbacks.
    """
    # UTF-8 BOM
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")

    # Try utf-8
    try:
        return data.decode("utf-8")
    except Exception:
        pass

    # Try cp1251 (часто для русских txt)
    try:
        return data.decode("cp1251")
    except Exception:
        pass

    # Fallback
    return data.decode("utf-8", errors="replace")


def _pretty_json(text: str) -> str:
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return text


def _normalize_text(text: str) -> str:
    """
    Normalize whitespace:
    - unify newlines
    - remove excessive blank lines
    - trim trailing spaces
    """
    if not text:
        return ""

    # normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # trim trailing spaces per line
    text = "\n".join([ln.rstrip() for ln in text.split("\n")])

    # collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # strip global
    return text.strip()


def _is_likely_text(ext: str, mime: str) -> bool:
    """
    Heuristic to decide if file is text-like.
    """
    text_exts = {
        ".txt", ".md", ".markdown", ".rst",
        ".csv", ".tsv",
        ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
        ".json", ".yaml", ".yml",
        ".ini", ".cfg",
        ".html", ".htm", ".xml",
        ".tex",
        ".log",
    }
    if ext in text_exts:
        return True
    if mime.startswith("text/"):
        return True
    if mime in ("application/json", "application/xml"):
        return True
    return False


def _split_long_text(text: str, chunk_size: int) -> List[str]:
    """
    Split a single long paragraph into smaller pieces.
    Strategy:
    - try sentence split
    - fallback to hard slicing
    """
    t = text.strip()
    if len(t) <= chunk_size:
        return [t]

    # Split by sentence-like boundaries (works ok for RU/EN)
    # Keep delimiter by rebuilding.
    sentences = re.split(r"(?<=[.!?…])\s+", t)
    if len(sentences) <= 1:
        # no sentence boundaries
        return _hard_slice(t, chunk_size)

    parts: List[str] = []
    buf = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not buf:
            buf = s
            continue
        if len(buf) + 1 + len(s) <= chunk_size:
            buf = buf + " " + s
        else:
            parts.append(buf)
            buf = s
    if buf:
        parts.append(buf)

    # if some parts still too big (e.g. one huge sentence)
    out: List[str] = []
    for p in parts:
        if len(p) <= chunk_size:
            out.append(p)
        else:
            out.extend(_hard_slice(p, chunk_size))
    return out


def _hard_slice(text: str, chunk_size: int) -> List[str]:
    """
    Hard slicing with preference to split on spaces.
    """
    t = text.strip()
    out: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(i + chunk_size, n)
        if j < n:
            # try to break at nearest space
            k = t.rfind(" ", i, j)
            if k > i + int(chunk_size * 0.6):
                j = k
        out.append(t[i:j].strip())
        i = j
    return [p for p in out if p]


# Convenience: make chunks from file in one go


def extract_and_chunk(
        filename: str,
        data: bytes,
        *,
        mime: Optional[str] = None,
        chunk_size: int = 1400,
        overlap: int = 200,
        chunk_id_prefix: str = "chunk",
) -> Tuple[ExtractResult, List[Chunk]]:
    """
    Helper for pipelines: extract + split.
    """
    res = extract_text_from_bytes(filename, data, mime=mime)
    chunks = split_to_chunks(
        res.text,
        chunk_size=chunk_size,
        overlap=overlap,
        source_meta={**res.meta, "source_filename": filename},
        chunk_id_prefix=chunk_id_prefix,
    )
    return res, chunks
