import os
import re
from typing import Dict, Any, List

import pdfplumber  # for PDFs
import markdown

def _clean_text(text: str) -> str:
    """Normalize newlines and remove control characters."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", "", text)  # strip control chars
    text = re.sub(r"\n{3,}", "\n\n", text)  # collapse large blank gaps
    return text.strip()

def extract_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return _clean_text(f.read())


def extract_md(path: str) -> str:
    """Convert Markdown to plain text (roughly)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        md_text = f.read()
    # Convert markdown → HTML → strip tags (basic approach)
    html = markdown.markdown(md_text)
    plain = re.sub(r"<[^>]+>", "", html)
    return _clean_text(plain)


def extract_pdf(path: str) -> str:
    """Extract text from a text-based PDF using pdfplumber (no OCR fallback)."""
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_pages.append(txt)
    return _clean_text("\n\n".join(text_pages))

def extract_submission(path: str) -> Dict[str, Any]:
    """
    Extract raw text from a student submission file.
    Returns: { source_type, raw_text, diagnostics }
    """
    diagnostics: List[Dict[str, Any]] = []
    raw_text = ""
    source_type = "unknown"

    if not os.path.exists(path):
        return {
            "source_type": source_type,
            "raw_text": "",
            "diagnostics": [{"severity": "error", "message": f"File not found: {path}"}]
        }

    try:
        if path.lower().endswith(".txt"):
            source_type = "txt"
            raw_text = extract_txt(path)
        elif path.lower().endswith((".md", ".markdown")):
            source_type = "md"
            raw_text = extract_md(path)
        elif path.lower().endswith(".pdf"):
            source_type = "pdf"
            raw_text = extract_pdf(path)
        else:
            diagnostics.append({"severity": "error", "message": "Unsupported file type"})
    except Exception as e:
        diagnostics.append({"severity": "error", "message": str(e)})

    # flag empty text
    if not raw_text.strip():
        diagnostics.append({"severity": "warning", "message": "Extracted text is empty"})

    return {
        "source_type": source_type,
        "raw_text": raw_text,
        "diagnostics": diagnostics
    }

import re

def parse_student_submission(file_path: str):
    text = extract_submission(file_path)['raw_text']
    # Regex pattern for question numbers like "1.", "2a.", "3b."
    pattern = re.compile(r'^(\d+\w*\.)', re.MULTILINE)

    matches = list(pattern.finditer(text))
    result = {}
    diagnostics = {"missing_answers": [], "skipped_questions": [], "duplicates": []}

    # 🧩 Handle case: no question numbers found
    if not matches:
        diagnostics["error"] = "No question numbers found in submission text."
        return {"answers": {}, "diagnostics": diagnostics}

    seen_numbers = []
    previous_num = 0

    for i, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer_text = text[start:end].strip()

        # Detect empty or missing answers
        if not answer_text:
            diagnostics["missing_answers"].append(key)

        # Detect duplicate question numbers
        if key in seen_numbers:
            diagnostics["duplicates"].append(key)
        seen_numbers.append(key)

        # Detect skipped numeric questions
        try:
            current_num = int(re.match(r"(\d+)", key).group(1))
            if current_num - previous_num > 1:
                skipped = [f"{n}." for n in range(previous_num + 1, current_num)]
                diagnostics["skipped_questions"].extend(skipped)
            previous_num = current_num
        except Exception:
            pass  # ignore if non-numeric like 1a.

        result[key] = answer_text

    return {"answers": result, "diagnostics": diagnostics}