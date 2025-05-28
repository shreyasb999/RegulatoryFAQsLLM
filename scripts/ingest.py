#!/usr/bin/env python3
"""
RAG Data Ingestion & Parsing
============================
This script handles **downloading, parsing, and validating** the legal corpora for the
Retrieval-Augmented Generation (RAG) pipeline in the Regulatory FAQ project.

Features:
  • GDPR HTML → Article-level JSON chunks
  • HIPAA combined PDF → Section-level JSON chunks
  • Registry for additional regulations (e.g., CCPA) via simple dict entries
  • Automated **chunking** (~200–500 words) for embedding
  • **Schema validation** that can be used in unit tests
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# ─────────────────────────────────── Constants ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # rag_project/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"

GDPR_URL = "https://gdpr-info.eu/"
HIPAA_PDF_URL = (
    "https://www.hhs.gov/sites/default/files/ocr/privacy/hipaa/administrative/combined/"
    "hipaa-simplification-201303.pdf"
)

# Registry for regulations
REGISTRY: List[Dict] = [
    {"name": "GDPR", "type": "html", "url": GDPR_URL, "out": PARSED_DIR / "gdpr.json"},
    {"name": "HIPAA", "type": "pdf", "url": HIPAA_PDF_URL, "out": PARSED_DIR / "hipaa.json"},
    # Add more entries here
]

# Ensure directories exist
for d in (RAW_DIR, PARSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ───────────────────────────────── Utilities ────────────────────────────────────

def _download(url: str, dest: Path, binary: bool = False) -> None:
    """
    Fetch a URL to a local file if missing.
    For text downloads, write with UTF-8 encoding to avoid Windows cp1252 errors.
    """
    if dest.exists():
        print(f"[SKIP] {dest.name} exists")
        return
    print(f"[DOWNLOAD] {url} → {dest.relative_to(PROJECT_ROOT)}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    }
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    if binary:
        mode = 'wb'
        with open(dest, mode) as f:
            f.write(resp.content)
    else:
        mode = 'w'
        # explicitly set utf-8 encoding
        with open(dest, mode, encoding='utf-8') as f:
            f.write(resp.text)

def _chunk_text(text: str, min_words: int = 100, max_words: int = 350) -> List[str]:
    """
    Split text into chunks between min_words and max_words at paragraph boundaries.
    Merges any under-length chunk forward, and ensures the last chunk meets min_words
    by merging it backward if necessary.
    """
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    raw_chunks, curr = [], []
    wc = lambda s: len(s.split())

    # 1) Build up to max_words
    for p in paras:
        if curr and wc(' '.join(curr + [p])) > max_words:
            raw_chunks.append('\n'.join(curr))
            curr = [p]
        else:
            curr.append(p)
    if curr:
        raw_chunks.append('\n'.join(curr))

    # 2) Merge under-length chunks forward
    merged = []
    i = 0
    while i < len(raw_chunks):
        chunk = raw_chunks[i]
        if wc(chunk) < min_words and i + 1 < len(raw_chunks):
            raw_chunks[i+1] = chunk + '\n' + raw_chunks[i+1]
        else:
            merged.append(chunk)
        i += 1

    # 3) If the very last chunk is still too short, merge it backward
    if len(merged) > 1 and wc(merged[-1]) < min_words:
        merged[-2] = merged[-2] + '\n' + merged[-1]
        merged.pop(-1)

    return merged

# ───────────────────────────── GDPR Parsing (HTML) ─────────────────────────────

def _parse_gdpr() -> None:
    """Download and parse GDPR articles into JSON chunks."""
    index_file = RAW_DIR / 'gdpr_index.html'
    # Always re-download the index to avoid stale HTML
    if index_file.exists():
        index_file.unlink()
    _download(GDPR_URL, index_file)

    html = index_file.read_text('utf-8')
    soup = BeautifulSoup(html, 'html.parser')

    # Find any <a> whose text is "Article <number>"
    # -------- build list of article URLs --------
    article_links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if re.search(r'/art-\d+-gdpr/', href, re.IGNORECASE):
            article_links.append(urljoin(GDPR_URL, href))

    # deduplicate & keep order
    article_links = list(dict.fromkeys(article_links))

    if not article_links:
        raise RuntimeError('No Article links found on GDPR index page.')

    output = []
    for i, link in enumerate(article_links, 1):
        page_file = RAW_DIR / f'gdpr_article_{i}.html'
        _download(link, page_file)

        page = BeautifulSoup(page_file.read_text('utf-8'), 'html.parser')
        title = page.find('h1').get_text(' ', strip=True)
        body_div = page.find('div', class_='entry-content') or page.find('div', id='content')
        if not body_div:
            raise RuntimeError(f'Could not find content div on {link}')
        body = body_div.get_text('\n', strip=True)

        for ci, chunk in enumerate(_chunk_text(body)):
            output.append({
                'doc': 'GDPR',
                'article': title,
                'chunk_id': f'{i}-{ci}',
                'text': chunk,
                'source_url': link,
            })

    out_path = PARSED_DIR / 'gdpr.json'
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"[OK] GDPR parsed: {len(output)} chunks.")

# ───────────────────────────── HIPAA Parsing (PDF) ─────────────────────────────

SECTION_PATTERN = re.compile(r"^§\s*\d+\.\d+.*$", re.MULTILINE)

def _parse_hipaa() -> None:
    """Download and parse HIPAA PDF into section-level JSON chunks."""
    pdf_file = RAW_DIR / 'hipaa.pdf'
    _download(HIPAA_PDF_URL, pdf_file, binary=True)

    reader = PdfReader(str(pdf_file))
    text = '\n'.join(p.extract_text() or '' for p in reader.pages)
    parts = SECTION_PATTERN.split(text)
    headers = SECTION_PATTERN.findall(text)
    if not headers:
        raise RuntimeError('HIPAA split failed—check SECTION_PATTERN.')

    output = []
    for hdr, body in zip(headers, parts[1:]):
        hdr_clean = ' '.join(hdr.split())
        for ci, chunk in enumerate(_chunk_text(body)):
            word_count = len(chunk.split())
            if word_count < 20:
                continue  # Skip tiny chunks
            output.append({
                'doc': 'HIPAA',
                'section': hdr_clean,
                'chunk_id': f"{hdr_clean.split()[0]}-{ci}",
                'text': chunk,
            })

    (PARSED_DIR / 'hipaa.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(f"[OK] HIPAA parsed: {len(output)} chunks.")


# ───────────────────────────── Registry & Main ────────────────────────────────

def ingest_all() -> None:
    print('▶ Ingestion started...')
    _parse_gdpr()
    _parse_hipaa()
    print('✔ All docs ingested.')

# ────────────────────────────────── Schema Validation ────────────────────────────────
TEST_EXPECTS = {
    'GDPR': {'path': PARSED_DIR / 'gdpr.json', 'min_chunks': 90, 'keys': {'doc', 'article', 'text'}},
    'HIPAA': {'path': PARSED_DIR / 'hipaa.json', 'min_chunks': 50, 'keys': {'doc', 'section', 'text'}},
}

def _validate_schema(key: str):
    cfg = TEST_EXPECTS[key]
    data = json.loads(cfg['path'].read_text('utf-8'))
    assert len(data) >= cfg['min_chunks'], f"{key}: {len(data)} chunks < {cfg['min_chunks']}"
    for item in data[:3]:
        assert cfg['keys'].issubset(item.keys()), f"{key} missing keys {cfg['keys'] - item.keys()}"

if __name__ == '__main__':
    # Run ingestion; validation is handled separately in tests
    ingest_all()
    print('✔ Ingestion complete.')
