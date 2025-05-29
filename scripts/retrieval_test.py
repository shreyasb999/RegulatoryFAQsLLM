#!/usr/bin/env python3
import faiss, pickle, json
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"
BASE_DIR   = Path(__file__).resolve().parent.parent
INDEX_DIR  = BASE_DIR / "data" / "index"
PARSED_DIR = BASE_DIR / "data" / "parsed"

def retrieve(query: str, doc_name: str, k: int = 3):
    # 1) Load model
    model = SentenceTransformer(MODEL_NAME)
    
    # 2) Load FAISS index + metadata
    idx      = faiss.read_index(str(INDEX_DIR / f"{doc_name}.index"))
    metadata = pickle.load(open(INDEX_DIR / f"{doc_name}_meta.pkl", "rb"))
    
    # 3) Load parsed chunks (to get the text)
    parsed_chunks = json.loads((PARSED_DIR / f"{doc_name}.json").read_text(encoding="utf-8"))
    
    # 4) Embed & normalize
    q_emb = model.encode([query]).astype("float32")
    faiss.normalize_L2(q_emb)
    
    # 5) Search
    D, I = idx.search(q_emb, k)
    seen = set()
    print(f"\nTop {k} results for '{query}' from {doc_name.upper()}:\n")
    for score, idx in zip(D[0], I[0]):
        # Use .get(...) or fall back, without indexing a missing key directly
        section_or_article = metadata[idx].get("article") or metadata[idx].get("section") or metadata[idx].get("chunk_id")
        if section_or_article in seen:
            continue
        seen.add(section_or_article)

        md      = metadata[idx]
        chunk   = parsed_chunks[idx]
        title   = md.get("article") or md.get("section") or md.get("chunk_id")
        snippet = chunk["text"].replace("\n", " ")
        print(f"• [{score:.3f}] {title}\n  {snippet[:200]}…\n")

if __name__ == "__main__":
    retrieve("What is personal data?", "gdpr")
    retrieve("When must a breach of unsecured PHI be reported?", "hipaa")
