#!/usr/bin/env python3
"""
Build local FAISS indexes for GDPR and HIPAA chunks
using SentenceTransformer embeddings.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR   = PROJECT_ROOT / "data" / "parsed"
INDEX_DIR    = PROJECT_ROOT / "data" / "index"

# Ensure index directory exists
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Choose your model
MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast 384-dim embeddings
model = SentenceTransformer(MODEL_NAME)

def build_index(doc_name: str, json_path: Path):
    # 1) Load chunks
    data = json.loads(json_path.read_text(encoding="utf-8"))
    texts    = [item["text"] for item in data]
    metadata = [
        {k: item[k] for k in item if k not in ("text",)}
        for item in data
    ]
    
    # 2) Compute embeddings (in batches)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.vstack(embeddings).astype("float32")  # FAISS needs float32

    # 3) Build FAISS index (Inner Product on L2-normalized vectors ≈ cosine)
    #    Normalize vectors
    faiss.normalize_L2(embeddings)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # 4) Persist index + metadata
    faiss.write_index(index, str(INDEX_DIR / f"{doc_name}.index"))
    with open(INDEX_DIR / f"{doc_name}_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"[OK] Built {doc_name} index: {embeddings.shape[0]} vectors, dim={dim}")

if __name__ == "__main__":
    build_index("gdpr",  PARSED_DIR / "gdpr.json")
    build_index("hipaa", PARSED_DIR / "hipaa.json")
