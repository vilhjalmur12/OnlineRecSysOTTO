# src/ottorecsys/artifacts.py
from pathlib import Path
import json
import pickle
import duckdb
import numpy as np
import faiss

from .config import RUN_DIR

def load_manifest(run_dir: Path):
    with open(run_dir / "manifest.json") as f:
        return json.load(f)

def load_artifacts(run_dir: Path):
    manifest = load_manifest(run_dir)

    # Popular items (parquet)
    con = duckdb.connect()
    pop_path = (run_dir / "popular_items.parquet").as_posix()
    popular_items = [
        row[0]
        for row in con.execute(
            f"SELECT aid FROM read_parquet('{pop_path}') ORDER BY score DESC"
        ).fetchall()
    ]

    # Co-vis neighbors
    with open(run_dir / "covis_neighbors_index.pkl", "rb") as f:
        neighbors_index = pickle.load(f)

    # Item2Vec + FAISS index
    item2vec_dir = run_dir / "item2vec"
    item_vectors = np.load(item2vec_dir / "item_vectors.npy")
    with open(item2vec_dir / "w2v_items.txt") as f:
        w2v_items = np.array([line.strip() for line in f])

    w2v_item_to_idx = {item: i for i, item in enumerate(w2v_items)}
    item_vectors_f = np.ascontiguousarray(item_vectors.astype("float32"))
    faiss.normalize_L2(item_vectors_f)
    dim = item_vectors_f.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(item_vectors_f)

    return {
        "manifest": manifest,
        "popular_items": popular_items,
        "neighbors_index": neighbors_index,
        "w2v_items": w2v_items,
        "w2v_item_to_idx": w2v_item_to_idx,
        "item_vectors_f": item_vectors_f,
        "faiss_index": faiss_index,
    }

# Global serving state (loaded once when the process starts)
SERVING_STATE = load_artifacts(RUN_DIR)
