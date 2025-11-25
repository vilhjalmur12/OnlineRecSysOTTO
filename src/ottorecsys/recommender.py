# src/ottorecsys/recommender.py
from typing import List, Dict, Any

from .artifacts import SERVING_STATE
from .config import TOP_K_DEFAULT

def hybrid_recommend(
    session_aids: List[int],
    session_types: List[str],
    k: int = TOP_K_DEFAULT,
) -> List[int]:
    """
    Core hybrid logic:
      - last N events from the session
      - co-vis neighbours from SERVING_STATE["neighbors_index"]
      - item2vec/FAISS similarity from SERVING_STATE["faiss_index"]
      - popularity fallback from SERVING_STATE["popular_items"]
    """

    # Pseudocode placeholder – replace with your current notebook logic:
    # 1. score from co-vis
    # 2. score from item2vec
    # 3. combine + rank
    # 4. fill to k with popular_items
    # Ensure no duplicates and ignore items not allowed, etc.

    popular_items = SERVING_STATE["popular_items"]
    # TODO: implement actual hybrid; for now just use popularity
    recs = list(popular_items[:k])
    return recs


def recommend_for_session(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    request: {
      "session_id": str | int | None,
      "events": [{"aid": int, "ts": int, "type": str}, ...],
      "limit": int
    }
    """
    events = sorted(request["events"], key=lambda e: e["ts"])
    session_aids = [e["aid"] for e in events]
    session_types = [e["type"] for e in events]
    k = request.get("limit", TOP_K_DEFAULT)

    recs = hybrid_recommend(session_aids, session_types, k=k)

    meta = SERVING_STATE["manifest"]
    return {
        "session_id": request.get("session_id"),
        "recommendations": recs,
        "model_version": meta.get("model_version"),
        "run_id": meta.get("run_id"),
    }
