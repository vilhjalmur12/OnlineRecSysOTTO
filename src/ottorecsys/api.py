# src/ottorecsys/api.py
from fastapi import FastAPI

from .api_schemas import RecommendRequest, RecommendResponse
from .recommender import recommend_for_session

app = FastAPI(
    title="OTTO Recommender API",
    version="1.0",
)

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    result = recommend_for_session(req.dict())
    return result
