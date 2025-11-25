# src/ottorecsys/api_schemas.py
from typing import List, Optional
from pydantic import BaseModel

class EventIn(BaseModel):
    aid: int
    ts: int
    type: str

class RecommendRequest(BaseModel):
    session_id: Optional[str] = None
    events: List[EventIn]
    limit: int = 20

class RecommendResponse(BaseModel):
    session_id: Optional[str]
    recommendations: List[int]
    model_version: str
    run_id: str
