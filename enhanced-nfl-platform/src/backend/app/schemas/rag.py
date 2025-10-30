"""
RAG Pydantic schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class RAGQuery(BaseModel):
    question: str
    top_k: int = 5

class RAGResponse(BaseModel):
    question: str
    answer: str
    relevant_docs: List[Dict[str, Any]]
    confidence: float

class RAGStats(BaseModel):
    total_documents: int
    vector_db_status: str
    embedding_model: str
    index_name: Optional[str] = None

