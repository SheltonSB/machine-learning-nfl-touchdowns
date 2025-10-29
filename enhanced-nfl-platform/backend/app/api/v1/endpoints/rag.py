"""
RAG (Retrieval-Augmented Generation) API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from app.core.rag_system import RAGSystem, get_rag_system
from app.schemas.rag import RAGQuery, RAGResponse, RAGStats
from typing import List

router = APIRouter()

@router.post("/query", response_model=RAGResponse)
async def query_rag(
    query: RAGQuery,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Ask a question about NFL data using RAG"""
    try:
        result = await rag_system.query(query.question, top_k=query.top_k)
        return RAGResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@router.get("/stats", response_model=RAGStats)
async def get_rag_stats(rag_system: RAGSystem = Depends(get_rag_system)):
    """Get RAG system statistics"""
    try:
        stats = await rag_system.get_knowledge_stats()
        return RAGStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@router.post("/add-document")
async def add_document(
    content: str,
    metadata: dict,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Add a new document to the knowledge base"""
    try:
        await rag_system.add_document(content, metadata)
        return {"message": "Document added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")

@router.get("/suggestions")
async def get_query_suggestions():
    """Get suggested questions for the RAG system"""
    suggestions = [
        "Who are the top 5 quarterbacks this season?",
        "What's the average touchdown rate for home games?",
        "Compare Tom Brady and Aaron Rodgers' performance",
        "Which team has the best passing offense?",
        "What are the key factors for touchdown predictions?",
        "How does weather affect quarterback performance?",
        "What's the trend in passing yards over the years?",
        "Which quarterbacks have the highest completion percentage?",
        "How do rookie quarterbacks perform compared to veterans?",
        "What's the correlation between rushing yards and passing touchdowns?"
    ]
    return {"suggestions": suggestions}

