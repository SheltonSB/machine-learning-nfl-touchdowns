"""
RAG (Retrieval-Augmented Generation) System for NFL Data
Uses vector embeddings and LLMs to answer natural language questions
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from app.core.config import settings
from app.core.database import get_db

SKIP_RAG_IMPORTS = os.getenv("SKIP_RAG_IMPORTS") == "1"

if not SKIP_RAG_IMPORTS:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        SentenceTransformer = None
else:
    SentenceTransformer = None

if not SKIP_RAG_IMPORTS:
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        pipeline = None
        AutoTokenizer = None
        AutoModelForCausalLM = None
else:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None

if not SKIP_RAG_IMPORTS:
    try:
        import pinecone  # type: ignore
        from pinecone import Pinecone  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        pinecone = None
        Pinecone = None
else:
    pinecone = None
    Pinecone = None

logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG System for NFL data insights"""
    
    def __init__(self):
        self.embedding_model = None
        self.llm_pipeline = None
        self.vector_db = None
        self.index_name = "nfl-knowledge-base"
        self.embeddings_cache = {}
        
    async def initialize(self):
        """Initialize the RAG system"""
        logger.info("Initializing RAG System...")
        
        try:
            # Initialize embedding model
            await self._initialize_embedding_model()
            
            # Initialize LLM
            await self._initialize_llm()
            
            # Initialize vector database
            await self._initialize_vector_db()
            
            # Create knowledge base if it doesn't exist
            await self._create_knowledge_base()
            
            logger.info("RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    async def _initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        logger.info("Loading embedding model...")
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers is not installed; install it to enable RAG embeddings.')
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    
    async def _initialize_llm(self):
        """Initialize the language model"""
        logger.info("Loading language model...")
        if pipeline is None:
            raise RuntimeError('transformers is not installed; install it to enable language model support.')
        
        # Use a smaller model for faster inference
        model_name = "microsoft/DialoGPT-small"  # Changed to smaller model
        
        self.llm_pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256
        )
        
        logger.info("Language model loaded successfully")
    
    async def _initialize_vector_db(self):
        """Initialize Pinecone vector database"""
        logger.info("Initializing vector database...")
        
        if Pinecone is None:
            logger.warning('Pinecone client is not installed; defaulting to in-memory vector store.')
            self.vector_db = None
            return
        
        try:
            # Initialize Pinecone
            pc = Pinecone(api_key=settings.VECTOR_DB_API_KEY)
            
            # Check if index exists
            if self.index_name not in pc.list_indexes().names():
                # Create index
                pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine"
                )
                logger.info(f"Created new index: {self.index_name}")
            
            # Connect to index
            self.vector_db = pc.Index(self.index_name)
            logger.info("Vector database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            # Fallback to in-memory storage for development
            self.vector_db = None
            logger.warning("Using in-memory storage as fallback")
    
    async def _create_knowledge_base(self):
        """Create knowledge base from NFL data"""
        logger.info("Creating knowledge base...")
        
        try:
            # Get database session
            db = next(get_db())
            
            # Extract knowledge from database
            knowledge_docs = await self._extract_knowledge_from_db(db)
            
            if not knowledge_docs:
                logger.warning("No knowledge documents found")
                return
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(knowledge_docs)
            
            # Store in vector database
            await self._store_embeddings(embeddings, knowledge_docs)
            
            logger.info(f"Created knowledge base with {len(knowledge_docs)} documents")
            
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            raise
    
    async def _extract_knowledge_from_db(self, db: Session) -> List[Dict[str, Any]]:
        """Extract knowledge documents from database"""
        knowledge_docs = []
        
        try:
            # Extract player statistics
            from app.models.database import Player, GameLog, Game, Team
            
            # Get player summaries
            players = db.query(Player).limit(100).all()  # Limit for demo
            for player in players:
                doc = {
                    "content": f"Player {player.first_name} {player.last_name} is a {player.position} "
                              f"who is {player.age} years old, {player.height} inches tall, "
                              f"weighs {player.weight} pounds, and has {player.experience} years of experience. "
                              f"Currently plays for {player.current_team or 'Unknown team'}.",
                    "metadata": {
                        "type": "player",
                        "player_id": player.player_id,
                        "position": player.position,
                        "team": player.current_team
                    }
                }
                knowledge_docs.append(doc)
            
            # Get game statistics
            game_logs = db.query(GameLog).join(Player).limit(200).all()  # Limit for demo
            for log in game_logs:
                doc = {
                    "content": f"In a game, {log.player.first_name} {log.player.last_name} "
                              f"threw for {log.passing_yards} yards, {log.td_passes} touchdowns, "
                              f"and {log.interceptions} interceptions. "
                              f"Completed {log.passes_completed} of {log.passes_attempted} passes "
                              f"for a {log.completion_percentage:.1f}% completion rate.",
                    "metadata": {
                        "type": "game_log",
                        "player_id": log.player.player_id,
                        "passing_yards": log.passing_yards,
                        "td_passes": log.td_passes
                    }
                }
                knowledge_docs.append(doc)
            
            # Get team information
            teams = db.query(Team).all()
            for team in teams:
                doc = {
                    "content": f"The {team.team_name} ({team.team_code}) are based in {team.city} "
                              f"and play in the {team.conference} conference, {team.division} division.",
                    "metadata": {
                        "type": "team",
                        "team_code": team.team_code,
                        "conference": team.conference,
                        "division": team.division
                    }
                }
                knowledge_docs.append(doc)
            
            # Add general NFL knowledge
            general_docs = [
                {
                    "content": "The NFL regular season consists of 17 games per team, with playoffs following. "
                              "Teams are divided into two conferences: AFC and NFC, each with 4 divisions.",
                    "metadata": {"type": "general", "topic": "season_structure"}
                },
                {
                    "content": "Touchdown passes are worth 6 points plus an extra point attempt. "
                              "A quarterback throws a touchdown pass when the ball is caught in the end zone.",
                    "metadata": {"type": "general", "topic": "scoring"}
                },
                {
                    "content": "Quarterback statistics include passing yards, touchdown passes, interceptions, "
                              "completion percentage, yards per attempt, and passer rating.",
                    "metadata": {"type": "general", "topic": "qb_stats"}
                }
            ]
            knowledge_docs.extend(general_docs)
            
        except Exception as e:
            logger.error(f"Error extracting knowledge from database: {e}")
        
        return knowledge_docs
    
    async def _generate_embeddings(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for documents"""
        logger.info("Generating embeddings...")
        
        contents = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.encode(contents)
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    async def _store_embeddings(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Store embeddings in vector database"""
        if self.vector_db is None:
            logger.warning("Vector database not available, storing in memory")
            return
        
        logger.info("Storing embeddings in vector database...")
        
        try:
            # Prepare vectors for Pinecone
            vectors = []
            for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
                vector = {
                    "id": f"doc_{i}",
                    "values": embedding.tolist(),
                    # Ensure content is stored with metadata for retrieval consistency
                    "metadata": {**doc["metadata"], "content": doc["content"]}
                }
                vectors.append(vector)
            
            # Batch upsert
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.vector_db.upsert(vectors=batch)
            
            logger.info(f"Stored {len(vectors)} vectors in vector database")
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            raise
    
    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Answer a question using RAG"""
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question])
            
            # Retrieve relevant documents
            relevant_docs = await self._retrieve_documents(query_embedding[0], top_k)
            
            # Generate answer
            answer = await self._generate_answer(question, relevant_docs)
            
            return {
                "question": question,
                "answer": answer,
                "relevant_docs": relevant_docs,
                "confidence": self._calculate_confidence(relevant_docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "question": question,
                "answer": "I'm sorry, I encountered an error processing your question. Please try again.",
                "relevant_docs": [],
                "confidence": 0.0
            }
    
    async def _retrieve_documents(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from vector database"""
        if self.vector_db is None:
            # Fallback to simple keyword matching
            return await self._fallback_retrieval(query_embedding, top_k)
        
        try:
            # Query vector database
            results = self.vector_db.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Format results
            relevant_docs = []
            for match in results['matches']:
                doc = {
                    # We explicitly store content within metadata during upsert
                    "content": match['metadata'].get('content', ''),
                    "metadata": match['metadata'],
                    "score": match['score']
                }
                relevant_docs.append(doc)
            
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return await self._fallback_retrieval(query_embedding, top_k)
    
    async def _fallback_retrieval(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Fallback retrieval method"""
        # Simple fallback - return general NFL knowledge
        return [
            {
                "content": "The NFL is the National Football League, consisting of 32 teams divided into two conferences.",
                "metadata": {"type": "general"},
                "score": 0.8
            }
        ]
    
    async def _generate_answer(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM"""
        try:
            # Create context from relevant documents
            context = " ".join([doc["content"] for doc in relevant_docs])
            
            # Create prompt
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Generate response
            response = self.llm_pipeline(
                prompt,
                max_length=len(prompt.split()) + 50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            # Extract answer
            answer = response[0]['generated_text']
            answer = answer.replace(prompt, "").strip()
            
            # Clean up answer
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            return answer if answer else "I don't have enough information to answer that question."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm sorry, I couldn't generate an answer to your question."
    
    def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the answer"""
        if not relevant_docs:
            return 0.0
        
        # Average score of relevant documents
        scores = [doc.get('score', 0.5) for doc in relevant_docs]
        return sum(scores) / len(scores)
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add a new document to the knowledge base"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([content])[0]
            
            # Store in vector database
            doc_id = f"doc_{len(self.embeddings_cache)}"
            if self.vector_db is not None:
                self.vector_db.upsert(
                    vectors=[{
                        "id": doc_id,
                        "values": embedding.tolist(),
                        # Store content alongside metadata for retrieval
                        "metadata": {**metadata, "content": content}
                    }]
                )
            
            # Cache locally
            self.embeddings_cache[doc_id] = {
                "content": content,
                "metadata": metadata,
                "embedding": embedding
            }
            
            logger.info(f"Added document to knowledge base: {doc_id}")
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        try:
            if self.vector_db is None:
                return {
                    "total_documents": len(self.embeddings_cache),
                    "vector_db_status": "offline",
                    "embedding_model": settings.EMBEDDING_MODEL
                }
            
            # Get index stats
            stats = self.vector_db.describe_index_stats()
            
            return {
                "total_documents": stats.total_vector_count,
                "vector_db_status": "online",
                "embedding_model": settings.EMBEDDING_MODEL,
                "index_name": self.index_name
            }
            
        except Exception as e:
            logger.error(f"Error getting knowledge stats: {e}")
            return {
                "total_documents": 0,
                "vector_db_status": "error",
                "embedding_model": settings.EMBEDDING_MODEL
            }


_rag_system_singleton = None


def set_rag_system(instance: "RAGSystem") -> None:
    global _rag_system_singleton
    _rag_system_singleton = instance


def get_rag_system() -> "RAGSystem":
    if _rag_system_singleton is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System not initialized"
        )
    return _rag_system_singleton
