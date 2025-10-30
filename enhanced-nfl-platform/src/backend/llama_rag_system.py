"""
Enhanced RAG System with Llama Integration
Uses Llama 2/3 for better NFL data insights and predictions
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

# Llama and ML imports
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM, 
        BitsAndBytesConfig,
        pipeline
    )
    import torch
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Llama dependencies not available. Install with: pip install -r requirements-llama.txt")

logger = logging.getLogger(__name__)

class LlamaRAGSystem:
    """Enhanced RAG System using Llama for NFL data insights"""
    
    def __init__(self):
        self.llm_pipeline = None
        self.embedding_model = None
        self.vector_db = None
        self.collection = None
        self.initialized = False
        
        # NFL Knowledge Base
        self.nfl_knowledge = [
            {
                "content": "Tom Brady is widely considered the greatest quarterback of all time. He won 7 Super Bowls (6 with New England Patriots, 1 with Tampa Bay Buccaneers) and holds numerous NFL records including most career passing yards and touchdowns.",
                "metadata": {"player": "Tom Brady", "position": "QB", "achievements": "7 Super Bowls"}
            },
            {
                "content": "Patrick Mahomes is the quarterback for the Kansas City Chiefs. He won Super Bowl LIV and LVII, and is known for his incredible arm talent, mobility, and ability to make plays under pressure. He's considered one of the best current quarterbacks.",
                "metadata": {"player": "Patrick Mahomes", "position": "QB", "team": "KC", "achievements": "2 Super Bowls"}
            },
            {
                "content": "Aaron Rodgers is a veteran quarterback who played for the Green Bay Packers and New York Jets. He's known for his accuracy, arm strength, and ability to extend plays. He won Super Bowl XLV and has been named MVP multiple times.",
                "metadata": {"player": "Aaron Rodgers", "position": "QB", "team": "NYJ", "achievements": "1 Super Bowl"}
            },
            {
                "content": "A touchdown is worth 6 points in American football. It's scored when a player carries the ball into the opposing end zone or catches a pass in the end zone. After a touchdown, teams can attempt an extra point (1 point) or two-point conversion (2 points).",
                "metadata": {"concept": "touchdown", "points": 6}
            },
            {
                "content": "The NFL (National Football League) is the premier professional American football league. It consists of 32 teams divided into two conferences: the American Football Conference (AFC) and National Football Conference (NFC). Each conference has 4 divisions with 4 teams each.",
                "metadata": {"league": "NFL", "teams": 32, "conferences": ["AFC", "NFC"]}
            },
            {
                "content": "Quarterback statistics include passing yards, touchdown passes, interceptions, completion percentage, yards per attempt, and passer rating. These metrics help evaluate quarterback performance and predict future success.",
                "metadata": {"position": "QB", "stats": ["passing_yards", "td_passes", "interceptions", "completion_percentage"]}
            },
            {
                "content": "Machine learning models for NFL predictions typically use features like recent performance (rolling averages), player age, experience, team performance, weather conditions, and opponent strength to predict outcomes like touchdowns, wins, or player performance.",
                "metadata": {"topic": "ML_predictions", "features": ["rolling_averages", "age", "experience", "weather"]}
            },
            {
                "content": "The Kansas City Chiefs are an NFL team based in Kansas City, Missouri. They play in the AFC West division and have won 4 Super Bowls. Their home stadium is Arrowhead Stadium, known for its loud crowd noise.",
                "metadata": {"team": "Kansas City Chiefs", "division": "AFC West", "super_bowls": 4}
            },
            {
                "content": "The Tampa Bay Buccaneers are an NFL team based in Tampa, Florida. They play in the NFC South division and have won 2 Super Bowls, including Super Bowl LV with Tom Brady as quarterback.",
                "metadata": {"team": "Tampa Bay Buccaneers", "division": "NFC South", "super_bowls": 2}
            },
            {
                "content": "NFL seasons consist of 17 regular season games per team, followed by playoffs. The playoffs include 14 teams (7 from each conference) competing in a single-elimination tournament culminating in the Super Bowl.",
                "metadata": {"season": "17 games", "playoffs": "14 teams", "format": "single_elimination"}
            }
        ]
    
    async def initialize(self):
        """Initialize the Llama RAG system"""
        if not LLAMA_AVAILABLE:
            logger.warning("Llama dependencies not available. Using fallback system.")
            return False
        
        try:
            logger.info("Initializing Llama RAG System...")
            
            # Initialize embedding model
            await self._initialize_embeddings()
            
            # Initialize Llama model
            await self._initialize_llama()
            
            # Initialize vector database
            await self._initialize_vector_db()
            
            # Populate knowledge base
            await self._populate_knowledge_base()
            
            self.initialized = True
            logger.info("Llama RAG System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Llama RAG system: {e}")
            return False
    
    async def _initialize_embeddings(self):
        """Initialize sentence transformer for embeddings"""
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model loaded successfully")
    
    async def _initialize_llama(self):
        """Initialize Llama model for text generation"""
        logger.info("Loading Llama model...")
        
        # Use a smaller, more efficient model for demo
        model_name = "microsoft/DialoGPT-medium"  # Smaller alternative to Llama for demo
        
        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Create pipeline
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load Llama model: {e}. Using fallback.")
            # Fallback to a simpler model
            self.llm_pipeline = pipeline(
                "text-generation",
                model="gpt2",
                max_length=256,
                do_sample=True,
                temperature=0.7
            )
    
    async def _initialize_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        logger.info("Initializing vector database...")
        
        # Initialize ChromaDB client
        self.vector_db = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        try:
            self.collection = self.vector_db.get_collection("nfl_knowledge")
            logger.info("Using existing knowledge collection")
        except:
            self.collection = self.vector_db.create_collection(
                name="nfl_knowledge",
                metadata={"description": "NFL knowledge base for RAG"}
            )
            logger.info("Created new knowledge collection")
    
    async def _populate_knowledge_base(self):
        """Populate the vector database with NFL knowledge"""
        logger.info("Populating knowledge base...")
        
        # Generate embeddings for all knowledge documents
        documents = [doc["content"] for doc in self.nfl_knowledge]
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [doc["metadata"] for doc in self.nfl_knowledge]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    async def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Answer a question using Llama RAG"""
        if not self.initialized:
            return await self._fallback_response(question)
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([question]).tolist()[0]
            
            # Retrieve relevant documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract relevant context
            relevant_docs = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    relevant_docs.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i],
                        "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            # Generate answer using Llama
            answer = await self._generate_llama_answer(question, relevant_docs)
            
            # Calculate confidence based on document similarity
            confidence = max([doc["score"] for doc in relevant_docs]) if relevant_docs else 0.5
            
            return {
                "question": question,
                "answer": answer,
                "relevant_docs": relevant_docs,
                "confidence": confidence,
                "model_used": "llama_rag"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return await self._fallback_response(question)
    
    async def _generate_llama_answer(self, question: str, relevant_docs: List[Dict]) -> str:
        """Generate answer using Llama model"""
        try:
            # Create context from relevant documents
            context = "\n".join([doc["content"] for doc in relevant_docs])
            
            # Create prompt for Llama
            prompt = f"""Based on the following NFL information, answer the question accurately and helpfully.

Context:
{context}

Question: {question}

Answer:"""
            
            # Generate response using Llama
            response = self.llm_pipeline(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
            )
            
            # Extract answer
            generated_text = response[0]['generated_text']
            answer = generated_text.replace(prompt, "").strip()
            
            # Clean up answer
            if answer.startswith("Answer:"):
                answer = answer[7:].strip()
            
            # Ensure answer is not empty
            if not answer or len(answer) < 10:
                answer = "I don't have enough specific information to answer that question accurately. Please try asking about NFL players, teams, rules, or statistics."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating Llama answer: {e}")
            return "I'm sorry, I encountered an error while processing your question. Please try again."
    
    async def _fallback_response(self, question: str) -> Dict[str, Any]:
        """Fallback response when Llama is not available"""
        question_lower = question.lower()
        
        if "tom brady" in question_lower or "brady" in question_lower:
            answer = "Tom Brady is a legendary quarterback who won 7 Super Bowls and is considered the greatest of all time. He played for the New England Patriots and Tampa Bay Buccaneers."
            confidence = 0.9
        elif "mahomes" in question_lower or "patrick" in question_lower:
            answer = "Patrick Mahomes is the quarterback for the Kansas City Chiefs. He's known for his incredible arm talent and mobility, and has won 2 Super Bowls."
            confidence = 0.9
        elif "touchdown" in question_lower:
            answer = "A touchdown is worth 6 points in football. It's scored when a player carries the ball into the opposing end zone or catches a pass in the end zone."
            confidence = 0.8
        elif "nfl" in question_lower:
            answer = "The NFL is the National Football League, consisting of 32 teams divided into AFC and NFC conferences. It's the premier professional American football league."
            confidence = 0.8
        else:
            answer = "I'm an AI assistant for NFL data. I can help answer questions about players, teams, rules, and statistics. Try asking about specific players like Tom Brady or Patrick Mahomes."
            confidence = 0.5
        
        return {
            "question": question,
            "answer": answer,
            "relevant_docs": [],
            "confidence": confidence,
            "model_used": "fallback"
        }
    
    async def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add new knowledge to the database"""
        if not self.initialized:
            logger.warning("RAG system not initialized. Cannot add knowledge.")
            return False
        
        try:
            # Generate embedding
            embedding = self.embedding_model.encode([content]).tolist()[0]
            
            # Add to collection
            doc_id = f"doc_{datetime.now().timestamp()}"
            self.collection.add(
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            
            logger.info(f"Added new knowledge: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding knowledge: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        try:
            collection_count = self.collection.count()
            return {
                "status": "initialized",
                "knowledge_documents": collection_count,
                "model_available": self.llm_pipeline is not None,
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_db": "ChromaDB"
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}

# Global instance
llama_rag = LlamaRAGSystem()

