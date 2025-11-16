#!/usr/bin/env python3
"""
NFL AI Platform - Vercel Serverless Entry Point

This file is the entry point for Vercel's serverless Python runtime.
Vercel automatically handles the ASGI server, so we just need to export the FastAPI app.

IMPORTANT: Vercel's serverless functions don't run uvicorn - they use their own ASGI handler.
The app must be exported directly, not run with uvicorn.run().
"""
import sys
import os
import logging
from pathlib import Path

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to Python path for imports
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR / "backend"

# Add paths to sys.path in order of preference
paths_to_add = [
    str(CURRENT_DIR),  # Root directory
    str(BACKEND_DIR),  # Backend directory
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
        logger.info(f"Added to Python path: {path}")

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Script directory: {CURRENT_DIR}")
logger.info(f"Backend directory exists: {BACKEND_DIR.exists()}")
logger.info(f"Python path: {sys.path[:5]}")  # Show first 5 entries

# Import the FastAPI app
# Using working_app as it's the simplest and most compatible for serverless
try:
    logger.info("Attempting to import from backend.working_app...")
    from backend.working_app import app
    logger.info("Successfully imported app from backend.working_app")
except ImportError as e:
    logger.warning(f"Import from backend.working_app failed: {e}")
    # Fallback: try different import strategies
    try:
        logger.info("Trying alternative import path...")
        # Try importing as a module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "working_app", 
            BACKEND_DIR / "working_app.py"
        )
        if spec and spec.loader:
            working_app = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(working_app)
            app = working_app.app
            logger.info("Successfully imported app using importlib")
        else:
            raise ImportError("Could not create module spec")
    except Exception as e2:
        logger.error(f"Alternative import also failed: {e2}")
        # Last resort: create a minimal app with error message
        # Try to import FastAPI, but handle if it's not installed (local dev without deps)
        try:
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            
            app = FastAPI(
                title="NFL AI Platform",
                version="1.0.0",
                description="NFL AI Platform - Import Error Mode"
            )
            
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            @app.get("/")
            async def root():
                return {
                    "message": "NFL AI Platform",
                    "status": "error",
                    "error": "Could not import main app",
                    "import_error": str(e),
                    "alternative_error": str(e2) if 'e2' in locals() else None,
                    "python_path": sys.path[:5],
                    "backend_dir": str(BACKEND_DIR),
                    "backend_exists": BACKEND_DIR.exists(),
                    "note": "This is likely a local dependency issue. Install dependencies with: pip install -r requirements.txt"
                }
            
            @app.get("/health")
            async def health():
                return {"status": "degraded", "reason": "import_failed"}
        except ImportError as fastapi_error:
            # Even FastAPI isn't available - this is a local dev issue
            logger.critical(f"FastAPI not installed: {fastapi_error}")
            logger.critical("This is expected if dependencies aren't installed locally.")
            logger.critical("For local testing, run: pip install -r requirements.txt")
            logger.critical("For Vercel deployment, dependencies will be installed automatically.")
            
            # Create a minimal object that will fail gracefully
            class MinimalApp:
                def __init__(self):
                    self.title = "NFL AI Platform - Dependencies Missing"
                    self.version = "1.0.0"
            
            app = MinimalApp()
            logger.warning("Created minimal app stub. Install dependencies for full functionality.")

# Vercel expects the app to be available as 'app' or 'handler'
# FastAPI apps work directly with Vercel's Python runtime
# Both names work, but 'app' is the standard
handler = app

logger.info("Vercel entry point initialized successfully")
