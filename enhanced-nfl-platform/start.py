#!/usr/bin/env python3
"""
NFL AI Platform - Start Script for Render
"""
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "backend").exists() else CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the working app
if __name__ == "__main__":
    from backend.working_app import app
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
