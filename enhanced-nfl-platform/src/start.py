#!/usr/bin/env python3
"""
NFL AI Platform - Start Script for Render
"""
import sys
import os

# Add the enhanced-nfl-platform directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'enhanced-nfl-platform'))

# Import and run the working app
if __name__ == "__main__":
    from backend.working_app import app
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)
