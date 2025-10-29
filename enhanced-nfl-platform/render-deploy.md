# Deploy to Render with GitHub

## Quick Steps:
1. Go to https://render.com
2. Sign up with GitHub
3. Click "New" → "Web Service"
4. Connect `machine-learning-nfl-touchdowns` repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend/working_app.py`
6. Click "Create Web Service"

## What Render Uses:
- ✅ `requirements.txt` for dependencies
- ✅ Python backend detection
- ✅ Automatic build process

## Your App Will Be Live At:
`https://your-app-name.onrender.com`

## Features:
- Free tier available
- Automatic deployments on git push
- Custom domain support
- SSL certificates included
