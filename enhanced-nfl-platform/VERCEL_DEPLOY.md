# Deploying to Vercel

## ⚠️ Important: Vercel Deployment Options

You have **two options** for deploying to Vercel:

### Option 1: Frontend Only (Recommended)
Deploy just the React frontend to Vercel, and deploy the backend separately (Render, Railway, etc.)

### Option 2: Full Stack (Advanced)
Deploy both frontend and backend to Vercel using serverless functions

---

## Option 1: Frontend Only Deployment (Recommended)

This is the **easiest and most reliable** approach.

### Step 1: Deploy Backend First (Required)

Deploy your backend to one of these platforms:
- **Render**: https://render.com (free tier available)
- **Railway**: https://railway.app (free tier available)
- **Heroku**: https://heroku.com

Get your backend URL (e.g., `https://your-backend.onrender.com`)

### Step 2: Configure Vercel for Frontend

1. **Create `vercel.json` in the `frontend/` directory:**

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "build",
  "devCommand": "npm start",
  "installCommand": "npm install",
  "framework": "create-react-app",
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```

2. **Set Environment Variables in Vercel Dashboard:**
   - Go to your project → Settings → Environment Variables
   - Add: `REACT_APP_API_BASE_URL` = `https://your-backend.onrender.com`

3. **Deploy:**
   ```bash
   cd frontend
   vercel
   ```

### Step 3: Deploy from Vercel Dashboard

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository
4. **Root Directory**: Set to `frontend`
5. **Build Command**: `npm run build`
6. **Output Directory**: `build`
7. **Environment Variables**: Add `REACT_APP_API_BASE_URL` = your backend URL
8. Click "Deploy"

---

## Option 2: Full Stack Deployment (Advanced)

If you want to deploy both frontend and backend to Vercel:

### Configuration

The current `vercel.json` in the root is set up for this, but it requires:

1. **Backend as Serverless Functions:**
   - FastAPI needs to be adapted for serverless
   - Current setup uses `start.py` which should work

2. **Frontend Build:**
   - Need to configure both frontend and backend builds

3. **Updated `vercel.json`:**

```json
{
  "version": 2,
  "builds": [
    {
      "src": "frontend/package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    },
    {
      "src": "start.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "start.py"
    },
    {
      "src": "/(.*)",
      "dest": "frontend/build/$1"
    }
  ],
  "env": {
    "REACT_APP_API_BASE_URL": "/api"
  }
}
```

### Limitations of Full Stack on Vercel:

- ⚠️ Serverless functions have execution time limits (10s on free tier)
- ⚠️ Database connections may timeout (better for serverless databases)
- ⚠️ ML model loading can be slow in serverless
- ⚠️ More complex configuration

---

## ✅ Recommended Setup

**Deploy Frontend to Vercel + Backend to Render/Railway**

This gives you:
- ✅ Fast frontend hosting (Vercel CDN)
- ✅ Reliable backend (no serverless limits)
- ✅ Easy to configure
- ✅ Better performance

### Quick Steps:

1. **Deploy Backend to Render:**
   - Go to render.com
   - Create Web Service
   - Connect GitHub repo
   - Root Directory: `enhanced-nfl-platform`
   - Build: `pip install -r backend/requirements.txt`
   - Start: `python backend/main.py`
   - Get URL: `https://your-app.onrender.com`

2. **Deploy Frontend to Vercel:**
   - Go to vercel.com
   - New Project → Import repo
   - Root Directory: `frontend`
   - Build: `npm run build`
   - Output: `build`
   - Environment Variable: `REACT_APP_API_BASE_URL` = `https://your-app.onrender.com`

3. **Done!** Your frontend will call your backend automatically.

---

## Current Status Check

Your project has:
- ✅ React frontend ready (`frontend/`)
- ✅ FastAPI backend ready (`backend/`)
- ✅ `vercel.json` exists (configured for backend)
- ⚠️ Need to decide: Frontend-only or Full-stack?

## Recommendation

**Start with Option 1 (Frontend Only)**:
1. Deploy backend to Render (5 minutes)
2. Deploy frontend to Vercel (5 minutes)
3. Connect them with environment variable

This is the most reliable and performant setup!

