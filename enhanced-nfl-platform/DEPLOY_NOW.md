# Deploy Your Project Now (Without AWS RDS)

This guide helps you deploy your NFL AI Platform quickly. You can add AWS RDS later.

## 🚀 Quick Deployment Options

### Option 1: Docker Compose (Recommended for Local/Testing)
**Best for**: Running everything locally with one command

**Steps:**
1. Make sure Docker Desktop is running
2. Open terminal in `enhanced-nfl-platform` folder
3. Run:
   ```bash
   docker-compose up --build
   ```
4. Access:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

**What it includes:**
- MySQL database (auto-configured)
- Redis cache
- Backend API
- React frontend

**Note**: Uses local MySQL, not AWS RDS. Perfect for testing!

---

### Option 2: Render (Easiest Cloud Deployment)
**Best for**: Free cloud hosting with managed database

**Steps:**

1. **Sign up**: Go to [render.com](https://render.com) and sign up (free tier available)

2. **Create a Web Service:**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Name**: `nfl-ai-platform`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r backend/requirements.txt`
     - **Start Command**: `python backend/main.py` or `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

3. **Add Environment Variables** (in Render dashboard):
   ```
   PORT=8000
   DATABASE_URL=postgresql://... (Render provides free Postgres)
   REDIS_URL=redis://... (optional, Render provides free Redis)
   SECRET_KEY=your-secret-key-here
   TEST_MODE=false
   ```

4. **Add PostgreSQL Database** (free tier):
   - Click "New +" → "PostgreSQL"
   - Name it `nfl-db`
   - Copy the connection string to `DATABASE_URL` in your web service

5. **Deploy**: Click "Create Web Service"

**Access**: Your app will be live at `https://your-app-name.onrender.com`

---

### Option 3: Railway (Simple Cloud Deployment)
**Best for**: Easy deployment with automatic database setup

**Steps:**

1. **Sign up**: Go to [railway.app](https://railway.app) and sign up

2. **Deploy from GitHub:**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Add Database:**
   - Click "New" → "Database" → "PostgreSQL"
   - Railway automatically creates and connects it

4. **Set Environment Variables:**
   - Click on your service → "Variables"
   - Add:
     ```
     PORT=8000
     SECRET_KEY=your-secret-key
     TEST_MODE=false
     ```
   - `DATABASE_URL` is auto-set by Railway

5. **Deploy**: Railway auto-deploys on git push

**Access**: Your app will be live at `https://your-app-name.up.railway.app`

---

### Option 4: Heroku (Classic Cloud Platform)
**Best for**: Established platform with add-ons

**Steps:**

1. **Install Heroku CLI:**
   ```bash
   # Windows
   winget install Heroku.HerokuCLI
   ```

2. **Login and Create App:**
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Add PostgreSQL:**
   ```bash
   heroku addons:create heroku-postgresql:mini
   ```

4. **Set Environment Variables:**
   ```bash
   heroku config:set SECRET_KEY=your-secret-key
   heroku config:set TEST_MODE=false
   ```

5. **Deploy:**
   ```bash
   git push heroku main
   ```

**Access**: Your app will be live at `https://your-app-name.herokuapp.com`

---

## 📝 Pre-Deployment Checklist

Before deploying, make sure:

- [ ] Your code is committed to Git
- [ ] You have a `.env` file (or use platform's environment variables)
- [ ] Model files exist (if using ML features):
  - `models/qb_td_model.keras`
  - `models/feature_scaler.pkl`
  - `models/training_metrics.json`
- [ ] Dependencies are in `requirements.txt`
- [ ] You've tested locally first

## 🔧 Environment Variables Needed

For cloud deployment, set these in your platform's dashboard:

**Required:**
```
DATABASE_URL=postgresql://... (provided by platform)
SECRET_KEY=your-random-secret-key-here
PORT=8000
```

**Optional:**
```
REDIS_URL=redis://... (if using Redis)
OPENAI_API_KEY=... (if using RAG features)
TEST_MODE=false
```

## 🎯 Recommended: Start with Docker Compose

If you're not sure, start with **Docker Compose**:

```bash
cd enhanced-nfl-platform
docker-compose up --build
```

This gives you:
- ✅ Full stack running locally
- ✅ MySQL database included
- ✅ No cloud setup needed
- ✅ Easy to test and develop

Then when ready, deploy to Render or Railway for cloud hosting.

## 🔄 Adding AWS RDS Later

Once deployed, you can easily switch to AWS RDS:

1. Create your RDS instance in AWS
2. Update `DATABASE_URL` in your platform's environment variables
3. Add `DB_USE_SSL=true`
4. Redeploy

That's it! Your app will now use AWS RDS.

## 🆘 Need Help?

- **Docker issues**: Check `docker-compose.yml` and ensure Docker Desktop is running
- **Deployment errors**: Check platform logs in their dashboard
- **Database connection**: Verify `DATABASE_URL` is correct
- **Port issues**: Make sure `PORT` environment variable matches platform requirements

## 📚 More Details

- Full deployment guide: `DEPLOYMENT_GUIDE.md`
- AWS RDS setup (for later): `AWS_RDS_SETUP.md`

