# Deploy to Vercel via Website (No CLI Needed!)

Since you don't have npm/Node.js installed, you can deploy directly from the Vercel website - **no CLI required!**

## 🚀 Deploy from Vercel Website

### Step 1: Push Your Code to GitHub

1. Make sure your code is committed:
   ```powershell
   git add .
   git commit -m "Ready for Vercel deployment"
   git push origin main
   ```

2. If you don't have a GitHub repo yet:
   - Go to [github.com](https://github.com)
   - Create a new repository
   - Push your code:
     ```powershell
     git remote add origin https://github.com/yourusername/your-repo.git
     git push -u origin main
     ```

### Step 2: Deploy on Vercel Website

1. **Go to Vercel**: [vercel.com](https://vercel.com)

2. **Sign up/Login**: 
   - Click "Sign Up" or "Log In"
   - Use GitHub to sign in (recommended - connects automatically)

3. **Import Project**:
   - Click "Add New..." → "Project"
   - Click "Import Git Repository"
   - Select your GitHub repository
   - Click "Import"

4. **Configure Project**:
   - **Project Name**: `nfl-ai-platform` (or your choice)
   - **Root Directory**: Leave as `./` (root)
   - **Framework Preset**: Vercel will auto-detect
   - **Build Command**: Leave default (auto-detected)
   - **Output Directory**: Leave default
   - **Install Command**: Leave default

5. **Environment Variables** (Add these):
   Click "Environment Variables" and add:
   
   **Required:**
   ```
   DATABASE_URL = your-database-url-here
   SECRET_KEY = your-random-secret-key-here
   ```
   
   **Optional:**
   ```
   REDIS_URL = your-redis-url (if using)
   OPENAI_API_KEY = your-key (if using RAG)
   TEST_MODE = false
   PORT = 8000
   ```

6. **Deploy**:
   - Click "Deploy"
   - Wait for build to complete (2-5 minutes)
   - Your app will be live!

### Step 3: Get Your Database URL

Since you need a database, here are quick options:

#### Option A: Render (Free PostgreSQL - 5 minutes)

1. Go to [render.com](https://render.com)
2. Sign up (free)
3. Click "New +" → "PostgreSQL"
4. Name it `nfl-db`
5. Copy the **Internal Database URL** (for Vercel)
6. Paste into Vercel's `DATABASE_URL` environment variable

#### Option B: Railway (Free PostgreSQL - 3 minutes)

1. Go to [railway.app](https://railway.app)
2. Sign up (free)
3. Click "New Project" → "Deploy from GitHub"
4. Add PostgreSQL database
5. Copy connection string
6. Paste into Vercel's `DATABASE_URL`

### Step 4: Update Environment Variables

After deployment:

1. Go to your project on Vercel
2. Click **Settings** → **Environment Variables**
3. Add/Update variables:
   - `DATABASE_URL` = your database connection string
   - `SECRET_KEY` = generate a random string
4. Click **Redeploy** to apply changes

## ✅ That's It!

Your app will be live at:
- **Production**: `https://your-project-name.vercel.app`
- **Preview URLs**: Each commit gets a preview URL

## 🔄 Updating Your App

Just push to GitHub:
```powershell
git add .
git commit -m "Update app"
git push origin main
```

Vercel automatically deploys on every push!

## 🆘 Troubleshooting

### Build Fails

- Check build logs in Vercel dashboard
- Make sure `vercel.json` is in root directory
- Verify all files are committed to Git

### API Not Working

- Check that environment variables are set
- Verify database URL is correct
- Check function logs in Vercel dashboard

### Frontend Not Loading

- Check that `frontend/package.json` exists
- Verify build completed successfully
- Check browser console for errors

## 📝 Quick Checklist

Before deploying:
- [ ] Code pushed to GitHub
- [ ] `vercel.json` is in root directory
- [ ] Database URL ready (Render/Railway)
- [ ] Environment variables list ready

## 💡 Pro Tips

1. **Preview Deployments**: Every commit gets a preview URL - test before merging!
2. **Environment Variables**: Set different values for Production/Preview/Development
3. **Monitor Logs**: Check Vercel dashboard for real-time logs
4. **Auto Deploy**: Every git push = automatic deployment

---

**No CLI needed - just use the website!** 🎉

