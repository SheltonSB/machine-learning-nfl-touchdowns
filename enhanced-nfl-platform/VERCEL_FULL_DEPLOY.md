# Deploy Full Application to Vercel

## ✅ Your Application is Now Ready!

I've updated `vercel.json` to deploy **both frontend and backend together** on Vercel.

## 📋 What's Configured

The `vercel.json` now:
- ✅ Builds your React frontend (`frontend/`)
- ✅ Deploys your FastAPI backend as serverless functions (`start.py`)
- ✅ Routes `/api/*` to backend
- ✅ Routes everything else to frontend
- ✅ Sets `REACT_APP_API_BASE_URL=/api` automatically

## 🚀 Deployment Steps

### Step 1: Install Vercel CLI (if not installed)

```bash
npm install -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Deploy from Project Root

```bash
cd enhanced-nfl-platform
vercel
```

Follow the prompts:
- **Set up and deploy?** → Yes
- **Which scope?** → Your account
- **Link to existing project?** → No (first time) or Yes (updates)
- **Project name?** → `nfl-ai-platform` (or your choice)
- **Directory?** → `./` (current directory)
- **Override settings?** → No

### Step 4: Set Environment Variables

After first deployment, set environment variables in Vercel Dashboard:

1. Go to your project on [vercel.com](https://vercel.com)
2. Click **Settings** → **Environment Variables**
3. Add these variables:

**Required:**
```
DATABASE_URL=your-database-url
SECRET_KEY=your-secret-key-here
```

**Optional (but recommended):**
```
REDIS_URL=your-redis-url (if using Redis)
OPENAI_API_KEY=your-openai-key (if using RAG)
TEST_MODE=false
PORT=8000
```

**For AWS RDS (if using later):**
```
DB_HOST=your-rds-endpoint
DB_USER=your-username
DB_PASSWORD=your-password
DB_NAME=your-database
DB_USE_SSL=true
```

### Step 5: Redeploy

After adding environment variables:

```bash
vercel --prod
```

Or trigger a new deployment from the Vercel dashboard.

## 🎯 How It Works

1. **Frontend**: React app is built and served as static files
2. **Backend**: FastAPI runs as serverless functions
3. **Routing**:
   - `/api/*` → Goes to FastAPI backend
   - `/*` → Serves React frontend
4. **API Calls**: Frontend automatically uses `/api` as base URL

## ⚠️ Important Notes

### Serverless Limitations

Vercel serverless functions have:
- **Execution timeout**: 10 seconds (free tier), 60 seconds (pro)
- **Cold starts**: First request may be slower
- **Memory limits**: 1024 MB (free tier)

### Database Considerations

- **SQLite**: Won't work on Vercel (read-only filesystem)
- **PostgreSQL/MySQL**: Use external database (Render, Railway, AWS RDS)
- **Connection pooling**: May need adjustment for serverless

### ML Models

- Large models may exceed timeout limits
- Consider lazy loading or model optimization
- Use `TEST_MODE=true` to skip heavy initialization if needed

## 🔧 Troubleshooting

### Build Fails

**Frontend build error:**
```bash
# Test locally first
cd frontend
npm install
npm run build
```

**Backend import error:**
- Check that `start.py` can import your app
- Verify all dependencies in `backend/requirements.txt`

### API Routes Not Working

- Check that routes start with `/api/`
- Verify `start.py` exports the FastAPI app correctly
- Check Vercel function logs in dashboard

### Environment Variables Not Working

- Variables must be set in Vercel dashboard
- Redeploy after adding variables
- Use `vercel env pull` to sync locally

## 📝 Pre-Deployment Checklist

- [ ] Code committed to Git
- [ ] `vercel.json` is in root directory
- [ ] Frontend builds successfully (`cd frontend && npm run build`)
- [ ] Backend imports work (`python start.py` should not error)
- [ ] Database URL configured (external database required)
- [ ] Environment variables ready to add
- [ ] Tested locally if possible

## 🎉 After Deployment

Your app will be live at:
- **Production**: `https://your-project.vercel.app`
- **Preview**: Each commit gets a preview URL

## 🔄 Updating Deployment

```bash
# Deploy to production
vercel --prod

# Or just push to main branch (if connected to GitHub)
git push origin main
```

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- [Vercel Environment Variables](https://vercel.com/docs/concepts/projects/environment-variables)

## 💡 Pro Tips

1. **Use Preview Deployments**: Every PR gets a preview URL - test before merging!
2. **Monitor Function Logs**: Check Vercel dashboard for errors
3. **Optimize Cold Starts**: Keep dependencies minimal
4. **Database Connection**: Use connection pooling for serverless
5. **Environment Variables**: Use different values for production/preview

---

**Ready to deploy?** Run `vercel` from the project root! 🚀

