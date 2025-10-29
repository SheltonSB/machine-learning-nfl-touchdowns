# Deploy to Heroku with GitHub

## Quick Steps:
1. Go to https://heroku.com
2. Sign up with GitHub
3. Click "New" → "Create new app"
4. Connect to `machine-learning-nfl-touchdowns` repository
5. Enable "Deploy from GitHub"
6. Click "Deploy Branch"

## What Heroku Uses:
- ✅ `Procfile` for start command
- ✅ `requirements.txt` for dependencies
- ✅ `runtime.txt` for Python version
- ✅ `heroku.yml` for build configuration

## Your App Will Be Live At:
`https://your-app-name.herokuapp.com`

## Manual Deploy Commands:
```bash
# If you want to deploy manually
heroku login
heroku create your-nfl-ai-app
git push heroku main
```
