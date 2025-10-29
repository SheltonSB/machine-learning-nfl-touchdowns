#!/bin/bash

echo "🚀 NFL AI Platform - Live Deployment"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_info "Choose your deployment platform:"
echo "1) Railway (Recommended - Free tier available)"
echo "2) Heroku (Free tier available)"
echo "3) Render (Free tier available)"
echo "4) Vercel (Free tier available)"
echo "5) DigitalOcean App Platform"
echo "6) AWS (Advanced)"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        print_info "Deploying to Railway..."
        print_info "1. Go to https://railway.app"
        print_info "2. Sign up with GitHub"
        print_info "3. Click 'New Project' -> 'Deploy from GitHub repo'"
        print_info "4. Select this repository"
        print_info "5. Railway will automatically detect the configuration"
        print_info "6. Your app will be live at: https://your-app-name.railway.app"
        ;;
    2)
        print_info "Deploying to Heroku..."
        print_info "1. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli"
        print_info "2. Run: heroku login"
        print_info "3. Run: heroku create your-nfl-ai-app"
        print_info "4. Run: git add ."
        print_info "5. Run: git commit -m 'Deploy NFL AI Platform'"
        print_info "6. Run: git push heroku main"
        print_info "7. Your app will be live at: https://your-nfl-ai-app.herokuapp.com"
        ;;
    3)
        print_info "Deploying to Render..."
        print_info "1. Go to https://render.com"
        print_info "2. Sign up with GitHub"
        print_info "3. Click 'New' -> 'Web Service'"
        print_info "4. Connect your GitHub repository"
        print_info "5. Set Build Command: pip install -r requirements.txt"
        print_info "6. Set Start Command: python backend/working_app.py"
        print_info "7. Click 'Create Web Service'"
        print_info "8. Your app will be live at: https://your-app-name.onrender.com"
        ;;
    4)
        print_info "Deploying to Vercel..."
        print_info "1. Go to https://vercel.com"
        print_info "2. Sign up with GitHub"
        print_info "3. Click 'New Project'"
        print_info "4. Import your GitHub repository"
        print_info "5. Vercel will auto-detect the configuration"
        print_info "6. Click 'Deploy'"
        print_info "7. Your app will be live at: https://your-app-name.vercel.app"
        ;;
    5)
        print_info "Deploying to DigitalOcean App Platform..."
        print_info "1. Go to https://cloud.digitalocean.com/apps"
        print_info "2. Click 'Create App'"
        print_info "3. Connect your GitHub repository"
        print_info "4. Select 'Web Service'"
        print_info "5. Set Build Command: pip install -r requirements.txt"
        print_info "6. Set Run Command: python backend/working_app.py"
        print_info "7. Click 'Create Resources'"
        print_info "8. Your app will be live at: https://your-app-name.ondigitalocean.app"
        ;;
    6)
        print_info "Deploying to AWS..."
        print_info "1. Go to AWS Elastic Beanstalk"
        print_info "2. Create new application"
        print_info "3. Upload your code as a ZIP file"
        print_info "4. Configure environment variables"
        print_info "5. Deploy the application"
        print_info "6. Your app will be live at: https://your-app-name.region.elasticbeanstalk.com"
        ;;
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

print_info "After deployment, update the API_BASE URL in professional_clean.html:"
print_info "Change: const API_BASE = 'https://your-nfl-ai-platform.herokuapp.com';"
print_info "To your actual deployed URL"

print_info "Deployment files created:"
print_status "✅ Procfile (Heroku)"
print_status "✅ requirements.txt (All platforms)"
print_status "✅ railway.json (Railway)"
print_status "✅ professional_clean.html (Clean frontend)"
print_status "✅ working_app.py (Backend)"

print_info "Your NFL AI Platform is ready for live deployment!"
print_info "Features:"
print_info "  - Advanced ML predictions for any player"
print_info "  - Google-style AI text completion"
print_info "  - Professional clean interface"
print_info "  - No emojis, clear black text"
print_info "  - Production-ready backend"
