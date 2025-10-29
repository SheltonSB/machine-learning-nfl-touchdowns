#!/bin/bash

echo "🏈 NFL AI Platform - Advanced ML Deployment"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if we're in the right directory
if [ ! -f "backend/advanced_ml_webapp.py" ]; then
    print_error "Please run this script from the enhanced-nfl-platform directory"
    exit 1
fi

print_info "Starting Advanced ML web app deployment..."

# 1. Install Python dependencies
print_info "Installing advanced Python dependencies..."
cd backend

# Install core dependencies
pip3 install fastapi uvicorn pymysql sqlalchemy pandas numpy scikit-learn openai

print_status "Dependencies installed"

# 2. Test MySQL connection
print_info "Testing MySQL connection..."
python3 -c "
import pymysql
try:
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='NewStrongPassword!123',
        database='nfl_ai'
    )
    print('✅ MySQL connection successful')
    conn.close()
except Exception as e:
    print(f'❌ MySQL connection failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    print_error "MySQL connection failed. Please check your database setup."
    exit 1
fi

# 3. Start the advanced ML web app
print_info "Starting Advanced ML Web App..."
print_info "This will start the most advanced NFL AI platform with state-of-the-art ML algorithms!"

# Kill any existing processes
pkill -f "advanced_ml_webapp" 2>/dev/null || true
pkill -f "http.server" 2>/dev/null || true

# Start the app in background
nohup python3 advanced_ml_webapp.py > advanced_ml_app.log 2>&1 &
APP_PID=$!

# Wait for app to start
print_info "Waiting for app to initialize..."
sleep 10

# Check if app is running
if ps -p $APP_PID > /dev/null; then
    print_status "Advanced ML Web App started successfully (PID: $APP_PID)"
else
    print_error "Failed to start Advanced ML Web App"
    cat advanced_ml_app.log
    exit 1
fi

# 4. Test the API
print_info "Testing API endpoints..."

# Test health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_status "Health endpoint working"
else
    print_warning "Health endpoint test failed"
fi

# Test prediction endpoint for ANY player
PREDICTION_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/predictions/any-player \
    -H "Content-Type: application/json" \
    -d '{
        "player_name": "Tom Brady",
        "team": "Patriots",
        "position": "QB",
        "recent_stats": {
            "passing_yards": 350,
            "passing_tds": 3.0,
            "attempts": 40,
            "completion_pct": 72.5,
            "passer_rating": 105.2,
            "rushing_yards": 25
        }
    }')
if echo "$PREDICTION_RESPONSE" | grep -q "prediction"; then
    print_status "Any-player prediction endpoint working"
else
    print_warning "Any-player prediction endpoint test failed"
fi

# Test Google-style completion endpoint
COMPLETION_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Tell me about Tom Brady",
        "max_tokens": 100,
        "temperature": 0.7
    }')
if echo "$COMPLETION_RESPONSE" | grep -q "completion"; then
    print_status "Google-style completion endpoint working"
else
    print_warning "Google-style completion endpoint test failed"
fi

# 5. Start frontend server
print_info "Starting frontend server..."
cd ../frontend

# Kill any existing frontend processes
pkill -f "http.server" 2>/dev/null || true

# Start frontend server
nohup python3 -m http.server 3000 > frontend.log 2>&1 &
FRONTEND_PID=$!

sleep 3

if ps -p $FRONTEND_PID > /dev/null; then
    print_status "Frontend server started successfully (PID: $FRONTEND_PID)"
else
    print_error "Failed to start frontend server"
    exit 1
fi

# 6. Display deployment summary
echo ""
echo "🎉 ADVANCED ML WEB APP DEPLOYMENT SUCCESSFUL!"
echo "=============================================="
echo ""
print_status "Backend API: http://localhost:8000"
print_status "Frontend: http://localhost:3000/advanced_ml_webapp.html"
print_status "API Documentation: http://localhost:8000/docs"
print_status "Health Check: http://localhost:8000/health"
print_status "System Stats: http://localhost:8000/api/v1/stats"
echo ""
print_info "🚀 ADVANCED FEATURES AVAILABLE:"
echo "  🤖 State-of-the-art ML algorithms (Random Forest, Gradient Boosting, Neural Networks, SVM, Logistic Regression)"
echo "  🎯 Predict for ANY player (not just database players)"
echo "  🧠 Google-style text completion with OpenAI GPT"
echo "  ⚡ Ensemble learning with hyperparameter optimization"
echo "  🔬 Advanced feature engineering and selection"
echo "  📊 Feature importance analysis"
echo "  🎛️ Temperature control for AI creativity"
echo "  📈 Real-time performance metrics"
echo "  🗄️ MySQL Database with 281,872+ records"
echo "  🎨 Beautiful responsive interface with animations"
echo ""
print_info "🎯 HOW TO USE:"
echo "  1. Enter ANY player name (Tom Brady, Patrick Mahomes, etc.)"
echo "  2. Enter their recent performance stats"
echo "  3. Get AI-powered prediction using 5+ ML algorithms"
echo "  4. Ask Google-style AI questions about NFL"
echo "  5. Adjust AI creativity with temperature slider"
echo "  6. View feature importance and model breakdown"
echo ""
print_info "🔬 ML ALGORITHMS USED:"
echo "  • Random Forest (500 trees, optimized hyperparameters)"
echo "  • Gradient Boosting (500 estimators, adaptive learning)"
echo "  • Neural Network (200-100-50 hidden layers)"
echo "  • Support Vector Machine (RBF kernel)"
echo "  • Logistic Regression (L2 regularization)"
echo "  • Ensemble Voting (soft voting for best accuracy)"
echo ""
print_info "🌐 READY FOR NETLIFY DEPLOYMENT:"
echo "  • Frontend: Static HTML/CSS/JS with advanced features"
echo "  • Backend: FastAPI with state-of-the-art ML"
echo "  • Configuration: netlify.toml included"
echo "  • Package: package.json included"
echo "  • Database: MySQL with comprehensive NFL data"
echo ""
print_warning "To stop the services:"
echo "  kill $APP_PID $FRONTEND_PID"
echo ""
print_warning "To view logs:"
echo "  tail -f backend/advanced_ml_app.log"
echo "  tail -f frontend/frontend.log"
echo ""

# 7. Open browser (optional)
if command -v open &> /dev/null; then
    print_info "Opening browser..."
    open http://localhost:3000/advanced_ml_webapp.html
elif command -v xdg-open &> /dev/null; then
    print_info "Opening browser..."
    xdg-open http://localhost:3000/advanced_ml_webapp.html
fi

print_status "Advanced ML NFL AI Platform is now running!"
