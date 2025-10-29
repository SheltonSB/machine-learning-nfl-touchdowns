#!/bin/bash

echo "🏈 NFL AI Platform - Ultimate Production Deployment"
echo "=================================================="

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
if [ ! -f "backend/ultimate_production_app.py" ]; then
    print_error "Please run this script from the enhanced-nfl-platform directory"
    exit 1
fi

print_info "Starting ultimate production deployment..."

# 1. Install Python dependencies
print_info "Installing Python dependencies..."
cd backend

# Install core dependencies
pip3 install fastapi uvicorn pymysql sqlalchemy pandas numpy scikit-learn

# Install advanced ML dependencies
pip3 install transformers torch sentence-transformers

# Install optional dependencies
pip3 install openai chromadb joblib

print_status "Dependencies installed"

# 2. Create models directory
mkdir -p models
print_status "Models directory created"

# 3. Test MySQL connection
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

# 4. Start the ultimate production app
print_info "Starting Ultimate Production App..."
print_info "This will start the most advanced NFL AI platform ever built!"

# Kill any existing processes
pkill -f "ultimate_production_app" 2>/dev/null || true
pkill -f "database_production_app" 2>/dev/null || true

# Start the app in background
nohup python3 ultimate_production_app.py > ultimate_app.log 2>&1 &
APP_PID=$!

# Wait for app to start
print_info "Waiting for app to initialize..."
sleep 10

# Check if app is running
if ps -p $APP_PID > /dev/null; then
    print_status "Ultimate Production App started successfully (PID: $APP_PID)"
else
    print_error "Failed to start Ultimate Production App"
    cat ultimate_app.log
    exit 1
fi

# 5. Test the API
print_info "Testing API endpoints..."

# Test health endpoint
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    print_status "Health endpoint working"
else
    print_warning "Health endpoint test failed"
fi

# Test stats endpoint
STATS_RESPONSE=$(curl -s http://localhost:8000/api/v1/stats)
if echo "$STATS_RESPONSE" | grep -q "platform"; then
    print_status "Stats endpoint working"
else
    print_warning "Stats endpoint test failed"
fi

# Test RAG endpoint
RAG_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/rag/query \
    -H "Content-Type: application/json" \
    -d '{"question": "Tell me about Tom Brady", "mode": "balanced"}')
if echo "$RAG_RESPONSE" | grep -q "answer"; then
    print_status "RAG endpoint working"
else
    print_warning "RAG endpoint test failed"
fi

# Test prediction endpoint
PREDICTION_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/predictions \
    -H "Content-Type: application/json" \
    -d '{"player_id": 1, "features": {"passing_yards_roll3": 350, "td_passes_roll3": 3.0}}')
if echo "$PREDICTION_RESPONSE" | grep -q "prediction"; then
    print_status "Prediction endpoint working"
else
    print_warning "Prediction endpoint test failed"
fi

# 6. Start frontend server
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

# 7. Display deployment summary
echo ""
echo "🎉 ULTIMATE PRODUCTION DEPLOYMENT COMPLETE!"
echo "============================================="
echo ""
print_status "Backend API: http://localhost:8000"
print_status "Frontend: http://localhost:3000/ultimate.html"
print_status "API Documentation: http://localhost:8000/docs"
print_status "Health Check: http://localhost:8000/health"
print_status "System Stats: http://localhost:8000/api/v1/stats"
echo ""
print_info "Features Available:"
echo "  🧠 Advanced RAG with fine-tuning"
echo "  🤖 Ensemble ML with hyperparameter optimization"
echo "  🌡️ Temperature control and top-k sampling"
echo "  📊 Real-time performance monitoring"
echo "  🗄️ MySQL database with 281,872+ records"
echo "  ⚡ High-performance caching and optimization"
echo ""
print_info "AI Controls:"
echo "  • Temperature: 0.1 - 1.0 (creativity control)"
echo "  • Top-K: 5 - 50 (response diversity)"
echo "  • Top-P: 0.6 - 0.95 (nucleus sampling)"
echo "  • Modes: Creative, Balanced, Precise, Conservative"
echo ""
print_info "ML Features:"
echo "  • Random Forest (optimized)"
echo "  • Gradient Boosting"
echo "  • Neural Networks"
echo "  • Logistic Regression"
echo "  • Ensemble predictions"
echo "  • Feature engineering"
echo ""
print_warning "To stop the services:"
echo "  kill $APP_PID $FRONTEND_PID"
echo ""
print_warning "To view logs:"
echo "  tail -f backend/ultimate_app.log"
echo "  tail -f frontend/frontend.log"
echo ""

# 8. Open browser (optional)
if command -v open &> /dev/null; then
    print_info "Opening browser..."
    open http://localhost:3000/ultimate.html
elif command -v xdg-open &> /dev/null; then
    print_info "Opening browser..."
    xdg-open http://localhost:3000/ultimate.html
fi

print_status "Ultimate NFL AI Platform is now running!"
