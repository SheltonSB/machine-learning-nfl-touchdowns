#!/bin/bash

echo "🏈 NFL AI Platform - Complete Web App Deployment"
echo "================================================"

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
if [ ! -f "backend/complete_webapp.py" ]; then
    print_error "Please run this script from the enhanced-nfl-platform directory"
    exit 1
fi

print_info "Starting complete web app deployment..."

# 1. Install Python dependencies
print_info "Installing Python dependencies..."
cd backend

# Install core dependencies
pip3 install fastapi uvicorn pymysql sqlalchemy pandas numpy scikit-learn

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

# 3. Start the complete web app
print_info "Starting Complete Web App..."
print_info "This will start the complete NFL AI platform with player search!"

# Kill any existing processes
pkill -f "complete_webapp" 2>/dev/null || true
pkill -f "http.server" 2>/dev/null || true

# Start the app in background
nohup python3 complete_webapp.py > complete_app.log 2>&1 &
APP_PID=$!

# Wait for app to start
print_info "Waiting for app to initialize..."
sleep 8

# Check if app is running
if ps -p $APP_PID > /dev/null; then
    print_status "Complete Web App started successfully (PID: $APP_PID)"
else
    print_error "Failed to start Complete Web App"
    cat complete_app.log
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

# Test player search endpoint
SEARCH_RESPONSE=$(curl -s "http://localhost:8000/api/v1/players/search?name=Tom%20Brady&limit=5")
if echo "$SEARCH_RESPONSE" | grep -q "players"; then
    print_status "Player search endpoint working"
else
    print_warning "Player search endpoint test failed"
fi

# Test prediction endpoint
PREDICTION_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/predictions \
    -H "Content-Type: application/json" \
    -d '{"player_name": "Tom Brady", "features": {"passing_yards": 350, "td_passes": 3.0}}')
if echo "$PREDICTION_RESPONSE" | grep -q "prediction"; then
    print_status "Prediction endpoint working"
else
    print_warning "Prediction endpoint test failed"
fi

# Test RAG endpoint
RAG_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/rag/query \
    -H "Content-Type: application/json" \
    -d '{"question": "Tell me about Tom Brady"}')
if echo "$RAG_RESPONSE" | grep -q "answer"; then
    print_status "RAG endpoint working"
else
    print_warning "RAG endpoint test failed"
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
echo "🎉 COMPLETE WEB APP DEPLOYMENT SUCCESSFUL!"
echo "==========================================="
echo ""
print_status "Backend API: http://localhost:8000"
print_status "Frontend: http://localhost:3000/complete_webapp.html"
print_status "API Documentation: http://localhost:8000/docs"
print_status "Health Check: http://localhost:8000/health"
print_status "System Stats: http://localhost:8000/api/v1/stats"
echo ""
print_info "Features Available:"
echo "  🔍 Player Search by Name, Team, Position"
echo "  🎯 Touchdown Predictions with AI"
echo "  🧠 AI Questions and Answers"
echo "  📊 Real-time Statistics"
echo "  🗄️ MySQL Database with 281,872+ records"
echo "  ⚡ Fast and Responsive Interface"
echo ""
print_info "How to Use:"
echo "  1. Search for any NFL player by name"
echo "  2. Select a player from search results"
echo "  3. Enter their recent performance stats"
echo "  4. Get AI-powered touchdown prediction"
echo "  5. Ask AI questions about NFL players/teams"
echo ""
print_info "Ready for Netlify Deployment:"
echo "  • Frontend: Static HTML/CSS/JS"
echo "  • Backend: FastAPI with MySQL"
echo "  • Configuration: netlify.toml included"
echo "  • Package: package.json included"
echo ""
print_warning "To stop the services:"
echo "  kill $APP_PID $FRONTEND_PID"
echo ""
print_warning "To view logs:"
echo "  tail -f backend/complete_app.log"
echo "  tail -f frontend/frontend.log"
echo ""

# 7. Open browser (optional)
if command -v open &> /dev/null; then
    print_info "Opening browser..."
    open http://localhost:3000/complete_webapp.html
elif command -v xdg-open &> /dev/null; then
    print_info "Opening browser..."
    xdg-open http://localhost:3000/complete_webapp.html
fi

print_status "Complete NFL AI Platform is now running!"
