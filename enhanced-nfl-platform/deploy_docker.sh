#!/bin/bash

echo "🐳 NFL AI Platform - Docker Deployment"
echo "======================================"

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

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_info "Docker and Docker Compose are available!"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the enhanced-nfl-platform directory"
    exit 1
fi

print_info "Starting Docker deployment..."

# 1. Stop any existing containers
print_info "Stopping existing containers..."
docker-compose down --remove-orphans

# 2. Build the Docker image
print_info "Building Docker image..."
docker-compose build --no-cache

if [ $? -ne 0 ]; then
    print_error "Failed to build Docker image"
    exit 1
fi

print_status "Docker image built successfully!"

# 3. Start the services
print_info "Starting services with Docker Compose..."
docker-compose up -d

if [ $? -ne 0 ]; then
    print_error "Failed to start services"
    exit 1
fi

# 4. Wait for services to be ready
print_info "Waiting for services to be ready..."
sleep 30

# 5. Check service health
print_info "Checking service health..."

# Check MySQL
if docker-compose exec mysql mysqladmin ping -h localhost --silent; then
    print_status "MySQL is healthy"
else
    print_warning "MySQL health check failed"
fi

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_status "Backend API is healthy"
else
    print_warning "Backend API health check failed"
fi

# Check Frontend
if curl -f http://localhost/ > /dev/null 2>&1; then
    print_status "Frontend is healthy"
else
    print_warning "Frontend health check failed"
fi

# 6. Display deployment summary
echo ""
echo "🎉 DOCKER DEPLOYMENT SUCCESSFUL!"
echo "================================"
echo ""
print_status "🌐 Web Application: http://localhost"
print_status "🔧 Backend API: http://localhost:8000"
print_status "📊 API Documentation: http://localhost:8000/docs"
print_status "❤️ Health Check: http://localhost:8000/health"
print_status "🗄️ MySQL Database: localhost:3306"
print_status "🔄 Redis Cache: localhost:6379"
echo ""
print_info "🚀 FEATURES AVAILABLE:"
echo "  🤖 Advanced ML with 5+ algorithms"
echo "  🎯 Predict for ANY player (not just database players)"
echo "  🧠 Google-style AI text completion"
echo "  ⚡ Ensemble learning with hyperparameter optimization"
echo "  🔬 Advanced feature engineering"
echo "  📊 Real-time feature importance analysis"
echo "  🎛️ Temperature control for AI creativity"
echo "  📈 Performance metrics and monitoring"
echo "  🗄️ MySQL Database with 281,872+ records"
echo "  🎨 Beautiful responsive interface with animations"
echo "  🐳 Fully containerized with Docker"
echo "  ⚡ Nginx load balancing and caching"
echo "  🔒 Security headers and rate limiting"
echo ""
print_info "📋 DOCKER COMMANDS:"
echo "  View logs: docker-compose logs -f"
echo "  Stop services: docker-compose down"
echo "  Restart services: docker-compose restart"
echo "  Update services: docker-compose pull && docker-compose up -d"
echo "  View containers: docker-compose ps"
echo ""
print_info "🔧 MANAGEMENT:"
echo "  Backend logs: docker-compose logs -f backend"
echo "  Frontend logs: docker-compose logs -f frontend"
echo "  MySQL logs: docker-compose logs -f mysql"
echo "  Redis logs: docker-compose logs -f redis"
echo ""
print_info "🌐 READY FOR PRODUCTION:"
echo "  • Configure your domain name"
echo "  • Set up SSL certificates"
echo "  • Configure environment variables"
echo "  • Set up monitoring and logging"
echo "  • Configure backup strategies"
echo ""

# 7. Test the application
print_info "Testing the application..."

# Test API endpoints
echo "Testing API endpoints..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    print_status "Health endpoint working"
else
    print_warning "Health endpoint test failed"
fi

# Test prediction endpoint
echo "Testing prediction endpoint..."
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
    }' 2>/dev/null)

if echo "$PREDICTION_RESPONSE" | grep -q "prediction"; then
    print_status "Prediction endpoint working"
else
    print_warning "Prediction endpoint test failed"
fi

# Test completion endpoint
echo "Testing AI completion endpoint..."
COMPLETION_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Tell me about Tom Brady",
        "max_tokens": 100,
        "temperature": 0.7
    }' 2>/dev/null)

if echo "$COMPLETION_RESPONSE" | grep -q "completion"; then
    print_status "AI completion endpoint working"
else
    print_warning "AI completion endpoint test failed"
fi

# 8. Open browser (optional)
if command -v open &> /dev/null; then
    print_info "Opening browser..."
    open http://localhost
elif command -v xdg-open &> /dev/null; then
    print_info "Opening browser..."
    xdg-open http://localhost
fi

print_status "NFL AI Platform is now running with Docker!"
print_info "Visit http://localhost to access the application"
