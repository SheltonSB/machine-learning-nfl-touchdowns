#!/bin/bash

# NFL AI/ML Platform Setup Script
# This script sets up the complete development environment

set -e

echo "🏈 Setting up NFL AI/ML Platform..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker is installed"
}

# Check if Docker Compose is installed
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose is installed"
}

# Check if Node.js is installed
check_node() {
    if ! command -v node &> /dev/null; then
        print_warning "Node.js is not installed. Installing via Docker..."
        return 1
    fi
    print_status "Node.js is installed"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    print_status "Python 3 is installed"
}

# Create environment files
create_env_files() {
    print_status "Creating environment files..."
    
    # Backend .env
    if [ ! -f backend/.env ]; then
        cat > backend/.env << EOF
# Database
DATABASE_URL=postgresql://nfl_user:nfl_password@localhost:5432/nfl_platform
REDIS_URL=redis://localhost:6379

# JWT
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# ML Models
MODEL_PATH=models/
XGBOOST_MODEL_PATH=models/xgboost_model.pkl
TENSORFLOW_MODEL_PATH=models/tensorflow_model.h5
PYTORCH_MODEL_PATH=models/pytorch_model.pth

# RAG System
VECTOR_DB_API_KEY=your-pinecone-api-key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=microsoft/DialoGPT-small

# AWS (for production)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1
S3_BUCKET=

# Logging
LOG_LEVEL=INFO
EOF
        print_status "Created backend/.env"
    fi
    
    # Frontend .env
    if [ ! -f frontend/.env ]; then
        cat > frontend/.env << EOF
REACT_APP_API_URL=http://localhost:8000
REACT_APP_VERSION=1.0.0
EOF
        print_status "Created frontend/.env"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    cd backend
    pip install -r requirements.txt
    cd ..
    print_status "Python dependencies installed"
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    cd frontend
    npm install
    cd ..
    print_status "Node.js dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    mkdir -p backend/models
    mkdir -p backend/data/raw
    mkdir -p backend/data/processed
    mkdir -p logs
    print_status "Directories created"
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    docker-compose up -d db redis
    sleep 10  # Wait for services to start
    
    # Run database migrations
    cd backend
    python -c "
from app.core.database import engine, Base
Base.metadata.create_all(bind=engine)
print('Database initialized')
"
    cd ..
    print_status "Database initialized"
}

# Build Docker images
build_docker_images() {
    print_status "Building Docker images..."
    docker-compose build
    print_status "Docker images built"
}

# Start services
start_services() {
    print_status "Starting services..."
    docker-compose up -d
    print_status "Services started"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for backend
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            print_status "Backend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Backend failed to start"
        exit 1
    fi
    
    # Wait for frontend
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 &> /dev/null; then
            print_status "Frontend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Frontend failed to start"
        exit 1
    fi
}

# Main setup function
main() {
    print_status "Starting NFL AI/ML Platform setup..."
    
    # Check prerequisites
    check_docker
    check_docker_compose
    check_python
    
    # Create environment files
    create_env_files
    
    # Create directories
    create_directories
    
    # Install dependencies
    install_python_deps
    if check_node; then
        install_node_deps
    fi
    
    # Initialize database
    init_database
    
    # Build Docker images
    build_docker_images
    
    # Start services
    start_services
    
    # Wait for services
    wait_for_services
    
    print_status "🎉 Setup complete!"
    echo ""
    echo "Your NFL AI/ML Platform is now running:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo ""
    echo "To stop the services, run: docker-compose down"
    echo "To view logs, run: docker-compose logs -f"
}

# Run main function
main "$@"

