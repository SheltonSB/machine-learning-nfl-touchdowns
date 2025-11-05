# Deployment Guide

## Quick Start with Docker Compose

### Prerequisites
- Docker Desktop (or Docker Engine) with Compose support
- Git
- Trained model artifacts in the repository `models/` directory (`qb_td_model.keras`, `feature_scaler.pkl`, `training_metrics.json`). Generate them with:

  ```bash
  python main.py --workflow --train-model --generate-shap
  ```

### 1. Clone and Prepare
```bash
git clone <your-repo-url>
cd machine-learning-nfl-touchdowns
python main.py --workflow --train-model --generate-shap
cd enhanced-nfl-platform
```

### 2. Verify artifacts & tests
```bash
# From repository root
make backend-test
make frontend-test

# Optional: regenerate static landing pages
python enhanced-nfl-platform/frontend/landing/build_landing_pages.py
```

### 3. Prepare production artifacts
- Backend artifacts: ensure the following files are present and committed/bundled with deploys:
  - `models/qb_td_model.keras`
  - `models/feature_scaler.pkl`
  - `models/training_metrics.json`
  - `models/shap_summary.png`
- Dependency lock: the backend relies on `matplotlib==3.8.2` and `shap==0.44.1` (see `backend/requirements.txt`). If you maintain platform-specific lock files (e.g., Pipenv, Poetry), include these packages there as well.
- Coverage gate: `make backend-test` enforces ≥80% coverage. The `.coveragerc` in `backend/` omits experimental modules (ML training, RAG orchestration) from coverage calculations; keep those exclusions unless you add tests for them.
- Frontend unit tests must still pass after any UI changes.

### 4. Launch the stack
```bash
docker-compose up --build
```

### 5. Access the Application
- Web app: http://localhost:3000
- API: http://localhost:8000
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

## 🌐 Cloud Deployment Options

### Option 1: Railway (Recommended)
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the `railway.json` configuration
3. Set environment variables in Railway dashboard
4. Deploy with one click!

### Option 2: Heroku
1. Install Heroku CLI
2. Login to Heroku: `heroku login`
3. Create app: `heroku create your-nfl-ai-app`
4. Deploy: `git push heroku main`

### Option 3: Vercel
1. Connect your GitHub repository to Vercel
2. Vercel will automatically detect the `vercel.json` configuration
3. Deploy with automatic builds

### Option 4: DigitalOcean App Platform
1. Connect your GitHub repository
2. Select Docker as the build method
3. Configure environment variables
4. Deploy

## 🔧 Environment Variables

Create a `.env` file with:
```env
DATABASE_URL=mysql+pymysql://nfl_user:nfl_password@mysql:3306/nfl_ai
MYSQL_HOST=mysql
MYSQL_USER=nfl_user
MYSQL_PASSWORD=nfl_password
MYSQL_DATABASE=nfl_ai
PORT=8000
HOST=0.0.0.0
MODEL_PATH=/app/models
TENSORFLOW_MODEL_PATH=/app/models/qb_td_model.keras
TENSORFLOW_SCALER_PATH=/app/models/feature_scaler.pkl
TENSORFLOW_METRICS_PATH=/app/models/training_metrics.json
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
REDIS_URL=redis://redis:6379/0
OPENAI_API_KEY=<optional-if-RAG-enabled>
ENVIRONMENT=production
MODEL_ARTIFACT_MOUNT=/app/models
```
Mount or copy the local `models/` directory into the container path referenced by `MODEL_ARTIFACT_MOUNT`.

## 🗄️ Database Setup

### MySQL Configuration
- **Host**: mysql (or your database host)
- **Port**: 3306
- **Database**: nfl_ai
- **Username**: nfl_user
- **Password**: nfl_password

### Data Loading
The application will automatically load NFL data from CSV files on first startup.

## 🎯 Features Available

### 🤖 Advanced ML System
- **5+ ML Algorithms**: Random Forest, Gradient Boosting, Neural Networks, SVM, Logistic Regression
- **Ensemble Learning**: Combines all models for best accuracy
- **Hyperparameter Optimization**: Automatically tunes model parameters
- **Feature Engineering**: Advanced feature selection and creation

### 🎯 Prediction System
- **Any Player Prediction**: Works with ANY NFL player, not just database players
- **Real-time Analysis**: Instant predictions with confidence scores
- **Feature Importance**: Shows which stats matter most
- **Model Breakdown**: See individual model predictions

### 🧠 AI Text Completion
- **Google-style AI**: Natural language processing
- **Temperature Control**: Adjust AI creativity level
- **Context Awareness**: Understands NFL terminology
- **Multiple Response Lengths**: Short, medium, or long responses

### 📊 Real-time Features
- **Live Statistics**: Real-time system performance
- **Health Monitoring**: Automatic health checks
- **Performance Metrics**: ML model accuracy tracking
- **Usage Analytics**: Track predictions and queries

## 🔒 Security Features

- **Rate Limiting**: Prevents API abuse
- **Security Headers**: XSS, CSRF protection
- **Input Validation**: Comprehensive data validation
- **SQL Injection Protection**: Parameterized queries
- **CORS Configuration**: Secure cross-origin requests

## 📈 Performance Optimization

- **Nginx Load Balancing**: Efficient request handling
- **Gzip Compression**: Faster data transfer
- **Static Asset Caching**: Optimized file delivery
- **Database Connection Pooling**: Efficient database usage
- **Redis Caching**: Fast data retrieval

## 🐳 Docker Commands

### Basic Operations
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Update services
docker-compose pull && docker-compose up -d
```

### Service Management
```bash
# Backend logs
docker-compose logs -f backend

# Frontend logs
docker-compose logs -f frontend

# MySQL logs
docker-compose logs -f mysql

# Redis logs
docker-compose logs -f redis
```

### Database Operations
```bash
# Access MySQL
docker-compose exec mysql mysql -u nfl_user -p nfl_ai

# Backup database
docker-compose exec mysql mysqldump -u nfl_user -p nfl_ai > backup.sql

# Restore database
docker-compose exec -T mysql mysql -u nfl_user -p nfl_ai < backup.sql
```

## 🔧 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill processes using port 8000
   lsof -ti:8000 | xargs kill -9
   ```

2. **Docker Build Fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   docker-compose build --no-cache
   ```

3. **Database Connection Issues**
   ```bash
   # Check MySQL status
   docker-compose exec mysql mysqladmin ping
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # In Docker Desktop settings
   ```

### Health Checks

- **Backend**: http://localhost:8000/health
- **Frontend**: http://localhost:3000/
- **MySQL**: `docker-compose exec mysql mysqladmin ping`
- **Redis**: `docker-compose exec redis redis-cli ping`

## 📊 Monitoring

### Application Metrics
- **Response Time**: Monitor API response times
- **Error Rate**: Track failed requests
- **Throughput**: Requests per second
- **Resource Usage**: CPU, memory, disk usage

### Database Metrics
- **Connection Pool**: Active connections
- **Query Performance**: Slow query analysis
- **Storage Usage**: Database size monitoring

## 🚀 Production Deployment

### Pre-deployment Checklist
- [ ] Environment variables configured
- [ ] Database credentials secure
- [ ] SSL certificates installed
- [ ] Domain name configured
- [ ] Monitoring setup
- [ ] Backup strategy implemented

### Post-deployment Tasks
- [ ] Test all endpoints
- [ ] Verify data loading
- [ ] Check performance metrics
- [ ] Monitor error logs
- [ ] Set up alerts

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Check GitHub issues
4. Contact support team

## 🎉 Success!

Your NFL AI Platform is now live and ready to predict touchdowns for any player with state-of-the-art machine learning algorithms!

**Features:**
- ✅ Predict for ANY NFL player
- ✅ Advanced ML with 5+ algorithms
- ✅ Google-style AI text completion
- ✅ Real-time performance monitoring
- ✅ Beautiful responsive interface
- ✅ Fully containerized with Docker
- ✅ Production-ready security
- ✅ Scalable architecture
