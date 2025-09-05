# 🏈 NFL QB Touchdown Predictor

A **database-driven** machine learning project that predicts whether a quarterback (QB) will throw at least one touchdown (TD) in a given NFL game using player statistics, game context, and historical performance data.

---

## 🎯 Project Objective

> Predict whether an NFL quarterback will throw a **touchdown pass** in a game using past performance and game details.

This binary classification model uses **historical QB game logs**, **player career stats**, and **basic bio data** stored in a SQLite database, and demonstrates feature engineering, model evaluation, and explainability techniques.

---

## ✨ Key Features

- **🗄️ Database-driven**: All data stored in SQLite for easy management and validation
- **🎯 Real-time predictions**: Make predictions using current player data from database
- **📊 Data validation**: Comprehensive data quality checks and validation
- **🔄 Automated workflow**: One-command setup and deployment
- **📈 Historical tracking**: View prediction history and accuracy
- **🌐 Modern web app**: Beautiful Streamlit interface with multiple pages

---

## 🏗️ Project Structure

```
machine-learning-nfl-touchdowns/
├── 📁 data/
│   ├── 📁 raw/          # Original CSV files
│   └── 📁 processed/    # Cleaned & engineered datasets
├── 📁 src/
│   ├── 🗄️ database.py      # Database management
│   ├── 📥 data_loader.py   # Load CSV data into database
│   ├── ✅ data_validator.py # Data quality validation
│   ├── 🔄 preprocess.py    # Database-driven preprocessing
│   ├── 🎯 train_model.py   # Model training
│   └── 📊 explain_shap.py  # Model explainability
├── 📁 app/
│   └── 🌐 app.py           # Streamlit web application
├── 📁 models/
│   └── 🤖 qb_td_model.pkl  # Trained model file
├── 📁 notebooks/
│   └── 📓 eda.ipynb        # Exploratory data analysis
├── 🚀 main.py              # Main orchestration script
├── 📋 requirements.txt     # Python dependencies
└── 📖 README.md           # This file
```

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Database** | SQLite | Data storage and management |
| **Data Processing** | pandas, numpy | Data manipulation and analysis |
| **Machine Learning** | scikit-learn, xgboost | Model training and prediction |
| **Web App** | Streamlit | User interface |
| **Validation** | Custom validation framework | Data quality assurance |
| **Orchestration** | Python scripts | Workflow automation |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Workflow
```bash
python main.py
```

This will:
- ✅ Load all CSV data into the database
- ✅ Validate data quality
- ✅ Preprocess data for modeling
- ✅ Launch the Streamlit web app

### 3. Alternative Commands

```bash
# Just set up the database
python main.py --setup

# Validate data quality
python main.py --validate

# Preprocess data only
python main.py --preprocess

# Launch app only
python main.py --app

# Check project status
python main.py --status

# Force reload data
python main.py --workflow --force-reload
```

---

## 📊 Database Schema

The project uses a relational SQLite database with the following tables:

### Core Tables
- **`basic_stats`**: Player demographics and physical info
- **`game_logs`**: Game-by-game performance records
- **`qb_stats`**: QB-specific game statistics
- **`career_stats`**: Season-level career statistics
- **`qb_career_passing`**: QB career passing stats
- **`predictions`**: Model prediction history

### Key Relationships
- Players linked by `player_id`
- Game logs linked to QB stats by `game_log_id`
- Career stats linked to QB passing stats by `career_id`

---

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 88% |
| **F1 Score** | 85% |
| **ROC-AUC** | 91% |

---

## 🌐 Web Application

The Streamlit app provides multiple pages:

### 🎯 Make Prediction
- Select quarterback from database
- View recent performance stats
- Make real-time predictions
- Save predictions to database

### 🗄️ Player Database
- Browse all available data
- View database statistics
- Explore sample records

### 📊 Prediction History
- Track all predictions made
- View prediction accuracy
- Analyze confidence scores

### ℹ️ About
- Project information
- Technical details
- Usage instructions

---

## 🔧 Advanced Usage

### Manual Data Loading
```python
from src.data_loader import NFLDataLoader

loader = NFLDataLoader()
loader.load_all_data()
```

### Data Validation
```python
from src.data_validator import NFLDataValidator

validator = NFLDataValidator()
results = validator.validate_all_data()
```

### Database Queries
```python
from src.database import NFLDatabase

db = NFLDatabase()
db.connect()

# Get QB data for prediction
qb_data = db.get_qb_data_for_prediction("player_id_123")

# Save prediction
db.save_prediction(
    player_id="player_id_123",
    game_date="2024-01-15",
    opponent="KC",
    prediction=1,
    confidence=0.85,
    features_used='{"age": 28, "passing_yards": 275}'
)
```

---

## 📈 Data Sources

- **Game Logs**: NFL.com game-by-game statistics
- **Career Stats**: Season-level performance data
- **Basic Stats**: Player demographics and physical attributes

### Files Used
- `Game_Logs_Quarterback.csv`
- `Career_Stats_Passing.csv`
- `Basic_Stats.csv`

---

## 🔍 Data Validation

The project includes comprehensive data validation:

- ✅ **Data completeness**: Check for missing values
- ✅ **Data consistency**: Verify relationships between tables
- ✅ **Value ranges**: Ensure statistics are reasonable
- ✅ **Duplicate detection**: Find and handle duplicates
- ✅ **Date validation**: Check for valid game dates
- ✅ **QB-specific checks**: Validate quarterback data quality

---

## 🛠️ Development

### Adding New Data Sources
1. Add CSV file to `data/raw/`
2. Update `data_loader.py` to handle new file
3. Add validation rules in `data_validator.py`
4. Update preprocessing in `preprocess.py`

### Extending the Model
1. Modify feature engineering in `preprocess.py`
2. Update model training in `train_model.py`
3. Adjust prediction logic in `app/app.py`

### Database Schema Changes
1. Update table creation in `database.py`
2. Modify data loading logic
3. Update validation queries
4. Test with existing data

---

## 🐛 Troubleshooting

### Common Issues

**Database not found**
```bash
# Recreate database
python main.py --setup --force-reload
```

**Model file missing**
```bash
# Train model (if you have the training script)
python src/train_model.py
```

**Streamlit not working**
```bash
# Install streamlit
pip install streamlit

# Launch manually
streamlit run app/app.py
```

**Data validation errors**
```bash
# Check data quality
python main.py --validate

# Review validation output for specific issues
```

---

## 📝 License

This project is open-source under the MIT License.

---

## � Author

**Shelton Bumhe**  
*Data Scientist | Software Engneer  | AI | NFL Fan*

- 📬 [LinkedIn](https://linkedin.com/in/sheltonbumhe)
- 🌐 [Portfolio](https://sheltonbumhe.com)
- 📧 [Email](mailto:shelton@example.com)

---

## Acknowledgments

- NFL.com for providing the statistical data
- Streamlit for the excellent web app framework
- The open-source community for the amazing tools

---

## 📊 Project Statistics

- **Total Records**: 100,000+ game logs
- **Quarterbacks**: 500+ players
- **Years Covered**: 2000-2024
- **Features**: 15+ engineered features
- **Predictions**: Real-time with confidence scores

---

*Ready to predict some touchdowns? 🏈 Let's go!*


