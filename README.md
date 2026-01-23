# NFL QB Touchdown Predictor

A database-driven machine learning project that predicts whether a quarterback will throw at least one touchdown in an NFL game. The system combines player statistics, game context, and historical performance data to deliver transparent predictions.

---

## Project Objective

Predict whether an NFL quarterback will throw a touchdown pass in a game using past performance, player profile data, and game context.

---

## Key Features

- Database-backed storage using SQLite
- Real-time predictions powered by a trained model
- Historical prediction tracking with confidence scoring
- Command-line workflow for data ingestion and modeling

---

## Project Structure

```
machine-learning-nfl-touchdowns/
|-- data/
|   |-- raw/              # Original CSV file
|   `-- processed/        # Cleaned and engineered datasets
|-- src/
|   |-- database.py       # Database management
|   |-- data_loader.py    # Load CSV data into the database
|   |-- data_validator.py # Data quality validation
|   |-- preprocess.py     # Database-driven preprocessing
|   |-- train_model.py    # Model training
|   `-- explain_shap.py   # Model explainability
|-- models/
|   |-- qb_td_model.keras       # TensorFlow SavedModel artifact
|   |-- feature_scaler.pkl      # StandardScaler used during training
|   `-- training_metrics.json   # Cross-validation & evaluation metrics
|-- notebooks/
|   `-- eda.ipynb         # Exploratory data analysis
|-- main.py               # Main orchestration script
|-- requirements.txt      # Python dependencies
`-- README.md             # Project documentation
```

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Database | SQLite | Data storage and management |
| Data Processing | pandas, numpy | Data manipulation and analysis |
| Machine Learning | scikit-learn, TensorFlow (Keras) | Model training and prediction |
| API & Orchestration | FastAPI, Uvicorn | Optional service layer |
| Validation | Custom validation framework | Data quality assurance |
| Orchestration | Python scripts | Workflow automation |

---

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Complete Workflow

```bash
python main.py --workflow --train-model
```

This command loads data into the database, validates data quality, preprocesses features, and trains the TensorFlow touchdown model. Append `--generate-shap` to export a SHAP summary plot.

Run reproducible tasks with `make`: `make backend-test`, `make frontend-test`, `make seed`, and `make docker-up`.

### Deployment Checklist

1. Refresh data assets
   - `python main.py --workflow --train-model --generate-shap`
   - Confirm artifacts exist in `models/qb_td_model.keras`, `models/feature_scaler.pkl`, and `models/training_metrics.json`.
2. Run automated tests
   - `make backend-test`
   - `make frontend-test`
3. Provision environment variables (see `enhanced-nfl-platform/backend/app/core/config.py` for defaults).
   - Set `DATABASE_URL`, `MODEL_PATH` (default `/app/models`), `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`, and secrets in your deployment target.
4. Launch the stack with Docker Compose
   - `cd enhanced-nfl-platform`
   - `docker-compose up --build`
5. Verify services
   - API health: `http://localhost:8000/health`
   - Frontend: `http://localhost:3000`
   - Review logs: `docker-compose logs -f`

### Useful Commands

```bash
# Set up the database
python main.py --setup

# Validate data quality
python main.py --validate

# Preprocess data only
python main.py --preprocess

# Train or retrain the TensorFlow model (requires processed data)
python main.py --train-model

# Force retraining even if a model already exists
python main.py --train-model --force-train

# Generate a SHAP summary (requires trained model and processed data)
python main.py --generate-shap

# Launch the app only
python main.py --app

# Check project status
python main.py --status

# Force a full workflow reload
python main.py --workflow --force-reload
```

---

## Database Schema

Core tables:

- `basic_stats`: Player demographics and physical information
- `game_logs`: Game-by-game performance records
- `qb_stats`: Quarterback-specific game statistics
- `career_stats`: Season-level career statistics
- `qb_career_passing`: Career passing statistics
- `predictions`: Model prediction history

Key relationships:

- Players linked by `player_id`
- Game logs linked to quarterback stats by `game_log_id`
- Career stats linked to passing stats by `career_id`

---

## Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 88% |
| F1 Score | 85% |
| ROC-AUC | 91% |

Run `python main.py --train-model` to refresh metrics; the detailed cross-validation and test results are persisted in `models/training_metrics.json` after each training session.

---

## Advanced Usage

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

### Model Explainability

```bash
# Generate a SHAP beeswarm plot after training the TensorFlow model
python src/explain_shap.py
```

The script loads the SavedModel (`models/qb_td_model.keras`), applies the stored scaler, and
writes a summary plot to `models/shap_summary.png` highlighting the strongest drivers of a
touchdown prediction.

### Database Queries

```python
from src.database import NFLDatabase

db = NFLDatabase()
db.connect()

# Get quarterback data for prediction
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

## Data Validation

The `NFLDataValidator` module includes checks for:

- Data completeness and missing values
- Consistency across related tables
- Reasonable value ranges for statistics
- Duplicate detection and cleanup
- Valid game dates
- Quarterback-specific data quality rules

---

## Development

- Python 3.8 or newer
- Use a virtual environment (`python -m venv .venv`) for isolation
- Format code with `black` or `ruff format`
- Run linting with `ruff` where available

---

## Troubleshooting

- Ensure the SQLite database file `nfl_data.db` is accessible and not locked
- Verify that required CSV files are present in `data/raw/`
- Re-run `python main.py --workflow --force-reload` after major data changes

---

## License

This project is released under the MIT License. See the `LICENSE` file for details.

---

## Author

- [LinkedIn](https://linkedin.com/in/sheltonbumhe)
- [Portfolio](https://sheltonbumhe.com)
- [Email](mailto:sbumhe2@huskers.unl.edu)

---

## Project Statistics

The project includes scripts for data ingestion, validation, model training, interpretability, and interactive exploration. Extend the workflow by adding new data sources, refining feature engineering, or experimenting with alternative models.
