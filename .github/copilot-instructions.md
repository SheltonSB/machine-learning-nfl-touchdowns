<!-- Copilot / AI agent instructions for `machine-learning-nfl-touchdowns` -->

# Quick Orientation

This repository trains and serves a TensorFlow model that predicts whether an NFL quarterback will throw a touchdown in a game. The canonical orchestration is `python main.py` (see examples below). Key directories are `src/` (project code), `data/` (raw + processed CSVs), `models/` (trained artifacts), and `enhanced-nfl-platform/` (deployment + production-ready services).

# Important Files & Entry Points

- `main.py` — primary CLI/orchestrator. Use flags like `--setup`, `--preprocess`, `--train-model`, `--workflow`, and `--generate-shap` to run tasks programmatically. The code adds `src` to `sys.path` and instantiates `NFLProjectOrchestrator`.
- `src/data_loader.py` — CSV→DB loader used by `--setup`.
- `src/data_validator.py` — data quality checks invoked by `--validate`.
- `src/preprocess.py` — creates `data/processed/final_dataset.csv` used for training.
- `src/train_model.py` — training script invoked from `main.py` (returns exit codes).
- `src/explain_shap.py` — generates SHAP plots after training.
- `models/qb_td_model.keras` — expected TensorFlow SavedModel artifact name and location. A legacy `models/qb_td_model.pkl` may exist; prefer regenerating the TF model.
- `data/processed/final_dataset.csv` — canonical processed dataset used for training and reported in status commands.
- `enhanced-nfl-platform/backend/app/core/config.py` — deployment config defaults and env var names.

# How This Repo Is Typically Used (Commands)

Examples you can execute or call programmatically:

```
# Install
pip install -r requirements.txt

# Full workflow (load → validate → preprocess → train)
python main.py --workflow --train-model

# Preprocess only
python main.py --preprocess

# Train only (assumes processed CSV present)
python main.py --train-model

# Force retrain and generate SHAP
python main.py --train-model --force-train --generate-shap

# Generate SHAP when a model already exists
python main.py --generate-shap

# Check project status (DB counts, model, processed dataset)
python main.py --status
```

For containerized/deployment testing use `enhanced-nfl-platform/docker-compose.yml` (see `README.md` for full steps).

# Project Conventions & Patterns (for code changes)

- Single orchestrator pattern: `NFLProjectOrchestrator` performs database setup, validation, preprocessing, training, and SHAP generation. Prefer interacting via `main.py` rather than calling lower-level modules directly unless writing tests.
- File-based artifacts and strict paths: code expects `data/raw/`, `data/processed/final_dataset.csv`, and `models/qb_td_model.keras`. Do not rename these artifacts without updating `main.py` and referenced modules.
- Legacy vs canonical models: if `models/qb_td_model.pkl` exists, the orchestrator logs a warning. Prefer retraining to create the TF SavedModel.
- DB usage: a local SQLite DB is used (see `src/database.py`). Agents should avoid destructive DB schema changes unless tests are updated accordingly.
- Exit codes: `train_model.main()` and `explain_shap.main()` return exit codes; `main.py` checks these to determine success/failure. Keep return code semantics intact.

# Tests, Linting & CI

- Local tests: use `make backend-test` / `make frontend-test` mentioned in `README.md`. There is a `pytest.ini` in `enhanced-nfl-platform/backend/` — run backend tests from that directory when appropriate.
- Formatting/linting: project suggests `black` and `ruff`. Keep style consistent with existing files.

# Integration Points & Environment

- SQLite database file: `nfl_data.db` (ensure no concurrent locks during runs).
- Environment variables used for deployment: `DATABASE_URL`, `MODEL_PATH`, `EMBEDDING_MODEL` (defaults referenced in `enhanced-nfl-platform/backend/app/core/config.py`). When running locally, the CLI uses file paths in the repo root.
- External services: Optional FastAPI service in `enhanced-nfl-platform/backend/` and frontend in `enhanced-nfl-platform/frontend/` (Docker Compose orchestrates them).

# Guidance for an AI Coding Agent

- When making changes, prefer small, focused edits that preserve CLI behavior and artifact paths.
- For feature work that impacts training or data formats, update `README.md` and the `models/training_metrics.json` consumer expectations.
- If adding new env/config keys, add defaults in `enhanced-nfl-platform/backend/app/core/config.py` and document them in `README.md`.
- Use existing helper classes (`NFLDatabase`, `NFLDataLoader`, `NFLPreprocessor`) to avoid duplicating logic.
- When modifying training behavior, ensure `models/qb_td_model.keras` is still produced and that `train_model.main()` returns conventional exit codes (0 / os.EX_OK on success).

# Quick Contacts & Next Steps

If anything in this document is unclear or you want more examples (unit tests, example DB queries, or a proposed CI job), tell me which area to expand and I will iterate.
