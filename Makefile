PYTHON=python3
FRONTEND_DIR=enhanced-nfl-platform/frontend
BACKEND_DIR=enhanced-nfl-platform/backend

.PHONY: install backend-test frontend-test frontend-storybook seed docker-up audit

install:
	pip install -r requirements.txt && pip install -r $(BACKEND_DIR)/requirements.txt
	cd $(FRONTEND_DIR) && npm install

backend-test:
	cd $(BACKEND_DIR) && TEST_MODE=true SKIP_ML_IMPORTS=1 SKIP_RAG_IMPORTS=1 DATABASE_URL=sqlite:///:memory: REDIS_URL=redis://localhost:6379/0 pytest --cov=app --cov-report=term --cov-report=xml --cov-fail-under=80 tests

frontend-test:
	cd $(FRONTEND_DIR) && npm run test:ci

frontend-storybook:
	cd $(FRONTEND_DIR) && npm run storybook

seed:
	$(PYTHON) scripts/seed_data.py

docker-up:
	docker-compose up --build

docker-up-seeded:
	docker-compose up --build -d
	$(MAKE) seed

audit:
	pip-audit --requirement $(BACKEND_DIR)/requirements.txt
	cd $(FRONTEND_DIR) && npm audit --audit-level=moderate || true
