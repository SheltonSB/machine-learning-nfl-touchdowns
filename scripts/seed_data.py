#!/usr/bin/env python3
"""Seed the local SQLite database with sample data."""

import logging
from pathlib import Path

from src.data_loader import NFLDataLoader
from src.database import NFLDatabase

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / 'data' / 'raw'
    if not raw_dir.exists():
        raise SystemExit(f'Expected raw data directory at {raw_dir}')

    LOGGER.info('Seeding database from %s', raw_dir)
    db = NFLDatabase(project_root / 'nfl_data.db')
    loader = NFLDataLoader(raw_data_path=str(raw_dir), db=db)

    loader.load_basic_stats()
    loader.load_game_logs()
    LOGGER.info('Seed complete.')


if __name__ == '__main__':
    main()
