"""
NFL Data Loader

This script loads all CSV files from the data/raw/ directory into the SQLite database
with proper data transformation and cleaning.

Author: Shelton Bumhe
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
from database import NFLDatabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLDataLoader:
    """Handles loading and transforming NFL data into the database."""
    
    def __init__(self, raw_data_path: str = "../data/raw/", db: NFLDatabase = None):
        """
        Initialize the data loader.
        
        Args:
            raw_data_path (str): Path to raw CSV files
            db (NFLDatabase): Database instance
        """
        self.raw_data_path = Path(raw_data_path)
        self.db = db or NFLDatabase()
        
    def load_basic_stats(self):
        """Load and transform basic player statistics."""
        logger.info("Loading basic stats...")
        
        csv_path = self.raw_data_path / "Basic_Stats.csv"
        if not csv_path.exists():
            logger.error(f"Basic_Stats.csv not found at {csv_path}")
            return
        
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        # Clean and transform the data
        df_clean = self._clean_basic_stats(df)
        
        # Save to database
        self.db.connect()
        df_clean.to_sql('basic_stats', self.db.conn, if_exists='replace', index=False)
        self.db.conn.commit()
        self.db.disconnect()
        
        logger.info(f"Loaded {len(df_clean)} basic stats records")
    
    def _clean_basic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean basic stats data."""
        # Keep only essential columns
        keep_cols = ['Player Id', 'Name', 'Age', 'Height', 'Weight', 'Experience', 'Position']
        df = df[keep_cols].copy()
        
        # Clean up data types
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Height'] = pd.to_numeric(df['Height'], errors='coerce')
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        
        # Extract numeric experience
        df['Experience'] = df['Experience'].str.extract(r'(\d+)').astype(float)
        
        # Clean up position
        df['Position'] = df['Position'].fillna('Unknown')
        
        return df.dropna(subset=['Player Id'])
    
    def load_game_logs(self):
        """Load and transform game logs data."""
        logger.info("Loading game logs...")
        
        # Load QB game logs first
        qb_csv_path = self.raw_data_path / "Game_Logs_Quarterback.csv"
        if qb_csv_path.exists():
            self._load_qb_game_logs(qb_csv_path)
        
        # Load other position game logs
        game_log_files = [
            "Game_Logs_Runningback.csv",
            "Game_Logs_Wide_Receiver_and_Tight_End.csv",
            "Game_Logs_Kickers.csv",
            "Game_Logs_Punters.csv"
        ]
        
        for file_name in game_log_files:
            csv_path = self.raw_data_path / file_name
            if csv_path.exists():
                self._load_general_game_logs(csv_path)
    
    def _load_qb_game_logs(self, csv_path: Path):
        """Load quarterback-specific game logs."""
        logger.info(f"Loading QB game logs from {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Clean the data
        df_clean = self._clean_game_logs(df)
        
        # Split into general game logs and QB-specific stats
        game_logs = df_clean[['Player Id', 'Name', 'Year', 'Week', 'Season', 'Team', 
                             'Opponent', 'Position', 'Game Date', 'Home/Away', 'Result']].copy()
        
        qb_stats = df_clean[['Passing Yards', 'TD Passes', 'Interceptions', 
                            'Passes Attempted', 'Passes Completed', 'Completion Percentage',
                            'Yards per Attempt', 'Passer Rating']].copy()
        
        # Add binary TD target
        qb_stats['threw_td'] = (df_clean['TD Passes'] > 0).astype(int)
        
        # Save to database
        self.db.connect()
        
        # Save general game logs first
        game_logs.to_sql('game_logs', self.db.conn, if_exists='append', index=False)
        
        # Get the IDs of the inserted game logs
        game_log_ids = pd.read_sql_query(
            "SELECT id FROM game_logs ORDER BY id DESC LIMIT ?", 
            self.db.conn, params=(len(game_logs),)
        )
        
        # Add game log IDs to QB stats
        qb_stats['game_log_id'] = game_log_ids['id'].values
        
        # Save QB stats
        qb_stats.to_sql('qb_stats', self.db.conn, if_exists='append', index=False)
        
        self.db.conn.commit()
        self.db.disconnect()
        
        logger.info(f"Loaded {len(game_logs)} QB game log records")
    
    def _load_general_game_logs(self, csv_path: Path):
        """Load general game logs for other positions."""
        logger.info(f"Loading general game logs from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df_clean = self._clean_game_logs(df)
        
        # Keep only general columns
        general_cols = ['Player Id', 'Name', 'Year', 'Week', 'Season', 'Team', 
                       'Opponent', 'Position', 'Game Date', 'Home/Away', 'Result']
        df_clean = df_clean[general_cols].copy()
        
        # Save to database
        self.db.connect()
        df_clean.to_sql('game_logs', self.db.conn, if_exists='append', index=False)
        self.db.conn.commit()
        self.db.disconnect()
        
        logger.info(f"Loaded {len(df_clean)} general game log records")
    
    def _clean_game_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean game logs data."""
        # Remove preseason games
        df = df[df['Season'] == 'Regular Season'].copy()
        
        # Replace '--' with NaN
        df = df.replace('--', np.nan)
        
        # Convert numeric columns
        numeric_cols = ['Year', 'Week', 'Passing Yards', 'TD Passes', 'Interceptions',
                       'Passes Attempted', 'Passes Completed', 'Completion Percentage',
                       'Yards per Attempt', 'Passer Rating']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['Player Id', 'Name'])
    
    def load_career_stats(self):
        """Load and transform career statistics."""
        logger.info("Loading career stats...")
        
        # Load QB career passing stats
        qb_career_path = self.raw_data_path / "Career_Stats_Passing.csv"
        if qb_career_path.exists():
            self._load_qb_career_stats(qb_career_path)
        
        # Load other career stats
        career_files = [
            "Career_Stats_Rushing.csv",
            "Career_Stats_Receiving.csv",
            "Career_Stats_Defensive.csv"
        ]
        
        for file_name in career_files:
            csv_path = self.raw_data_path / file_name
            if csv_path.exists():
                self._load_general_career_stats(csv_path)
    
    def _load_qb_career_stats(self, csv_path: Path):
        """Load quarterback career passing statistics."""
        logger.info(f"Loading QB career stats from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df_clean = self._clean_career_stats(df)
        
        # Split into general career stats and QB-specific stats
        career_stats = df_clean[['Player Id', 'Name', 'Year', 'Team', 'Position']].copy()
        career_stats['games_played'] = 1  # Default value
        career_stats['games_started'] = 1  # Default value
        
        qb_career_passing = df_clean[['Passing Yards', 'TD Passes', 'Interceptions',
                                    'Attempts', 'Completions', 'Completion Percentage',
                                    'Yards per Attempt', 'Passer Rating']].copy()
        
        # Save to database
        self.db.connect()
        
        # Save general career stats first
        career_stats.to_sql('career_stats', self.db.conn, if_exists='append', index=False)
        
        # Get the IDs of the inserted career stats
        career_ids = pd.read_sql_query(
            "SELECT id FROM career_stats ORDER BY id DESC LIMIT ?", 
            self.db.conn, params=(len(career_stats),)
        )
        
        # Add career IDs to QB career passing stats
        qb_career_passing['career_id'] = career_ids['id'].values
        
        # Save QB career passing stats
        qb_career_passing.to_sql('qb_career_passing', self.db.conn, if_exists='append', index=False)
        
        self.db.conn.commit()
        self.db.disconnect()
        
        logger.info(f"Loaded {len(career_stats)} QB career stat records")
    
    def _load_general_career_stats(self, csv_path: Path):
        """Load general career stats for other positions."""
        logger.info(f"Loading general career stats from {csv_path}")
        
        df = pd.read_csv(csv_path)
        df_clean = self._clean_career_stats(df)
        
        # Keep only general columns
        general_cols = ['Player Id', 'Name', 'Year', 'Team', 'Position']
        df_clean = df_clean[general_cols].copy()
        df_clean['games_played'] = 1  # Default value
        df_clean['games_started'] = 1  # Default value
        
        # Save to database
        self.db.connect()
        df_clean.to_sql('career_stats', self.db.conn, if_exists='append', index=False)
        self.db.conn.commit()
        self.db.disconnect()
        
        logger.info(f"Loaded {len(df_clean)} general career stat records")
    
    def _clean_career_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean career stats data."""
        # Replace '--' with NaN
        df = df.replace('--', np.nan)
        
        # Convert numeric columns
        numeric_cols = ['Year', 'Passing Yards', 'TD Passes', 'Interceptions',
                       'Attempts', 'Completions', 'Completion Percentage',
                       'Yards per Attempt', 'Passer Rating']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df.dropna(subset=['Player Id', 'Name'])
    
    def load_all_data(self):
        """Load all data into the database."""
        logger.info("Starting data loading process...")
        
        try:
            # Load data in order
            self.load_basic_stats()
            self.load_game_logs()
            self.load_career_stats()
            
            # Print database summary
            self._print_database_summary()
            
            logger.info("Data loading completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during data loading: {e}")
            raise
    
    def _print_database_summary(self):
        """Print a summary of the loaded data."""
        table_info = self.db.get_table_info()
        
        logger.info("\n" + "="*50)
        logger.info("DATABASE SUMMARY")
        logger.info("="*50)
        
        for table, count in table_info.items():
            logger.info(f"{table:20}: {count:,} records")
        
        logger.info("="*50)

def main():
    """Main function to run the data loader."""
    loader = NFLDataLoader()
    loader.load_all_data()

if __name__ == "__main__":
    main() 