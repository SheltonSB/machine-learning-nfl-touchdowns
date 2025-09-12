"""
NFL Database Management Module

This module handles all database operations for storing and managing NFL data.
Uses SQLite for simplicity and beginner-friendliness.

Author: Shelton Bumhe
"""

import logging
import os
import sqlite3
from typing import Dict

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLDatabase:
    """Database manager for NFL data storage and retrieval."""
    
    def __init__(self, db_path: str = "nfl_data.db"):
        """
        Initialize the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.create_tables()
    
    def connect(self):
        """Create database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_tables(self):
        """Create all necessary tables if they don't exist."""
        self.connect()
        
        # Basic player information table
        basic_stats_sql = """
        CREATE TABLE IF NOT EXISTS basic_stats (
            player_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            height REAL,
            weight REAL,
            experience REAL,
            position TEXT
        )
        """
        
        # Game logs table (general structure)
        game_logs_sql = """
        CREATE TABLE IF NOT EXISTS game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            name TEXT,
            year INTEGER,
            week INTEGER,
            season TEXT,
            team TEXT,
            opponent TEXT,
            position TEXT,
            game_date TEXT,
            home_away TEXT,
            result TEXT,
            FOREIGN KEY (player_id) REFERENCES basic_stats (player_id)
        )
        """
        
        # QB specific stats table
        qb_stats_sql = """
        CREATE TABLE IF NOT EXISTS qb_stats (
            game_log_id INTEGER PRIMARY KEY,
            passing_yards INTEGER,
            td_passes INTEGER,
            interceptions INTEGER,
            passes_attempted INTEGER,
            passes_completed INTEGER,
            completion_percentage REAL,
            yards_per_attempt REAL,
            passer_rating REAL,
            threw_td INTEGER,
            FOREIGN KEY (game_log_id) REFERENCES game_logs (id)
        )
        """
        
        # Career stats table
        career_stats_sql = """
        CREATE TABLE IF NOT EXISTS career_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            name TEXT,
            year INTEGER,
            team TEXT,
            position TEXT,
            games_played INTEGER,
            games_started INTEGER,
            FOREIGN KEY (player_id) REFERENCES basic_stats (player_id)
        )
        """
        
        # QB career passing stats
        qb_career_passing_sql = """
        CREATE TABLE IF NOT EXISTS qb_career_passing (
            career_id INTEGER PRIMARY KEY,
            passing_yards INTEGER,
            td_passes INTEGER,
            interceptions INTEGER,
            attempts INTEGER,
            completions INTEGER,
            completion_percentage REAL,
            yards_per_attempt REAL,
            passer_rating REAL,
            FOREIGN KEY (career_id) REFERENCES career_stats (id)
        )
        """
        
        # Model predictions table
        predictions_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            game_date TEXT,
            opponent TEXT,
            prediction INTEGER,
            confidence REAL,
            features_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES basic_stats (player_id)
        )
        """
        
        tables = [
            basic_stats_sql,
            game_logs_sql,
            qb_stats_sql,
            career_stats_sql,
            qb_career_passing_sql,
            predictions_sql
        ]
        
        for table_sql in tables:
            try:
                self.conn.execute(table_sql)
                logger.info("Table created successfully")
            except Exception as e:
                logger.error(f"Error creating table: {e}")
        
        self.conn.commit()
        self.disconnect()
    
    def load_csv_to_db(self, csv_path: str, table_name: str, chunk_size: int = 1000):
        """
        Load CSV data into database table.
        
        Args:
            csv_path (str): Path to CSV file
            table_name (str): Name of the table to load data into
            chunk_size (int): Number of rows to process at once
        """
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return
        
        self.connect()
        
        try:
            # Read CSV in chunks to handle large files
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                chunk.to_sql(table_name, self.conn, if_exists='append', index=False)
                logger.info(f"Loaded {len(chunk)} rows into {table_name}")
            
            logger.info(f"Successfully loaded {csv_path} into {table_name}")
            
        except Exception as e:
            logger.error(f"Error loading CSV to database: {e}")
        finally:
            self.disconnect()
    
    def get_qb_data_for_prediction(self, player_id: str, limit_games: int = 3) -> pd.DataFrame:
        """
        Get QB data for model prediction.
        
        Args:
            player_id (str): Player ID
            limit_games (int): Number of recent games to include
            
        Returns:
            pd.DataFrame: Processed data for prediction
        """
        self.connect()
        
        query = """
        SELECT 
            bs.player_id,
            bs.name,
            bs.age,
            bs.height,
            bs.weight,
            bs.experience,
            gl.year,
            gl.week,
            gl.team,
            gl.opponent,
            qs.passing_yards,
            qs.td_passes,
            qs.passes_attempted,
            qs.threw_td
        FROM basic_stats bs
        LEFT JOIN game_logs gl ON bs.player_id = gl.player_id
        LEFT JOIN qb_stats qs ON gl.id = qs.game_log_id
        WHERE bs.player_id = ? AND gl.position = 'QB'
        ORDER BY gl.year DESC, gl.week DESC
        LIMIT ?
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(player_id, limit_games))
            return df
        except Exception as e:
            logger.error(f"Error getting QB data: {e}")
            return pd.DataFrame()
        finally:
            self.disconnect()
    
    def save_prediction(self, player_id: str, game_date: str, opponent: str, 
                       prediction: int, confidence: float, features_used: str):
        """
        Save a model prediction to the database.
        
        Args:
            player_id (str): Player ID
            game_date (str): Date of the game
            opponent (str): Opponent team
            prediction (int): Model prediction (0 or 1)
            confidence (float): Prediction confidence
            features_used (str): JSON string of features used
        """
        self.connect()
        
        query = """
        INSERT INTO predictions (player_id, game_date, opponent, prediction, confidence, features_used)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        
        try:
            self.conn.execute(query, (player_id, game_date, opponent, prediction, confidence, features_used))
            self.conn.commit()
            logger.info(f"Prediction saved for player {player_id}")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
        finally:
            self.disconnect()
    
    def get_prediction_history(self, player_id: str = None) -> pd.DataFrame:
        """
        Get prediction history from database.
        
        Args:
            player_id (str): Optional player ID to filter by
            
        Returns:
            pd.DataFrame: Prediction history
        """
        self.connect()
        
        if player_id:
            query = """
            SELECT p.*, bs.name 
            FROM predictions p
            JOIN basic_stats bs ON p.player_id = bs.player_id
            WHERE p.player_id = ?
            ORDER BY p.created_at DESC
            """
            df = pd.read_sql_query(query, self.conn, params=(player_id,))
        else:
            query = """
            SELECT p.*, bs.name 
            FROM predictions p
            JOIN basic_stats bs ON p.player_id = bs.player_id
            ORDER BY p.created_at DESC
            """
            df = pd.read_sql_query(query, self.conn)
        
        self.disconnect()
        return df
    
    def get_table_info(self) -> Dict[str, int]:
        """
        Get information about all tables and their row counts.
        
        Returns:
            Dict[str, int]: Table names and their row counts
        """
        self.connect()
        
        tables = ['basic_stats', 'game_logs', 'qb_stats', 'career_stats', 
                 'qb_career_passing', 'predictions']
        
        info = {}
        for table in tables:
            try:
                count = pd.read_sql_query(f"SELECT COUNT(*) as count FROM {table}", self.conn)
                info[table] = count['count'].iloc[0]
            except Exception as e:
                logger.error(f"Error getting count for {table}: {e}")
                info[table] = 0
        
        self.disconnect()
        return info

# Global database instance used by modules that need quick access.
db = NFLDatabase()
