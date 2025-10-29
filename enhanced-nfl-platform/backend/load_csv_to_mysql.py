#!/usr/bin/env python3
"""
🏈 NFL AI Platform - CSV to MySQL Data Loader
Load all CSV files into MySQL database
"""

import pandas as pd
import pymysql
import os
import sys
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL Configuration
MYSQL_USERNAME = "root"
MYSQL_PASSWORD = "NewStrongPassword!123"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "nfl_ai"

def connect_to_mysql():
    """Connect to MySQL database"""
    try:
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USERNAME,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
            charset='utf8mb4'
        )
        logger.info("✅ Connected to MySQL database")
        return connection
    except Exception as e:
        logger.error(f"❌ Error connecting to MySQL: {e}")
        return None

def create_tables(connection):
    """Create all necessary tables for NFL data"""
    try:
        with connection.cursor() as cursor:
            # Basic Stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS basic_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    age INT,
                    height VARCHAR(10),
                    weight INT,
                    experience INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Career Stats - Passing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS career_stats_passing (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    games_played INT,
                    completions INT,
                    attempts INT,
                    completion_pct DECIMAL(5,2),
                    passing_yards INT,
                    passing_tds INT,
                    interceptions INT,
                    passer_rating DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Career Stats - Rushing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS career_stats_rushing (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    games_played INT,
                    rushing_attempts INT,
                    rushing_yards INT,
                    rushing_tds INT,
                    yards_per_attempt DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Career Stats - Receiving
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS career_stats_receiving (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    games_played INT,
                    receptions INT,
                    receiving_yards INT,
                    receiving_tds INT,
                    yards_per_reception DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Game Logs - Quarterback
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_logs_quarterback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    team VARCHAR(100),
                    opponent VARCHAR(100),
                    game_date DATE,
                    week INT,
                    season INT,
                    completions INT,
                    attempts INT,
                    completion_pct DECIMAL(5,2),
                    passing_yards INT,
                    passing_tds INT,
                    interceptions INT,
                    passer_rating DECIMAL(5,2),
                    rushing_attempts INT,
                    rushing_yards INT,
                    rushing_tds INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Game Logs - Running Back
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_logs_runningback (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    team VARCHAR(100),
                    opponent VARCHAR(100),
                    game_date DATE,
                    week INT,
                    season INT,
                    rushing_attempts INT,
                    rushing_yards INT,
                    rushing_tds INT,
                    receptions INT,
                    receiving_yards INT,
                    receiving_tds INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Game Logs - Wide Receiver
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_logs_wide_receiver (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    team VARCHAR(100),
                    opponent VARCHAR(100),
                    game_date DATE,
                    week INT,
                    season INT,
                    receptions INT,
                    receiving_yards INT,
                    receiving_tds INT,
                    targets INT,
                    catch_pct DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Career Stats - Defensive
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS career_stats_defensive (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    games_played INT,
                    tackles INT,
                    assists INT,
                    sacks DECIMAL(5,2),
                    interceptions INT,
                    passes_defended INT,
                    forced_fumbles INT,
                    fumble_recoveries INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Career Stats - Kickers
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS career_stats_kickers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    games_played INT,
                    field_goals_made INT,
                    field_goals_attempted INT,
                    field_goal_pct DECIMAL(5,2),
                    extra_points_made INT,
                    extra_points_attempted INT,
                    extra_point_pct DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Career Stats - Punters
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS career_stats_punters (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50),
                    name VARCHAR(255),
                    position VARCHAR(10),
                    team VARCHAR(100),
                    games_played INT,
                    punts INT,
                    punt_yards INT,
                    avg_punt_yards DECIMAL(5,2),
                    long_punt INT,
                    punts_inside_20 INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            logger.info("✅ All tables created successfully")
            
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        raise

def load_basic_stats(connection, csv_path):
    """Load basic stats CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Basic Stats: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO basic_stats (player_id, name, position, team, age, height, weight, experience)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    position = VALUES(position),
                    team = VALUES(team),
                    age = VALUES(age),
                    height = VALUES(height),
                    weight = VALUES(weight),
                    experience = VALUES(experience)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Position', '')),
                    str(row.get('Team', '')),
                    int(row.get('Age', 0)) if pd.notna(row.get('Age')) else None,
                    str(row.get('Height', '')),
                    int(row.get('Weight', 0)) if pd.notna(row.get('Weight')) else None,
                    int(row.get('Experience', 0)) if pd.notna(row.get('Experience')) else None
                ))
        
        connection.commit()
        logger.info("✅ Basic Stats loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Basic Stats: {e}")

def load_career_stats_passing(connection, csv_path):
    """Load career stats passing CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Career Stats Passing: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO career_stats_passing 
                    (player_id, name, position, team, games_played, completions, attempts, completion_pct, 
                     passing_yards, passing_tds, interceptions, passer_rating)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    position = VALUES(position),
                    team = VALUES(team),
                    games_played = VALUES(games_played),
                    completions = VALUES(completions),
                    attempts = VALUES(attempts),
                    completion_pct = VALUES(completion_pct),
                    passing_yards = VALUES(passing_yards),
                    passing_tds = VALUES(passing_tds),
                    interceptions = VALUES(interceptions),
                    passer_rating = VALUES(passer_rating)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Position', '')),
                    str(row.get('Team', '')),
                    int(row.get('Games Played', 0)) if pd.notna(row.get('Games Played')) else 0,
                    int(row.get('Completions', 0)) if pd.notna(row.get('Completions')) else 0,
                    int(row.get('Attempts', 0)) if pd.notna(row.get('Attempts')) else 0,
                    float(row.get('Completion %', 0)) if pd.notna(row.get('Completion %')) else 0.0,
                    int(row.get('Passing Yards', 0)) if pd.notna(row.get('Passing Yards')) else 0,
                    int(row.get('Passing TDs', 0)) if pd.notna(row.get('Passing TDs')) else 0,
                    int(row.get('Interceptions', 0)) if pd.notna(row.get('Interceptions')) else 0,
                    float(row.get('Passer Rating', 0)) if pd.notna(row.get('Passer Rating')) else 0.0
                ))
        
        connection.commit()
        logger.info("✅ Career Stats Passing loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Career Stats Passing: {e}")

def load_career_stats_rushing(connection, csv_path):
    """Load career stats rushing CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Career Stats Rushing: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO career_stats_rushing 
                    (player_id, name, position, team, games_played, rushing_attempts, 
                     rushing_yards, rushing_tds, yards_per_attempt)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    position = VALUES(position),
                    team = VALUES(team),
                    games_played = VALUES(games_played),
                    rushing_attempts = VALUES(rushing_attempts),
                    rushing_yards = VALUES(rushing_yards),
                    rushing_tds = VALUES(rushing_tds),
                    yards_per_attempt = VALUES(yards_per_attempt)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Position', '')),
                    str(row.get('Team', '')),
                    int(row.get('Games Played', 0)) if pd.notna(row.get('Games Played')) else 0,
                    int(row.get('Rushing Attempts', 0)) if pd.notna(row.get('Rushing Attempts')) else 0,
                    int(row.get('Rushing Yards', 0)) if pd.notna(row.get('Rushing Yards')) else 0,
                    int(row.get('Rushing TDs', 0)) if pd.notna(row.get('Rushing TDs')) else 0,
                    float(row.get('Yards Per Attempt', 0)) if pd.notna(row.get('Yards Per Attempt')) else 0.0
                ))
        
        connection.commit()
        logger.info("✅ Career Stats Rushing loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Career Stats Rushing: {e}")

def load_career_stats_receiving(connection, csv_path):
    """Load career stats receiving CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Career Stats Receiving: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO career_stats_receiving 
                    (player_id, name, position, team, games_played, receptions, 
                     receiving_yards, receiving_tds, yards_per_reception)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    name = VALUES(name),
                    position = VALUES(position),
                    team = VALUES(team),
                    games_played = VALUES(games_played),
                    receptions = VALUES(receptions),
                    receiving_yards = VALUES(receiving_yards),
                    receiving_tds = VALUES(receiving_tds),
                    yards_per_reception = VALUES(yards_per_reception)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Position', '')),
                    str(row.get('Team', '')),
                    int(row.get('Games Played', 0)) if pd.notna(row.get('Games Played')) else 0,
                    int(row.get('Receptions', 0)) if pd.notna(row.get('Receptions')) else 0,
                    int(row.get('Receiving Yards', 0)) if pd.notna(row.get('Receiving Yards')) else 0,
                    int(row.get('Receiving TDs', 0)) if pd.notna(row.get('Receiving TDs')) else 0,
                    float(row.get('Yards Per Reception', 0)) if pd.notna(row.get('Yards Per Reception')) else 0.0
                ))
        
        connection.commit()
        logger.info("✅ Career Stats Receiving loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Career Stats Receiving: {e}")

def load_game_logs_quarterback(connection, csv_path):
    """Load game logs quarterback CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Game Logs Quarterback: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                # Parse date
                game_date = None
                if pd.notna(row.get('Date')):
                    try:
                        game_date = pd.to_datetime(row.get('Date')).date()
                    except:
                        pass
                
                cursor.execute("""
                    INSERT INTO game_logs_quarterback 
                    (player_id, name, team, opponent, game_date, week, season, completions, 
                     attempts, completion_pct, passing_yards, passing_tds, interceptions, 
                     passer_rating, rushing_attempts, rushing_yards, rushing_tds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Team', '')),
                    str(row.get('Opponent', '')),
                    game_date,
                    int(row.get('Week', 0)) if pd.notna(row.get('Week')) else 0,
                    int(row.get('Season', 0)) if pd.notna(row.get('Season')) else 0,
                    int(row.get('Completions', 0)) if pd.notna(row.get('Completions')) else 0,
                    int(row.get('Attempts', 0)) if pd.notna(row.get('Attempts')) else 0,
                    float(row.get('Completion %', 0)) if pd.notna(row.get('Completion %')) else 0.0,
                    int(row.get('Passing Yards', 0)) if pd.notna(row.get('Passing Yards')) else 0,
                    int(row.get('Passing TDs', 0)) if pd.notna(row.get('Passing TDs')) else 0,
                    int(row.get('Interceptions', 0)) if pd.notna(row.get('Interceptions')) else 0,
                    float(row.get('Passer Rating', 0)) if pd.notna(row.get('Passer Rating')) else 0.0,
                    int(row.get('Rushing Attempts', 0)) if pd.notna(row.get('Rushing Attempts')) else 0,
                    int(row.get('Rushing Yards', 0)) if pd.notna(row.get('Rushing Yards')) else 0,
                    int(row.get('Rushing TDs', 0)) if pd.notna(row.get('Rushing TDs')) else 0
                ))
        
        connection.commit()
        logger.info("✅ Game Logs Quarterback loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Game Logs Quarterback: {e}")

def load_game_logs_runningback(connection, csv_path):
    """Load game logs running back CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Game Logs Running Back: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                # Parse date
                game_date = None
                if pd.notna(row.get('Date')):
                    try:
                        game_date = pd.to_datetime(row.get('Date')).date()
                    except:
                        pass
                
                cursor.execute("""
                    INSERT INTO game_logs_runningback 
                    (player_id, name, team, opponent, game_date, week, season, rushing_attempts, 
                     rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Team', '')),
                    str(row.get('Opponent', '')),
                    game_date,
                    int(row.get('Week', 0)) if pd.notna(row.get('Week')) else 0,
                    int(row.get('Season', 0)) if pd.notna(row.get('Season')) else 0,
                    int(row.get('Rushing Attempts', 0)) if pd.notna(row.get('Rushing Attempts')) else 0,
                    int(row.get('Rushing Yards', 0)) if pd.notna(row.get('Rushing Yards')) else 0,
                    int(row.get('Rushing TDs', 0)) if pd.notna(row.get('Rushing TDs')) else 0,
                    int(row.get('Receptions', 0)) if pd.notna(row.get('Receptions')) else 0,
                    int(row.get('Receiving Yards', 0)) if pd.notna(row.get('Receiving Yards')) else 0,
                    int(row.get('Receiving TDs', 0)) if pd.notna(row.get('Receiving TDs')) else 0
                ))
        
        connection.commit()
        logger.info("✅ Game Logs Running Back loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Game Logs Running Back: {e}")

def load_game_logs_wide_receiver(connection, csv_path):
    """Load game logs wide receiver CSV"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Game Logs Wide Receiver: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                # Parse date
                game_date = None
                if pd.notna(row.get('Date')):
                    try:
                        game_date = pd.to_datetime(row.get('Date')).date()
                    except:
                        pass
                
                cursor.execute("""
                    INSERT INTO game_logs_wide_receiver 
                    (player_id, name, team, opponent, game_date, week, season, receptions, 
                     receiving_yards, receiving_tds, targets, catch_pct)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(row.get('Player ID', '')),
                    str(row.get('Name', '')),
                    str(row.get('Team', '')),
                    str(row.get('Opponent', '')),
                    game_date,
                    int(row.get('Week', 0)) if pd.notna(row.get('Week')) else 0,
                    int(row.get('Season', 0)) if pd.notna(row.get('Season')) else 0,
                    int(row.get('Receptions', 0)) if pd.notna(row.get('Receptions')) else 0,
                    int(row.get('Receiving Yards', 0)) if pd.notna(row.get('Receiving Yards')) else 0,
                    int(row.get('Receiving TDs', 0)) if pd.notna(row.get('Receiving TDs')) else 0,
                    int(row.get('Targets', 0)) if pd.notna(row.get('Targets')) else 0,
                    float(row.get('Catch %', 0)) if pd.notna(row.get('Catch %')) else 0.0
                ))
        
        connection.commit()
        logger.info("✅ Game Logs Wide Receiver loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Game Logs Wide Receiver: {e}")

def main():
    """Main function to load all CSV data into MySQL"""
    print("🏈 NFL AI Platform - CSV to MySQL Data Loader")
    print("=" * 50)
    
    # Connect to MySQL
    connection = connect_to_mysql()
    if not connection:
        print("❌ Cannot connect to MySQL. Exiting.")
        sys.exit(1)
    
    try:
        # Create tables
        print("📋 Creating database tables...")
        create_tables(connection)
        
        # Load CSV files
        csv_files = {
            "Basic_Stats.csv": load_basic_stats,
            "Career_Stats_Passing.csv": load_career_stats_passing,
            "Career_Stats_Rushing.csv": load_career_stats_rushing,
            "Career_Stats_Receiving.csv": load_career_stats_receiving,
            "Game_Logs_Quarterback.csv": load_game_logs_quarterback,
            "Game_Logs_Runningback.csv": load_game_logs_runningback,
            "Game_Logs_Wide_Receiver_and_Tight_End.csv": load_game_logs_wide_receiver,
        }
        
        base_path = "../../data/raw/"
        
        for csv_file, loader_func in csv_files.items():
            csv_path = base_path + csv_file
            if os.path.exists(csv_path):
                print(f"\n📊 Loading {csv_file}...")
                loader_func(connection, csv_path)
            else:
                print(f"⚠️  File not found: {csv_path}")
        
        # Show summary
        print("\n📊 Database Summary:")
        with connection.cursor() as cursor:
            tables = [
                "basic_stats", "career_stats_passing", "career_stats_rushing", 
                "career_stats_receiving", "game_logs_quarterback", 
                "game_logs_runningback", "game_logs_wide_receiver"
            ]
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"   {table}: {count:,} records")
                except:
                    print(f"   {table}: Table not found")
        
        print("\n🎉 All CSV data loaded successfully into MySQL database!")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    main()
