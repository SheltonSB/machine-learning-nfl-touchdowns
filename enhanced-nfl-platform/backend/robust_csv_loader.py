#!/usr/bin/env python3
"""
🏈 NFL AI Platform - Robust CSV to MySQL Data Loader
Handle data cleaning and load all CSV files into MySQL database
"""

import pandas as pd
import pymysql
import os
import sys
import re
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

def clean_numeric(value):
    """Clean numeric values from CSV"""
    if pd.isna(value) or value == '' or value == '--' or value == '-':
        return 0
    
    # Convert to string and clean
    value_str = str(value).strip()
    
    # Remove non-numeric characters except decimal point and minus
    value_str = re.sub(r'[^\d.-]', '', value_str)
    
    # Handle empty string
    if value_str == '' or value_str == '-':
        return 0
    
    try:
        return float(value_str)
    except:
        return 0

def clean_integer(value):
    """Clean integer values from CSV"""
    cleaned = clean_numeric(value)
    return int(cleaned) if cleaned is not None else 0

def clean_string(value):
    """Clean string values from CSV"""
    if pd.isna(value):
        return ''
    return str(value).strip()

def clean_date(value):
    """Clean date values from CSV"""
    if pd.isna(value):
        return None
    
    try:
        return pd.to_datetime(value).date()
    except:
        return None

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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_position (position),
                    INDEX idx_team (team)
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_team (team)
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_team (team)
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_team (team)
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
                    week VARCHAR(20),
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_team (team),
                    INDEX idx_game_date (game_date)
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
                    week VARCHAR(20),
                    season INT,
                    rushing_attempts INT,
                    rushing_yards INT,
                    rushing_tds INT,
                    receptions INT,
                    receiving_yards INT,
                    receiving_tds INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_team (team),
                    INDEX idx_game_date (game_date)
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
                    week VARCHAR(20),
                    season INT,
                    receptions INT,
                    receiving_yards INT,
                    receiving_tds INT,
                    targets INT,
                    catch_pct DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_player_id (player_id),
                    INDEX idx_team (team),
                    INDEX idx_game_date (game_date)
                )
            """)
            
            logger.info("✅ All tables created successfully")
            
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        raise

def load_basic_stats(connection, csv_path):
    """Load basic stats CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Basic Stats: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
                    # Clean experience field
                    experience = clean_string(row.get('Experience', ''))
                    experience_int = 0
                    if experience and experience != '':
                        # Extract number from strings like "3 Seasons"
                        exp_match = re.search(r'(\d+)', experience)
                        if exp_match:
                            experience_int = int(exp_match.group(1))
                    
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
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Position', '')),
                        clean_string(row.get('Team', '')),
                        clean_integer(row.get('Age', 0)),
                        clean_string(row.get('Height', '')),
                        clean_integer(row.get('Weight', 0)),
                        experience_int
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Basic Stats loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Basic Stats: {e}")

def load_career_stats_passing(connection, csv_path):
    """Load career stats passing CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Career Stats Passing: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
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
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Position', '')),
                        clean_string(row.get('Team', '')),
                        clean_integer(row.get('Games Played', 0)),
                        clean_integer(row.get('Completions', 0)),
                        clean_integer(row.get('Attempts', 0)),
                        clean_numeric(row.get('Completion %', 0)),
                        clean_integer(row.get('Passing Yards', 0)),
                        clean_integer(row.get('Passing TDs', 0)),
                        clean_integer(row.get('Interceptions', 0)),
                        clean_numeric(row.get('Passer Rating', 0))
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Career Stats Passing loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Career Stats Passing: {e}")

def load_career_stats_rushing(connection, csv_path):
    """Load career stats rushing CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Career Stats Rushing: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
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
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Position', '')),
                        clean_string(row.get('Team', '')),
                        clean_integer(row.get('Games Played', 0)),
                        clean_integer(row.get('Rushing Attempts', 0)),
                        clean_integer(row.get('Rushing Yards', 0)),
                        clean_integer(row.get('Rushing TDs', 0)),
                        clean_numeric(row.get('Yards Per Attempt', 0))
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Career Stats Rushing loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Career Stats Rushing: {e}")

def load_career_stats_receiving(connection, csv_path):
    """Load career stats receiving CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Career Stats Receiving: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
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
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Position', '')),
                        clean_string(row.get('Team', '')),
                        clean_integer(row.get('Games Played', 0)),
                        clean_integer(row.get('Receptions', 0)),
                        clean_integer(row.get('Receiving Yards', 0)),
                        clean_integer(row.get('Receiving TDs', 0)),
                        clean_numeric(row.get('Yards Per Reception', 0))
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Career Stats Receiving loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Career Stats Receiving: {e}")

def load_game_logs_quarterback(connection, csv_path):
    """Load game logs quarterback CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Game Logs Quarterback: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
                    # Clean week field - handle "Preseason", "Regular Season", etc.
                    week = clean_string(row.get('Week', ''))
                    if week in ['Preseason', 'Regular Season', 'Playoffs']:
                        week = week
                    else:
                        week = str(clean_integer(week))
                    
                    cursor.execute("""
                        INSERT INTO game_logs_quarterback 
                        (player_id, name, team, opponent, game_date, week, season, completions, 
                         attempts, completion_pct, passing_yards, passing_tds, interceptions, 
                         passer_rating, rushing_attempts, rushing_yards, rushing_tds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Team', '')),
                        clean_string(row.get('Opponent', '')),
                        clean_date(row.get('Date')),
                        week,
                        clean_integer(row.get('Season', 0)),
                        clean_integer(row.get('Completions', 0)),
                        clean_integer(row.get('Attempts', 0)),
                        clean_numeric(row.get('Completion %', 0)),
                        clean_integer(row.get('Passing Yards', 0)),
                        clean_integer(row.get('Passing TDs', 0)),
                        clean_integer(row.get('Interceptions', 0)),
                        clean_numeric(row.get('Passer Rating', 0)),
                        clean_integer(row.get('Rushing Attempts', 0)),
                        clean_integer(row.get('Rushing Yards', 0)),
                        clean_integer(row.get('Rushing TDs', 0))
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Game Logs Quarterback loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Game Logs Quarterback: {e}")

def load_game_logs_runningback(connection, csv_path):
    """Load game logs running back CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Game Logs Running Back: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
                    # Clean week field
                    week = clean_string(row.get('Week', ''))
                    if week in ['Preseason', 'Regular Season', 'Playoffs']:
                        week = week
                    else:
                        week = str(clean_integer(week))
                    
                    cursor.execute("""
                        INSERT INTO game_logs_runningback 
                        (player_id, name, team, opponent, game_date, week, season, rushing_attempts, 
                         rushing_yards, rushing_tds, receptions, receiving_yards, receiving_tds)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Team', '')),
                        clean_string(row.get('Opponent', '')),
                        clean_date(row.get('Date')),
                        week,
                        clean_integer(row.get('Season', 0)),
                        clean_integer(row.get('Rushing Attempts', 0)),
                        clean_integer(row.get('Rushing Yards', 0)),
                        clean_integer(row.get('Rushing TDs', 0)),
                        clean_integer(row.get('Receptions', 0)),
                        clean_integer(row.get('Receiving Yards', 0)),
                        clean_integer(row.get('Receiving TDs', 0))
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Game Logs Running Back loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Game Logs Running Back: {e}")

def load_game_logs_wide_receiver(connection, csv_path):
    """Load game logs wide receiver CSV with data cleaning"""
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"📊 Loading Game Logs Wide Receiver: {len(df)} records")
        
        with connection.cursor() as cursor:
            for _, row in df.iterrows():
                try:
                    # Clean week field
                    week = clean_string(row.get('Week', ''))
                    if week in ['Preseason', 'Regular Season', 'Playoffs']:
                        week = week
                    else:
                        week = str(clean_integer(week))
                    
                    cursor.execute("""
                        INSERT INTO game_logs_wide_receiver 
                        (player_id, name, team, opponent, game_date, week, season, receptions, 
                         receiving_yards, receiving_tds, targets, catch_pct)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        clean_string(row.get('Player ID', '')),
                        clean_string(row.get('Name', '')),
                        clean_string(row.get('Team', '')),
                        clean_string(row.get('Opponent', '')),
                        clean_date(row.get('Date')),
                        week,
                        clean_integer(row.get('Season', 0)),
                        clean_integer(row.get('Receptions', 0)),
                        clean_integer(row.get('Receiving Yards', 0)),
                        clean_integer(row.get('Receiving TDs', 0)),
                        clean_integer(row.get('Targets', 0)),
                        clean_numeric(row.get('Catch %', 0))
                    ))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping row due to error: {e}")
                    continue
        
        connection.commit()
        logger.info("✅ Game Logs Wide Receiver loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading Game Logs Wide Receiver: {e}")

def main():
    """Main function to load all CSV data into MySQL"""
    print("🏈 NFL AI Platform - Robust CSV to MySQL Data Loader")
    print("=" * 60)
    
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
            
            total_records = 0
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    total_records += count
                    print(f"   {table}: {count:,} records")
                except:
                    print(f"   {table}: Table not found")
            
            print(f"\n🎉 Total records loaded: {total_records:,}")
        
        print("\n🎉 All CSV data loaded successfully into MySQL database!")
        
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    main()
