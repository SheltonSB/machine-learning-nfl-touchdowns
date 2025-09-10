"""
NFL QB Touchdown Predictor — Database-Driven Preprocessing Script

This script loads data from the SQLite database, creates rolling features,
and prepares the final dataset for model training.

Author: Shelton Bumhe
"""
                                                   
import pandas as pd
import numpy as np
import os
import logging
from database import NFLDatabase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLPreprocessor:
    """Handles data preprocessing for NFL QB touchdown prediction."""
    
    def __init__(self, db: NFLDatabase = None):
        """
        Initialize the preprocessor.
        
        Args:
            db (NFLDatabase): Database instance
        """
        self.db = db or NFLDatabase()
        self.processed_data_path = '../data/processed/'
        
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_path, exist_ok=True)
    
    def load_data_from_db(self) -> tuple:
        """
        Load data from the database.
        
        Returns:
            tuple: (game_logs, basic_stats) DataFrames
        """
        logger.info("Loading data from database...")
        
        self.db.connect()
        
        try:
            # Load QB game logs with stats
            game_logs_query = """
            SELECT 
                gl.id,
                gl.player_id,
                gl.name,
                gl.year,
                gl.week,
                gl.team,
                gl.opponent,
                gl.game_date,
                gl.home_away,
                gl.result,
                qs.passing_yards,
                qs.td_passes,
                qs.interceptions,
                qs.passes_attempted,
                qs.passes_completed,
                qs.completion_percentage,
                qs.yards_per_attempt,
                qs.passer_rating,
                qs.threw_td
            FROM game_logs gl
            JOIN qb_stats qs ON gl.id = qs.game_log_id
            WHERE gl.position = 'QB'
            ORDER BY gl.player_id, gl.year, gl.week
            """
            
            game_logs = pd.read_sql_query(game_logs_query, self.db.conn)
            
            # Load basic stats
            basic_stats_query = """
            SELECT player_id, name, age, height, weight, experience, position
            FROM basic_stats
            WHERE position = 'QB'
            """
            
            basic_stats = pd.read_sql_query(basic_stats_query, self.db.conn)
            
            logger.info(f"Loaded {len(game_logs)} QB game records")
            logger.info(f"Loaded {len(basic_stats)} QB basic stats records")
            
            return game_logs, basic_stats
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise
        finally:
            self.db.disconnect()
    
    def clean_game_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the game logs DataFrame.
        
        Args:
            df (DataFrame): Raw game logs from database
            
        Returns:
            DataFrame: Cleaned game logs
        """
        logger.info("Cleaning game logs...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove any rows with missing critical data
        critical_cols = ['player_id', 'name', 'year', 'week', 'passing_yards', 'td_passes']
        df_clean = df_clean.dropna(subset=critical_cols)
        
        # Ensure numeric types
        numeric_cols = ['passing_yards', 'td_passes', 'interceptions', 'passes_attempted',
                       'passes_completed', 'completion_percentage', 'yards_per_attempt', 'passer_rating']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill missing values with reasonable defaults
        df_clean['interceptions'] = df_clean['interceptions'].fillna(0)
        df_clean['completion_percentage'] = df_clean['completion_percentage'].fillna(50.0)
        df_clean['yards_per_attempt'] = df_clean['yards_per_attempt'].fillna(6.0)
        df_clean['passer_rating'] = df_clean['passer_rating'].fillna(70.0)
        
        # Ensure TD target is binary
        df_clean['threw_td'] = (df_clean['td_passes'] > 0).astype(int)
        
        logger.info(f"Cleaned game logs: {len(df_clean)} records")
        return df_clean
    
    def clean_basic_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the basic stats DataFrame.
        
        Args:
            df (DataFrame): Raw basic stats from database
            
        Returns:
            DataFrame: Cleaned basic stats
        """
        logger.info("Cleaning basic stats...")
        
        df_clean = df.copy()
        
        # Remove rows with missing critical data
        df_clean = df_clean.dropna(subset=['player_id', 'name'])
        
        # Ensure numeric types
        numeric_cols = ['age', 'height', 'weight', 'experience']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Fill missing values with reasonable defaults
        df_clean['age'] = df_clean['age'].fillna(25)
        df_clean['height'] = df_clean['height'].fillna(74)  # Average QB height
        df_clean['weight'] = df_clean['weight'].fillna(220)  # Average QB weight
        df_clean['experience'] = df_clean['experience'].fillna(3)
        
        logger.info(f"Cleaned basic stats: {len(df_clean)} records")
        return df_clean
    
    def merge_data(self, game_logs: pd.DataFrame, basic_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Merge game logs with basic stats.
        
        Args:
            game_logs (DataFrame): Cleaned game logs
            basic_stats (DataFrame): Cleaned basic stats
            
        Returns:
            DataFrame: Merged dataset
        """
        logger.info("Merging datasets...")
        
        # Merge on player_id
        merged = pd.merge(game_logs, basic_stats, on='player_id', how='left', suffixes=('', '_basic'))
        
        # Clean up duplicate name columns
        if 'name_basic' in merged.columns:
            merged = merged.drop(columns=['name_basic'])
        
        # Check merge quality
        missing_basic = merged['age'].isna().sum()
        if missing_basic > 0:
            logger.warning(f"Missing basic stats for {missing_basic} game records")
        
        logger.info(f"Merged dataset: {len(merged)} records")
        return merged
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling average features for selected stats.
        
        Args:
            df (DataFrame): Merged dataset
            
        Returns:
            DataFrame: Dataset with rolling features
        """
        logger.info("Creating rolling features...")
        
        # Sort by player and time
        df_sorted = df.sort_values(by=['player_id', 'year', 'week']).copy()
        
        # Features to create rolling averages for
        rolling_features = ['passing_yards', 'td_passes', 'passes_attempted', 'interceptions']
        
        for feature in rolling_features:
            if feature in df_sorted.columns:
                # Create rolling average (excluding current game)
                df_sorted[f'{feature}_roll3'] = (
                    df_sorted.groupby('player_id')[feature]
                    .shift(1)  # Don't include current game
                    .rolling(window=3, min_periods=1)
                    .mean()
                )
                
                # Create rolling average (including current game)
                df_sorted[f'{feature}_roll3_current'] = (
                    df_sorted.groupby('player_id')[feature]
                    .rolling(window=3, min_periods=1)
                    .mean()
                )
        
        # Create additional features
        df_sorted['completion_rate'] = df_sorted['completion_percentage'] / 100.0
        df_sorted['td_rate'] = df_sorted['td_passes'] / df_sorted['passes_attempted'].replace(0, 1)
        df_sorted['int_rate'] = df_sorted['interceptions'] / df_sorted['passes_attempted'].replace(0, 1)
        
        # Create rolling rates
        df_sorted['td_rate_roll3'] = (
            df_sorted.groupby('player_id')['td_rate']
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
        
        df_sorted['completion_rate_roll3'] = (
            df_sorted.groupby('player_id')['completion_rate']
            .shift(1)
            .rolling(window=3, min_periods=1)
            .mean()
        )
        
        logger.info("Rolling features created successfully")
        return df_sorted
    
    def create_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the final dataset for model training.
        
        Args:
            df (DataFrame): Dataset with rolling features
            
        Returns:
            DataFrame: Final dataset ready for modeling
        """
        logger.info("Creating final dataset...")
        
        # Select features for modeling
        feature_columns = [
            # Player characteristics
            'age', 'height', 'weight', 'experience',
            
            # Current game stats
            'passing_yards', 'td_passes', 'interceptions', 'passes_attempted',
            'completion_percentage', 'yards_per_attempt', 'passer_rating',
            
            # Rolling averages (excluding current game)
            'passing_yards_roll3', 'td_passes_roll3', 'passes_attempted_roll3',
            'td_rate_roll3', 'completion_rate_roll3',
            
            # Target variable
            'threw_td'
        ]
        
        # Filter to only include rows with all features
        final_df = df[feature_columns].copy()
        final_df = final_df.dropna()
        
        # Ensure all features are numeric
        for col in final_df.columns:
            if col != 'threw_td':  # Keep target as is
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        
        final_df = final_df.dropna()
        
        logger.info(f"Final dataset: {len(final_df)} records, {len(final_df.columns)} features")
        logger.info(f"Feature columns: {list(final_df.columns)}")
        
        return final_df
    
    def save_final_dataset(self, df: pd.DataFrame):
        """
        Save the final dataset to CSV.
        
        Args:
            df (DataFrame): Final dataset
        """
        output_path = os.path.join(self.processed_data_path, 'final_dataset.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Final dataset saved to {output_path}")
        
        # Print dataset summary
        logger.info("\n" + "="*50)
        logger.info("FINAL DATASET SUMMARY")
        logger.info("="*50)
        logger.info(f"Total records: {len(df):,}")
        logger.info(f"Total features: {len(df.columns) - 1}")  # Exclude target
        logger.info(f"Target distribution:")
        target_dist = df['threw_td'].value_counts()
        for value, count in target_dist.items():
            percentage = count / len(df) * 100
            status = "TD" if value == 1 else "No TD"
            logger.info(f"  {status}: {count:,} ({percentage:.1f}%)")
        logger.info("="*50)
    
    def process_all(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting complete preprocessing pipeline...")
        
        try:
            # Load data from database
            game_logs, basic_stats = self.load_data_from_db()
            
            # Clean data
            game_logs_clean = self.clean_game_logs(game_logs)
            basic_stats_clean = self.clean_basic_stats(basic_stats)
            
            # Merge datasets
            merged = self.merge_data(game_logs_clean, basic_stats_clean)
            
            # Create rolling features
            with_features = self.create_rolling_features(merged)
            
            # Create final dataset
            final_dataset = self.create_final_dataset(with_features)
            
            # Save final dataset
            self.save_final_dataset(final_dataset)
            
            logger.info("Preprocessing completed successfully!")
            
            return final_dataset
            
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

def main():
    """Main function to run preprocessing."""
    preprocessor = NFLPreprocessor()
    final_dataset = preprocessor.process_all()
    return final_dataset

if __name__ == '__main__':
    main()
