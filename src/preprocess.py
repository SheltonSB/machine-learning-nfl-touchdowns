"""
NFL QB Touchdown Predictor — Data Preprocessing Script

This script loads raw quarterback game data, career statistics, and basic player information,
cleans and merges the datasets, generates rolling features (e.g., last 3 games’ stats), 
and creates a final processed dataset for model training.

Author: Shelton Bumhe
"""

import pandas as pd
import numpy as np
import os

# Define paths for raw and processed data
RAW_DATA_PATH = '../data/raw/'
PROCESSED_DATA_PATH = '../data/processed/'

def load_data():
    """
    Loads raw CSV files from the data/raw/ directory.

    Returns:
        game_logs (DataFrame): Game-by-game quarterback stats.
        career_stats (DataFrame): Career-level QB stats by year.
        basic_stats (DataFrame): Player bio and physical data.
    """
    game_logs = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Game_Logs_Quarterback.csv'))
    career_stats = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Career_Stats_Passing.csv'))
    basic_stats = pd.read_csv(os.path.join(RAW_DATA_PATH, 'Basic_Stats.csv'))
    return game_logs, career_stats, basic_stats

def clean_game_logs(df):
    """
    Cleans the game logs DataFrame:
    - Removes preseason games
    - Converts '--' to NaN, then to numeric
    - Creates binary TD label (threw_td)

    Args:
        df (DataFrame): Raw game logs.

    Returns:
        DataFrame: Cleaned game logs with binary target.
    """
    # Drop the Position column (mostly NaN)
    df = df.drop(columns=['Position'], errors='ignore')

    # Keep only regular season games
    df = df[df['Season'] == 'Regular Season']

    # Replace string placeholders with NaN
    df = df.replace('--', np.nan)

    # Convert object columns to numeric where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                continue

    # Create binary classification target: 1 if at least 1 TD thrown
    df['threw_td'] = (df['TD Passes'] > 0).astype(int)

    return df

def clean_basic_stats(df):
    """
    Cleans the basic stats DataFrame:
    - Drops non-essential and sparse columns
    - Extracts numeric values from 'Experience'

    Args:
        df (DataFrame): Raw basic player info.

    Returns:
        DataFrame: Cleaned player bios.
    """
    drop_cols = [
        'Birth Place', 'Birthday', 'College', 'Current Status', 'Current Team',
        'High School', 'High School Location', 'Number', 'Position', 'Years Played'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Extract number of years from Experience strings like "3 Seasons", "5th season"
    df['Experience'] = df['Experience'].str.extract(r'(\d+)').astype(float)

    return df


def merge_data(game_logs, basic_stats):
    # Drop the duplicate Name column to avoid Name_x / Name_y clash
    basic_stats = basic_stats.drop(columns=['Name'], errors='ignore')

    merged = pd.merge(game_logs, basic_stats, on='Player Id', how='left')

    # After merge, restore Name column original name
    merged = merged.rename(columns={'Name_x': 'Name'})
    return merged


def create_rolling_features(df):
    """
    Creates 3-game rolling average features for selected stats.

    Args:
        df (DataFrame): Merged dataset with game logs.

    Returns:
        DataFrame: Dataset with new rolling average features.
    """
    df = df.sort_values(by=['Name', 'Year', 'Week'])

    rolling_features = ['Passing Yards', 'TD Passes', 'Passes Attempted']
    for feature in rolling_features:
        df[f'{feature}_roll3'] = (
            df.groupby('Name')[feature]
              .shift(1)  # Don’t include current game
              .rolling(window=3, min_periods=1)
              .mean()
        )

    return df

def save_final(df):
    """
    Saves the final processed dataset to data/processed/.

    Args:
        df (DataFrame): Final dataset ready for model training.
    """
    output_path = os.path.join(PROCESSED_DATA_PATH, 'final_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Final dataset saved to {output_path}")

def main():
    """Main script execution flow."""
    print("Loading data...")
    game_logs, career_stats, basic_stats = load_data()

    print("Cleaning game logs...")
    game_logs = clean_game_logs(game_logs)

    print("Cleaning basic stats...")
    basic_stats = clean_basic_stats(basic_stats)

    print(" Merging datasets...")
    merged = merge_data(game_logs, basic_stats)

    print("Creating rolling features...")
    final_df = create_rolling_features(merged)

    print("Saving final dataset...")
    save_final(final_df)

if __name__ == '__main__':
    main()
