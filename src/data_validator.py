"""
NFL Data Validator

This script validates the loaded data in the database to ensure data quality
and completeness.

Author: Shelton Bumhe
"""

import pandas as pd
import numpy as np
import logging
from database import NFLDatabase
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLDataValidator:
    """Validates NFL data quality and completeness."""
    
    def __init__(self, db: NFLDatabase = None):
        """
        Initialize the data validator.
        
        Args:
            db (NFLDatabase): Database instance
        """
        self.db = db or NFLDatabase()
    
    def validate_all_data(self) -> Dict[str, bool]:
        """
        Run all validation checks.
        
        Returns:
            Dict[str, bool]: Results of all validation checks
        """
        logger.info("Starting comprehensive data validation...")
        
        results = {}
        
        # Basic validation checks
        results['basic_stats_loaded'] = self._validate_basic_stats()
        results['game_logs_loaded'] = self._validate_game_logs()
        results['qb_stats_loaded'] = self._validate_qb_stats()
        results['career_stats_loaded'] = self._validate_career_stats()
        
        # Data quality checks
        results['no_duplicate_players'] = self._check_duplicate_players()
        results['consistent_player_ids'] = self._check_player_id_consistency()
        results['valid_date_ranges'] = self._check_date_ranges()
        results['reasonable_stat_values'] = self._check_stat_reasonableness()
        
        # QB-specific checks
        results['qb_data_completeness'] = self._check_qb_data_completeness()
        results['td_predictions_valid'] = self._check_td_predictions()
        
        # Print summary
        self._print_validation_summary(results)
        
        return results
    
    def _validate_basic_stats(self) -> bool:
        """Validate basic player statistics."""
        logger.info("Validating basic stats...")
        
        self.db.connect()
        
        try:
            # Check if table exists and has data
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM basic_stats", self.db.conn)
            if count['count'].iloc[0] == 0:
                logger.error("Basic stats table is empty")
                return False
            
            # Check for required columns
            columns = pd.read_sql_query("PRAGMA table_info(basic_stats)", self.db.conn)
            required_cols = ['player_id', 'name', 'age', 'height', 'weight']
            
            for col in required_cols:
                if col not in columns['name'].values:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            # Check for null values in key fields
            null_check = pd.read_sql_query("""
                SELECT COUNT(*) as null_count 
                FROM basic_stats 
                WHERE player_id IS NULL OR name IS NULL
            """, self.db.conn)
            
            if null_check['null_count'].iloc[0] > 0:
                logger.warning("Found null values in key fields")
            
            logger.info("Basic stats validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating basic stats: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _validate_game_logs(self) -> bool:
        """Validate game logs data."""
        logger.info("Validating game logs...")
        
        self.db.connect()
        
        try:
            # Check if table exists and has data
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM game_logs", self.db.conn)
            if count['count'].iloc[0] == 0:
                logger.error("Game logs table is empty")
                return False
            
            # Check for required columns
            columns = pd.read_sql_query("PRAGMA table_info(game_logs)", self.db.conn)
            required_cols = ['player_id', 'name', 'year', 'week', 'team', 'opponent']
            
            for col in required_cols:
                if col not in columns['name'].values:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            # Check for valid year/week ranges
            year_check = pd.read_sql_query("""
                SELECT MIN(year) as min_year, MAX(year) as max_year,
                       MIN(week) as min_week, MAX(week) as max_week
                FROM game_logs
            """, self.db.conn)
            
            min_year, max_year = year_check['min_year'].iloc[0], year_check['max_year'].iloc[0]
            min_week, max_week = year_check['min_week'].iloc[0], year_check['max_week'].iloc[0]
            
            if min_year < 2000 or max_year > 2030:
                logger.warning(f"Year range seems unusual: {min_year}-{max_year}")
            
            if min_week < 1 or max_week > 18:
                logger.warning(f"Week range seems unusual: {min_week}-{max_week}")
            
            logger.info("Game logs validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating game logs: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _validate_qb_stats(self) -> bool:
        """Validate QB-specific statistics."""
        logger.info("Validating QB stats...")
        
        self.db.connect()
        
        try:
            # Check if table exists and has data
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM qb_stats", self.db.conn)
            if count['count'].iloc[0] == 0:
                logger.error("QB stats table is empty")
                return False
            
            # Check for required columns
            columns = pd.read_sql_query("PRAGMA table_info(qb_stats)", self.db.conn)
            required_cols = ['game_log_id', 'passing_yards', 'td_passes', 'threw_td']
            
            for col in required_cols:
                if col not in columns['name'].values:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            # Check TD prediction consistency
            td_check = pd.read_sql_query("""
                SELECT COUNT(*) as inconsistent_count
                FROM qb_stats
                WHERE (td_passes > 0 AND threw_td = 0) OR (td_passes = 0 AND threw_td = 1)
            """, self.db.conn)
            
            if td_check['inconsistent_count'].iloc[0] > 0:
                logger.error("Found inconsistent TD predictions")
                return False
            
            logger.info("QB stats validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating QB stats: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _validate_career_stats(self) -> bool:
        """Validate career statistics."""
        logger.info("Validating career stats...")
        
        self.db.connect()
        
        try:
            # Check if table exists and has data
            count = pd.read_sql_query("SELECT COUNT(*) as count FROM career_stats", self.db.conn)
            if count['count'].iloc[0] == 0:
                logger.error("Career stats table is empty")
                return False
            
            # Check for required columns
            columns = pd.read_sql_query("PRAGMA table_info(career_stats)", self.db.conn)
            required_cols = ['player_id', 'name', 'year', 'team']
            
            for col in required_cols:
                if col not in columns['name'].values:
                    logger.error(f"Missing required column: {col}")
                    return False
            
            logger.info("Career stats validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating career stats: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _check_duplicate_players(self) -> bool:
        """Check for duplicate player IDs."""
        logger.info("Checking for duplicate players...")
        
        self.db.connect()
        
        try:
            # Check basic stats for duplicates
            duplicates = pd.read_sql_query("""
                SELECT player_id, COUNT(*) as count
                FROM basic_stats
                GROUP BY player_id
                HAVING COUNT(*) > 1
            """, self.db.conn)
            
            if len(duplicates) > 0:
                logger.error(f"Found {len(duplicates)} duplicate player IDs")
                return False
            
            logger.info("No duplicate players found")
            return True
            
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _check_player_id_consistency(self) -> bool:
        """Check player ID consistency across tables."""
        logger.info("Checking player ID consistency...")
        
        self.db.connect()
        
        try:
            # Get all player IDs from basic stats
            basic_ids = pd.read_sql_query("SELECT player_id FROM basic_stats", self.db.conn)
            
            # Check if all game log player IDs exist in basic stats
            orphaned_logs = pd.read_sql_query("""
                SELECT DISTINCT gl.player_id
                FROM game_logs gl
                LEFT JOIN basic_stats bs ON gl.player_id = bs.player_id
                WHERE bs.player_id IS NULL
            """, self.db.conn)
            
            if len(orphaned_logs) > 0:
                logger.warning(f"Found {len(orphaned_logs)} orphaned game log entries")
            
            logger.info("Player ID consistency check completed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking player ID consistency: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _check_date_ranges(self) -> bool:
        """Check for reasonable date ranges."""
        logger.info("Checking date ranges...")
        
        self.db.connect()
        
        try:
            # Check year ranges
            year_stats = pd.read_sql_query("""
                SELECT MIN(year) as min_year, MAX(year) as max_year,
                       COUNT(DISTINCT year) as unique_years
                FROM game_logs
            """, self.db.conn)
            
            min_year = year_stats['min_year'].iloc[0]
            max_year = year_stats['max_year'].iloc[0]
            unique_years = year_stats['unique_years'].iloc[0]
            
            if min_year < 2000 or max_year > 2030:
                logger.warning(f"Year range seems unusual: {min_year}-{max_year}")
            
            logger.info(f"Data spans {unique_years} years ({min_year}-{max_year})")
            return True
            
        except Exception as e:
            logger.error(f"Error checking date ranges: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _check_stat_reasonableness(self) -> bool:
        """Check if stat values are reasonable."""
        logger.info("Checking stat reasonableness...")
        
        self.db.connect()
        
        try:
            # Check QB stats for reasonable values
            qb_stats_check = pd.read_sql_query("""
                SELECT 
                    MIN(passing_yards) as min_yards,
                    MAX(passing_yards) as max_yards,
                    MIN(td_passes) as min_tds,
                    MAX(td_passes) as max_tds,
                    MIN(passes_attempted) as min_attempts,
                    MAX(passes_attempted) as max_attempts
                FROM qb_stats
                WHERE passing_yards IS NOT NULL
            """, self.db.conn)
            
            min_yards = qb_stats_check['min_yards'].iloc[0]
            max_yards = qb_stats_check['max_yards'].iloc[0]
            min_tds = qb_stats_check['min_tds'].iloc[0]
            max_tds = qb_stats_check['max_tds'].iloc[0]
            
            # Check for reasonable ranges
            if min_yards < 0 or max_yards > 600:
                logger.warning(f"Passing yards range seems unusual: {min_yards}-{max_yards}")
            
            if min_tds < 0 or max_tds > 10:
                logger.warning(f"TD passes range seems unusual: {min_tds}-{max_tds}")
            
            logger.info("Stat reasonableness check completed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking stat reasonableness: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _check_qb_data_completeness(self) -> bool:
        """Check QB data completeness."""
        logger.info("Checking QB data completeness...")
        
        self.db.connect()
        
        try:
            # Count QBs with complete data
            qb_completeness = pd.read_sql_query("""
                SELECT 
                    COUNT(DISTINCT bs.player_id) as total_qbs,
                    COUNT(DISTINCT gl.player_id) as qbs_with_games,
                    COUNT(DISTINCT qs.game_log_id) as total_games
                FROM basic_stats bs
                LEFT JOIN game_logs gl ON bs.player_id = gl.player_id AND gl.position = 'QB'
                LEFT JOIN qb_stats qs ON gl.id = qs.game_log_id
                WHERE bs.position = 'QB'
            """, self.db.conn)
            
            total_qbs = qb_completeness['total_qbs'].iloc[0]
            qbs_with_games = qb_completeness['qbs_with_games'].iloc[0]
            total_games = qb_completeness['total_games'].iloc[0]
            
            logger.info(f"QB Data Summary:")
            logger.info(f"  Total QBs: {total_qbs}")
            logger.info(f"  QBs with game data: {qbs_with_games}")
            logger.info(f"  Total QB games: {total_games}")
            
            if qbs_with_games == 0:
                logger.error("No QB game data found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking QB data completeness: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _check_td_predictions(self) -> bool:
        """Check TD prediction target validity."""
        logger.info("Checking TD predictions...")
        
        self.db.connect()
        
        try:
            # Check TD prediction distribution
            td_distribution = pd.read_sql_query("""
                SELECT 
                    threw_td,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM qb_stats), 2) as percentage
                FROM qb_stats
                GROUP BY threw_td
                ORDER BY threw_td
            """, self.db.conn)
            
            logger.info("TD Prediction Distribution:")
            for _, row in td_distribution.iterrows():
                td_status = "TD" if row['threw_td'] == 1 else "No TD"
                logger.info(f"  {td_status}: {row['count']:,} games ({row['percentage']}%)")
            
            # Check for balanced classes
            if len(td_distribution) == 2:
                td_rate = td_distribution[td_distribution['threw_td'] == 1]['percentage'].iloc[0]
                if td_rate < 20 or td_rate > 80:
                    logger.warning(f"TD rate ({td_rate}%) might be imbalanced for modeling")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking TD predictions: {e}")
            return False
        finally:
            self.db.disconnect()
    
    def _print_validation_summary(self, results: Dict[str, bool]):
        """Print validation summary."""
        logger.info("\n" + "="*60)
        logger.info("DATA VALIDATION SUMMARY")
        logger.info("="*60)
        
        passed = sum(results.values())
        total = len(results)
        
        for check, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{check:30}: {status}")
        
        logger.info("="*60)
        logger.info(f"Overall: {passed}/{total} checks passed ({passed/total*100:.1f}%)")
        logger.info("="*60)
        
        if passed == total:
            logger.info("🎉 All validation checks passed! Data is ready for modeling.")
        else:
            logger.warning("⚠️  Some validation checks failed. Please review the data.")

def main():
    """Main function to run data validation."""
    validator = NFLDataValidator()
    results = validator.validate_all_data()
    
    # Return exit code based on validation results
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    exit(main()) 