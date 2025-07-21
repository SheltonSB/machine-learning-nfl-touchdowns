#!/usr/bin/env python3
"""
NFL QB Touchdown Predictor - Main Orchestration Script

This script orchestrates the entire workflow:
1. Load data from CSV files into database
2. Validate data quality
3. Preprocess data for modeling
4. Train model (if needed)
5. Launch the Streamlit app

Author: Shelton Bumhe
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from database import NFLDatabase
from data_loader import NFLDataLoader
from data_validator import NFLDataValidator
from preprocess import NFLPreprocessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NFLProjectOrchestrator:
    """Orchestrates the entire NFL project workflow."""
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.db = NFLDatabase()
        self.data_loader = NFLDataLoader(db=self.db)
        self.validator = NFLDataValidator(db=self.db)
        self.preprocessor = NFLPreprocessor(db=self.db)
    
    def setup_database(self, force_reload=False):
        """
        Set up the database and load data.
        
        Args:
            force_reload (bool): Whether to reload data even if database exists
        """
        logger.info("Setting up database...")
        
        # Check if database already has data
        table_info = self.db.get_table_info()
        total_records = sum(table_info.values())
        
        if total_records > 0 and not force_reload:
            logger.info(f"Database already contains {total_records:,} records. Skipping data load.")
            logger.info("Use --force-reload to reload data anyway.")
            return True
        
        # Load all data
        try:
            self.data_loader.load_all_data()
            logger.info("Database setup completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            return False
    
    def validate_data(self):
        """
        Validate the loaded data.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating data...")
        
        try:
            results = self.validator.validate_all_data()
            all_passed = all(results.values())
            
            if all_passed:
                logger.info("✅ All validation checks passed!")
            else:
                logger.warning("⚠️ Some validation checks failed. Review the data.")
            
            return all_passed
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False
    
    def preprocess_data(self):
        """
        Preprocess data for modeling.
        
        Returns:
            bool: True if preprocessing succeeds, False otherwise
        """
        logger.info("Preprocessing data...")
        
        try:
            final_dataset = self.preprocessor.process_all()
            logger.info("✅ Preprocessing completed successfully!")
            return True
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return False
    
    def check_model(self):
        """
        Check if the model exists.
        
        Returns:
            bool: True if model exists, False otherwise
        """
        model_path = Path("models/qb_td_model.pkl")
        if model_path.exists():
            logger.info("✅ Model found!")
            return True
        else:
            logger.warning("⚠️ Model not found. You may need to train the model.")
            return False
    
    def run_complete_workflow(self, force_reload=False, skip_validation=False):
        """
        Run the complete workflow.
        
        Args:
            force_reload (bool): Whether to force reload data
            skip_validation (bool): Whether to skip validation
        """
        logger.info("🚀 Starting NFL QB Touchdown Predictor workflow...")
        
        # Step 1: Setup database
        if not self.setup_database(force_reload):
            logger.error("❌ Database setup failed. Exiting.")
            return False
        
        # Step 2: Validate data (optional)
        if not skip_validation:
            if not self.validate_data():
                logger.warning("⚠️ Data validation failed, but continuing...")
        else:
            logger.info("Skipping data validation...")
        
        # Step 3: Preprocess data
        if not self.preprocess_data():
            logger.error("❌ Preprocessing failed. Exiting.")
            return False
        
        # Step 4: Check model
        self.check_model()
        
        logger.info("✅ Workflow completed successfully!")
        return True
    
    def launch_app(self):
        """Launch the Streamlit app."""
        logger.info("🌐 Launching Streamlit app...")
        
        app_path = Path("app/app.py")
        if not app_path.exists():
            logger.error(f"❌ App file not found: {app_path}")
            return False
        
        try:
            import subprocess
            import streamlit
            
            # Check if streamlit is available
            logger.info("Starting Streamlit app...")
            logger.info("The app will open in your browser automatically.")
            logger.info("Press Ctrl+C to stop the app.")
            
            # Launch the app
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", 
                str(app_path), "--server.port", "8501"
            ])
            
        except ImportError:
            logger.error("❌ Streamlit not installed. Install with: pip install streamlit")
            return False
        except KeyboardInterrupt:
            logger.info("App stopped by user.")
            return True
        except Exception as e:
            logger.error(f"❌ Error launching app: {e}")
            return False
    
    def show_status(self):
        """Show the current status of the project."""
        logger.info("📊 Project Status Report")
        logger.info("=" * 50)
        
        # Database status
        table_info = self.db.get_table_info()
        total_records = sum(table_info.values())
        
        logger.info("Database Status:")
        for table, count in table_info.items():
            logger.info(f"  {table:20}: {count:,} records")
        
        logger.info(f"Total records: {total_records:,}")
        
        # Model status
        model_exists = self.check_model()
        
        # Processed data status
        processed_file = Path("data/processed/final_dataset.csv")
        if processed_file.exists():
            import pandas as pd
            try:
                df = pd.read_csv(processed_file)
                logger.info(f"Processed dataset: {len(df):,} records, {len(df.columns)} features")
            except Exception as e:
                logger.warning(f"Could not read processed dataset: {e}")
        else:
            logger.warning("Processed dataset not found")
        
        logger.info("=" * 50)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="NFL QB Touchdown Predictor - Main Orchestration Script"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true",
        help="Set up database and load data"
    )
    
    parser.add_argument(
        "--validate", 
        action="store_true",
        help="Validate loaded data"
    )
    
    parser.add_argument(
        "--preprocess", 
        action="store_true",
        help="Preprocess data for modeling"
    )
    
    parser.add_argument(
        "--app", 
        action="store_true",
        help="Launch the Streamlit app"
    )
    
    parser.add_argument(
        "--workflow", 
        action="store_true",
        help="Run complete workflow (setup + validate + preprocess)"
    )
    
    parser.add_argument(
        "--status", 
        action="store_true",
        help="Show project status"
    )
    
    parser.add_argument(
        "--force-reload", 
        action="store_true",
        help="Force reload data even if database exists"
    )
    
    parser.add_argument(
        "--skip-validation", 
        action="store_true",
        help="Skip data validation in workflow"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = NFLProjectOrchestrator()
    
    try:
        if args.status:
            orchestrator.show_status()
        
        elif args.setup:
            orchestrator.setup_database(args.force_reload)
        
        elif args.validate:
            orchestrator.validate_data()
        
        elif args.preprocess:
            orchestrator.preprocess_data()
        
        elif args.workflow:
            orchestrator.run_complete_workflow(args.force_reload, args.skip_validation)
        
        elif args.app:
            orchestrator.launch_app()
        
        else:
            # Default: run complete workflow and launch app
            if orchestrator.run_complete_workflow(args.force_reload, args.skip_validation):
                logger.info("🎉 Ready to launch app!")
                user_input = input("Press Enter to launch the Streamlit app (or Ctrl+C to exit): ")
                if user_input == "":
                    orchestrator.launch_app()
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 