#!/usr/bin/env python3
"""
NFL QB Touchdown Predictor - Main Orchestration Script

This script orchestrates the entire workflow:
1. Load data from CSV files into database
2. Validate data quality
3. Preprocess data for modeling
4. Train model (if needed)
5. (Optional) Deploy or serve predictions through external services

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
from train_model import main as train_model_main
from explain_shap import main as shap_main

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
                logger.info("All validation checks passed.")
            else:
                logger.warning("Some validation checks failed. Review the data.")
            
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
            logger.info("Preprocessing completed successfully.")
            return True
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return False
    
    def check_model(self):
        """Check for the presence of a trained model artifact."""

        tf_model_path = Path("models/qb_td_model.keras")
        legacy_model_path = Path("models/qb_td_model.pkl")

        if tf_model_path.exists():
            logger.info("TensorFlow model found at %s", tf_model_path)
            return True

        if legacy_model_path.exists():
            logger.warning(
                "Found legacy XGBoost model at %s. Consider retraining with TensorFlow (python main.py --train-model).",
                legacy_model_path,
            )
            return True

        logger.warning(
            "Model artifact not found. Run 'python main.py --train-model' after preprocessing to train the TensorFlow model."
        )
        return False
    
    def train_model(self, force: bool = False, generate_shap: bool = False) -> bool:
        """Train the TensorFlow model and optionally generate SHAP summary."""

        dataset_path = Path("data/processed/final_dataset.csv")
        model_path = Path("models/qb_td_model.keras")

        if not dataset_path.exists():
            logger.error(
                "Processed dataset not found at %s. Run preprocessing before training.",
                dataset_path,
            )
            return False

        if model_path.exists() and not force:
            logger.info(
                "Model already exists at %s. Use --force-train to retrain.",
                model_path,
            )
            if generate_shap:
                return self.generate_shap_summary()
            return True

        try:
            logger.info("Starting TensorFlow model training...")
            exit_code = train_model_main()
            if exit_code != os.EX_OK:
                logger.error("Training script exited with status %s", exit_code)
                return False
            logger.info("Model training completed successfully.")

            if generate_shap:
                return self.generate_shap_summary()

            return True
        except Exception as exc:
            logger.error("Error during model training: %s", exc)
            return False

    def generate_shap_summary(self) -> bool:
        """Generate SHAP summary visualization for the trained model."""

        try:
            logger.info("Generating SHAP summary plot...")
            exit_code = shap_main()
            if exit_code != os.EX_OK and exit_code != 0:
                logger.error("SHAP script exited with status %s", exit_code)
                return False
            logger.info("SHAP summary generated successfully.")
            return True
        except Exception as exc:
            logger.error("Error generating SHAP summary: %s", exc)
            return False

    def run_complete_workflow(
        self,
        force_reload: bool = False,
        skip_validation: bool = False,
        train: bool = False,
        generate_shap: bool = False,
        force_train: bool = False,
    ):
        """
        Run the complete workflow.
        
        Args:
            force_reload (bool): Whether to force reload data
            skip_validation (bool): Whether to skip validation
            train (bool): Whether to train the TensorFlow model after preprocessing
            generate_shap (bool): Whether to generate SHAP summary
            force_train (bool): Whether to retrain even if a model exists
        """
        logger.info("Starting NFL QB Touchdown Predictor workflow...")
        
        # Step 1: Setup database
        if not self.setup_database(force_reload):
            logger.error("Database setup failed. Exiting.")
            return False
        
        # Step 2: Validate data (optional)
        if not skip_validation:
            if not self.validate_data():
                logger.warning("Data validation failed, but continuing...")
        else:
            logger.info("Skipping data validation...")
        
        # Step 3: Preprocess data
        if not self.preprocess_data():
            logger.error("Preprocessing failed. Exiting.")
            return False
        
        train_needed = train or force_train

        if train_needed:
            if not self.train_model(force=force_train, generate_shap=generate_shap):
                logger.error("Model training failed. Exiting.")
                return False
            self.check_model()
        else:
            model_ready = self.check_model()
            if generate_shap and model_ready:
                if not self.generate_shap_summary():
                    logger.error("Failed to generate SHAP summary. Exiting.")
                    return False
            elif generate_shap:
                logger.error("Cannot generate SHAP summary because model is missing.")
                return False
        
        logger.info("Workflow completed successfully.")
        return True
    
    def show_status(self):
        """Show the current status of the project."""
        logger.info("Project status report")
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
    
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train the TensorFlow model using the processed dataset"
    )
    
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain the TensorFlow model even if an artifact already exists"
    )
    
    parser.add_argument(
        "--generate-shap",
        action="store_true",
        help="Generate SHAP summary visualization after training"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = NFLProjectOrchestrator()
    
    try:
        if args.status:
            orchestrator.show_status()
        
        elif args.train_model and not args.workflow:
            orchestrator.train_model(
                force=args.force_train, generate_shap=args.generate_shap
            )

        elif args.generate_shap and not args.workflow:
            orchestrator.generate_shap_summary()

        elif args.setup:
            orchestrator.setup_database(args.force_reload)
        
        elif args.validate:
            orchestrator.validate_data()
        
        elif args.preprocess:
            orchestrator.preprocess_data()
        
        elif args.workflow:
            orchestrator.run_complete_workflow(
                args.force_reload,
                args.skip_validation,
                train=args.train_model or args.force_train,
                generate_shap=args.generate_shap,
                force_train=args.force_train,
            )
        
        else:
            # Default: run complete workflow
            orchestrator.run_complete_workflow(
                args.force_reload,
                args.skip_validation,
                train=args.train_model or args.force_train,
                generate_shap=args.generate_shap,
                force_train=args.force_train,
            )
    
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
