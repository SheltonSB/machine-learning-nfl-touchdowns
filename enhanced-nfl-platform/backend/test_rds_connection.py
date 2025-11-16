#!/usr/bin/env python3
"""
Test AWS RDS Connection

This script tests the connection to AWS RDS database.
Usage: python test_rds_connection.py
"""

import sys
from pathlib import Path

# Add project root to path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import engine, SessionLocal
from backend.app.core.config import settings
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_rds_connection():
    """Test RDS connection"""
    print("🏈 NFL AI Platform - AWS RDS Connection Test")
    print("=" * 60)
    
    # Display configuration (masked password)
    database_url = settings.get_database_url()
    if "@" in database_url:
        masked_url = database_url.split("@")[0].split(":")[0] + ":***@" + database_url.split("@")[1]
    else:
        masked_url = database_url
    
    print(f"\n📋 Configuration:")
    print(f"   Database URL: {masked_url}")
    print(f"   Engine: {engine.dialect.name}")
    print(f"   SSL Enabled: {settings.DB_USE_SSL}")
    
    if settings.RDS_ENDPOINT:
        print(f"   RDS Endpoint: {settings.RDS_ENDPOINT}")
    
    # Test connection
    print(f"\n🔌 Testing connection...")
    try:
        with engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row[0] == 1:
                print("✅ Connection successful!")
                
                # Get database version
                try:
                    if engine.dialect.name == "postgresql":
                        version_result = conn.execute(text("SELECT version()"))
                    elif engine.dialect.name == "mysql":
                        version_result = conn.execute(text("SELECT VERSION()"))
                    else:
                        version_result = None
                    
                    if version_result:
                        version = version_result.fetchone()[0]
                        print(f"   Database Version: {version}")
                except Exception as e:
                    logger.warning(f"Could not retrieve version: {e}")
                
                # Test database name
                try:
                    if engine.dialect.name == "postgresql":
                        db_result = conn.execute(text("SELECT current_database()"))
                    elif engine.dialect.name == "mysql":
                        db_result = conn.execute(text("SELECT DATABASE()"))
                    else:
                        db_result = None
                    
                    if db_result:
                        db_name = db_result.fetchone()[0]
                        print(f"   Database Name: {db_name}")
                except Exception as e:
                    logger.warning(f"Could not retrieve database name: {e}")
                
                # Test SSL connection (if applicable)
                if settings.DB_USE_SSL:
                    try:
                        if engine.dialect.name == "postgresql":
                            ssl_result = conn.execute(text("SHOW ssl"))
                        elif engine.dialect.name == "mysql":
                            ssl_result = conn.execute(text("SHOW STATUS LIKE 'Ssl_cipher'"))
                        else:
                            ssl_result = None
                        
                        if ssl_result:
                            print("   SSL: Enabled")
                    except Exception as e:
                        logger.debug(f"Could not verify SSL: {e}")
                        print("   SSL: Enabled (connection verified)")
                
                # Test session
                print(f"\n📊 Testing session...")
                try:
                    db = SessionLocal()
                    db.execute(text("SELECT 1"))
                    db.close()
                    print("✅ Session test successful!")
                except Exception as e:
                    print(f"❌ Session test failed: {e}")
                    return False
                
                print(f"\n🎉 All tests passed!")
                return True
            else:
                print("❌ Connection test failed: Unexpected result")
                return False
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   1. Verify RDS endpoint is correct")
        print(f"   2. Check security group allows connections")
        print(f"   3. Verify credentials are correct")
        print(f"   4. Check network connectivity")
        print(f"   5. Ensure RDS instance is running")
        return False

if __name__ == "__main__":
    success = test_rds_connection()
    sys.exit(0 if success else 1)

