#!/usr/bin/env python3
"""
🏈 NFL AI Platform - MySQL Connection Test
Test and setup MySQL database connection
"""

import pymysql
import sys

# MySQL Configuration
MYSQL_USERNAME = "sheltonbumhe2"
MYSQL_PASSWORD = "NewStrongPassword!123"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "nfl_ai"

def test_connection():
    """Test MySQL connection"""
    try:
        print("🔌 Testing MySQL connection...")
        
        # Connect to MySQL server (without database)
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USERNAME,
            password=MYSQL_PASSWORD,
            charset='utf8mb4'
        )
        
        print("✅ Connected to MySQL server successfully!")
        
        # Create database if it doesn't exist
        with connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DATABASE}")
            print(f"✅ Database '{MYSQL_DATABASE}' created/verified!")
            
            # Use the database
            cursor.execute(f"USE {MYSQL_DATABASE}")
            
            # Create tables
            print("📋 Creating tables...")
            
            # Players table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id VARCHAR(50) UNIQUE NOT NULL,
                    first_name VARCHAR(100) NOT NULL,
                    last_name VARCHAR(100) NOT NULL,
                    position VARCHAR(10),
                    age INT,
                    height INT,
                    weight INT,
                    experience INT,
                    current_team VARCHAR(100),
                    stats JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            # Predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    player_id INT NOT NULL,
                    prediction BOOLEAN NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    model_used VARCHAR(100) NOT NULL,
                    features JSON,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # RAG queries table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_queries (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    model_used VARCHAR(100) NOT NULL,
                    sources JSON,
                    data_freshness VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # System stats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    metric_name VARCHAR(100) NOT NULL,
                    metric_value VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            print("✅ All tables created successfully!")
            
            # Insert sample data
            print("📊 Inserting sample data...")
            
            sample_players = [
                ('TB12', 'Tom', 'Brady', 'QB', 46, 'Retired'),
                ('PM15', 'Patrick', 'Mahomes', 'QB', 28, 'Kansas City Chiefs'),
                ('AR12', 'Aaron', 'Rodgers', 'QB', 40, 'New York Jets'),
                ('JA17', 'Josh', 'Allen', 'QB', 27, 'Buffalo Bills'),
                ('LJ8', 'Lamar', 'Jackson', 'QB', 27, 'Baltimore Ravens')
            ]
            
            for player in sample_players:
                cursor.execute("""
                    INSERT IGNORE INTO players (player_id, first_name, last_name, position, age, current_team)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, player)
            
            print("✅ Sample data inserted!")
            
            # Show tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"📋 Database contains {len(tables)} tables:")
            for table in tables:
                print(f"   - {table[0]}")
        
        connection.commit()
        connection.close()
        
        print("\n🎉 MySQL database setup completed successfully!")
        print(f"📊 Database: {MYSQL_DATABASE}")
        print(f"👤 User: {MYSQL_USERNAME}")
        print(f"🌐 Host: {MYSQL_HOST}:{MYSQL_PORT}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure MySQL is running")
        print("2. Check username and password")
        print("3. Verify MySQL port (default: 3306)")
        print("4. Check if user has CREATE DATABASE privileges")
        return False

if __name__ == "__main__":
    print("🏈 NFL AI Platform - MySQL Setup")
    print("=" * 40)
    
    success = test_connection()
    
    if success:
        print("\n Ready to start the NFL AI platform!")
        print("Run: python mysql_production_app.py")
    else:
        print("\n Please fix the MySQL connection issues first.")
        sys.exit(1)
