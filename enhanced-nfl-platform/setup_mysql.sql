-- 🏈 NFL AI Platform - MySQL Database Setup
-- Run this script to create the database and tables

-- Create the database
CREATE DATABASE IF NOT EXISTS nfl_ai;
USE nfl_ai;

-- Create players table
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
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    player_id INT NOT NULL,
    prediction BOOLEAN NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    features JSON,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create rag_queries table
CREATE TABLE IF NOT EXISTS rag_queries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    sources JSON,
    data_freshness VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create system_stats table
CREATE TABLE IF NOT EXISTS system_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert some sample data
INSERT INTO players (player_id, first_name, last_name, position, age, current_team) VALUES
('TB12', 'Tom', 'Brady', 'QB', 46, 'Retired'),
('PM15', 'Patrick', 'Mahomes', 'QB', 28, 'Kansas City Chiefs'),
('AR12', 'Aaron', 'Rodgers', 'QB', 40, 'New York Jets'),
('JA17', 'Josh', 'Allen', 'QB', 27, 'Buffalo Bills'),
('LJ8', 'Lamar', 'Jackson', 'QB', 27, 'Baltimore Ravens');

-- Show tables
SHOW TABLES;

-- Show database info
SELECT 'NFL AI Database created successfully!' as status;
