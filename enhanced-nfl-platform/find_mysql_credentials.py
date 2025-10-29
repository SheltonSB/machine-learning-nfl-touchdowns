#!/usr/bin/env python3
"""
Find the correct MySQL credentials by testing common combinations
"""

import subprocess
import sys

def test_mysql_connection(username, password):
    """Test MySQL connection using command line"""
    try:
        cmd = f"mysql -u {username} -p'{password}' -e 'SELECT 1;'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {username} with password '{password}'")
            return True
        else:
            print(f"❌ FAILED: {username} with password '{password}' - {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("🔍 Finding MySQL Credentials")
    print("=" * 40)
    
    # Common password combinations to try
    passwords_to_try = [
        "",  # No password
        "root",
        "password", 
        "mysql",
        "NewStrongPassword!123",
        "admin",
        "123456",
        "toor",
        "root123"
    ]
    
    usernames_to_try = [
        "root",
        "sheltonbumhe2", 
        "mysql",
        "admin",
        "user"
    ]
    
    working_combinations = []
    
    for username in usernames_to_try:
        for password in passwords_to_try:
            if test_mysql_connection(username, password):
                working_combinations.append((username, password))
    
    if working_combinations:
        print(f"\n🎉 Found {len(working_combinations)} working combinations:")
        for username, password in working_combinations:
            print(f"   Username: {username}")
            print(f"   Password: '{password}'")
        
        # Test with nfl_ai database
        username, password = working_combinations[0]
        print(f"\n🔍 Testing nfl_ai database access with {username}...")
        
        cmd = f"mysql -u {username} -p'{password}' -e 'USE nfl_ai; SHOW TABLES;'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Can access nfl_ai database!")
            print("Tables found:")
            print(result.stdout)
        else:
            print("❌ Cannot access nfl_ai database")
            print("Error:", result.stderr.strip())
    else:
        print("\n❌ No working credentials found")
        print("\n🔧 Please try:")
        print("1. Check if MySQL is running: brew services start mysql")
        print("2. Try connecting manually: mysql -u root -p")
        print("3. Check your MySQL installation")

if __name__ == "__main__":
    main()
