#!/usr/bin/env python3
"""
Test different MySQL credentials to find the right one
"""

import pymysql
import sys

def test_credentials(username, password, host="localhost", port=3306):
    """Test MySQL credentials"""
    try:
        print(f"🔍 Testing: {username}@{host}:{port}")
        connection = pymysql.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            charset='utf8mb4'
        )
        print(f"✅ SUCCESS: {username}@{host}:{port}")
        
        # Test if nfl_ai database exists
        with connection.cursor() as cursor:
            cursor.execute("SHOW DATABASES LIKE 'nfl_ai'")
            result = cursor.fetchone()
            if result:
                print("✅ nfl_ai database exists!")
            else:
                print("❌ nfl_ai database not found")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("🏈 Testing MySQL Credentials")
    print("=" * 40)
    
    # Test different common credentials
    credentials_to_test = [
        ("root", ""),
        ("root", "root"),
        ("root", "password"),
        ("root", "mysql"),
        ("sheltonbumhe2", "NewStrongPassword!123"),
        ("sheltonbumhe2", ""),
        ("sheltonbumhe2", "password"),
        ("mysql", ""),
        ("mysql", "mysql"),
    ]
    
    working_credentials = []
    
    for username, password in credentials_to_test:
        if test_credentials(username, password):
            working_credentials.append((username, password))
        print()
    
    if working_credentials:
        print("🎉 Working credentials found:")
        for username, password in working_credentials:
            print(f"   Username: {username}")
            print(f"   Password: {password}")
            print()
        
        # Test with nfl_ai database
        username, password = working_credentials[0]
        print(f"🔍 Testing with nfl_ai database using {username}...")
        try:
            connection = pymysql.connect(
                host="localhost",
                port=3306,
                user=username,
                password=password,
                database="nfl_ai",
                charset='utf8mb4'
            )
            print("✅ Can connect to nfl_ai database!")
            
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
                print(f"📋 Found {len(tables)} tables:")
                for table in tables:
                    print(f"   - {table[0]}")
            
            connection.close()
            
        except Exception as e:
            print(f"❌ Cannot connect to nfl_ai database: {e}")
    else:
        print("❌ No working credentials found")
        print("\n🔧 Please check:")
        print("1. MySQL is running")
        print("2. Try connecting manually with: mysql -u root -p")
        print("3. Check your MySQL installation")

if __name__ == "__main__":
    main()
