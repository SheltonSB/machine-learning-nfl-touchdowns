# Quick Start: Connect to AWS RDS

## Step 1: Get Your RDS Connection Details

1. Go to [AWS RDS Console](https://us-east-2.console.aws.amazon.com/rds/home?region=us-east-2#databases:)
2. Click on your database instance
3. Note down these details from the **Connectivity & security** tab:
   - **Endpoint**: (e.g., `your-db.xxxxx.us-east-2.rds.amazonaws.com`)
   - **Port**: (5432 for PostgreSQL, 3306 for MySQL)
   - **Database name**: (the database you created)
   - **Master username**: (the username you set)
   - **Master password**: (the password you set - you may need to reset it if you forgot)

## Step 2: Configure Security Group (IMPORTANT!)

Your RDS instance needs to allow connections from your IP address:

1. In the RDS Console, click on your database instance
2. Scroll to **Connectivity & security** section
3. Click on the **VPC security groups** link (it's a clickable link)
4. In the Security Group page:
   - Click **Edit inbound rules**
   - Click **Add rule**
   - **Type**: PostgreSQL (port 5432) or MySQL/Aurora (port 3306)
   - **Source**: 
     - For testing: Select **My IP** (AWS will auto-detect)
     - For production: Use a specific IP or security group
   - Click **Save rules**

## Step 3: Create Your .env File

1. Copy the example file:
   ```bash
   cd enhanced-nfl-platform
   copy env.example .env
   ```

2. Edit `.env` and update the database section with your RDS details:

   **For PostgreSQL RDS:**
   ```bash
   DATABASE_URL=postgresql://YOUR_USERNAME:YOUR_PASSWORD@YOUR_ENDPOINT:5432/YOUR_DATABASE_NAME
   DB_USE_SSL=true
   ```

   **OR use individual settings:**
   ```bash
   DB_HOST=your-db.xxxxx.us-east-2.rds.amazonaws.com
   DB_PORT=5432
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_NAME=your_database_name
   DB_ENGINE=postgresql
   DB_USE_SSL=true
   ```

   **For MySQL RDS:**
   ```bash
   DATABASE_URL=mysql+pymysql://YOUR_USERNAME:YOUR_PASSWORD@YOUR_ENDPOINT:3306/YOUR_DATABASE_NAME
   DB_USE_SSL=true
   ```

   **OR use individual settings:**
   ```bash
   DB_HOST=your-db.xxxxx.us-east-2.rds.amazonaws.com
   DB_PORT=3306
   DB_USER=your_username
   DB_PASSWORD=your_password
   DB_NAME=your_database_name
   DB_ENGINE=mysql
   DB_USE_SSL=true
   ```

## Step 4: Test the Connection

Run the connection test script:

```bash
cd enhanced-nfl-platform
python backend/test_rds_connection.py
```

If successful, you'll see:
```
✅ Connection successful!
✅ Session test successful!
🎉 All tests passed!
```

If it fails, check:
- Security group allows your IP
- Endpoint URL is correct
- Username and password are correct
- Database name exists
- RDS instance is running (not stopped)

## Step 5: Initialize Database Schema (if needed)

If your database is empty, you may need to create tables. Check if your application has migration scripts or database initialization code.

## Troubleshooting

### Connection Timeout
- Verify security group allows your IP address
- Check if RDS instance is in the same region
- Ensure RDS instance status is "Available"

### Authentication Failed
- Double-check username and password
- Verify database name is correct
- Check if password has special characters (may need URL encoding)

### SSL Error
- Ensure `DB_USE_SSL=true` is set
- For production, consider downloading RDS CA certificate (see AWS_RDS_SETUP.md)

## Next Steps

Once connected:
1. Run your application: `python backend/main.py` or `uvicorn backend.main:app --reload`
2. Verify database operations work correctly
3. Monitor RDS metrics in CloudWatch

