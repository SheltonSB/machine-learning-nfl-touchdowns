# AWS RDS Setup Guide

This guide explains how to configure and use AWS RDS (Relational Database Service) with the NFL AI/ML Platform.

## Overview

The platform supports AWS RDS for both **PostgreSQL** and **MySQL** databases. Using AWS RDS provides:
- Managed database service (no server management)
- Automatic backups and point-in-time recovery
- High availability and scalability
- Security features (VPC, encryption, SSL)
- Monitoring and performance insights

## Prerequisites

1. AWS Account with appropriate permissions
2. AWS CLI configured (optional, but recommended)
3. Database credentials from your RDS instance
4. RDS endpoint URL from AWS Console

## Setting Up AWS RDS

### Step 1: Create an RDS Instance

1. **Log in to AWS Console** and navigate to RDS
2. **Create database**:
   - Choose either **PostgreSQL** or **MySQL**
   - Select engine version (recommended: PostgreSQL 15+ or MySQL 8.0+)
   - Choose instance class based on your needs
   - Configure storage (minimum 20GB recommended)
   - Set master username and password
   - Choose VPC and security group (ensure your application can access it)

3. **Configure security**:
   - Enable encryption at rest (recommended)
   - Enable SSL/TLS connections
   - Configure security group to allow connections from your application
   - Note the endpoint URL (format: `your-db-name.xxxxx.region.rds.amazonaws.com`)

### Step 2: Configure Security Group

1. Open your RDS security group in AWS Console
2. Add inbound rule:
   - Type: PostgreSQL (port 5432) or MySQL/Aurora (port 3306)
   - Source: Your application's IP or security group
   - For production, restrict to specific IPs or VPC

### Step 3: Configure Environment Variables

Create or update your `.env` file with RDS credentials:

#### Option 1: Using DATABASE_URL (Recommended)

```bash
# For PostgreSQL RDS
DATABASE_URL=postgresql://your_username:your_password@your-rds-endpoint.region.rds.amazonaws.com:5432/nfl_platform

# For MySQL RDS
DATABASE_URL=mysql+pymysql://your_username:your_password@your-rds-endpoint.region.rds.amazonaws.com:3306/nfl_ai
```

**Note**: If your password contains special characters, they will be URL-encoded automatically.

#### Option 2: Using Individual Settings (Alternative)

```bash
# Database connection settings
DB_HOST=your-rds-endpoint.region.rds.amazonaws.com
DB_PORT=5432  # 5432 for PostgreSQL, 3306 for MySQL
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=nfl_platform
DB_ENGINE=postgresql  # or "mysql" for MySQL
DB_USE_SSL=true  # Enable SSL for secure connections

# Optional: RDS-specific settings
RDS_ENDPOINT=your-rds-endpoint.region.rds.amazonaws.com
RDS_CA_CERT_PATH=/path/to/rds-ca-cert.pem  # Optional: for custom CA certificate
```

#### Option 3: Using Legacy MySQL Settings

```bash
MYSQL_HOST=your-rds-endpoint.region.rds.amazonaws.com
MYSQL_PORT=3306
MYSQL_USER=your_username
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=nfl_ai
```

### Step 4: Download RDS CA Certificate (Optional but Recommended)

For enhanced security, download the RDS CA certificate:

1. Download from AWS: https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.SSL.html
2. Save to your project: `certs/rds-ca-cert.pem`
3. Update `.env`:
   ```bash
   RDS_CA_CERT_PATH=./certs/rds-ca-cert.pem
   ```

### Step 5: Initialize Database Schema

If you need to create tables, you can:

1. **Use existing migration scripts** (if available)
2. **Connect directly** and run SQL:
   ```bash
   # For PostgreSQL
   psql -h your-rds-endpoint.region.rds.amazonaws.com -U your_username -d nfl_platform

   # For MySQL
   mysql -h your-rds-endpoint.region.rds.amazonaws.com -u your_username -p nfl_ai
   ```

## Configuration Examples

### PostgreSQL RDS Example

```bash
# .env file
DATABASE_URL=postgresql://nfl_user:SecurePassword123!@nfl-db.xxxxx.us-east-1.rds.amazonaws.com:5432/nfl_platform
DB_USE_SSL=true
```

### MySQL RDS Example

```bash
# .env file
DB_HOST=nfl-db.xxxxx.us-east-1.rds.amazonaws.com
DB_PORT=3306
DB_USER=nfl_user
DB_PASSWORD=SecurePassword123!
DB_NAME=nfl_ai
DB_ENGINE=mysql
DB_USE_SSL=true
```

### Using RDS_ENDPOINT

```bash
# .env file
RDS_ENDPOINT=nfl-db.xxxxx.us-east-1.rds.amazonaws.com
DB_USER=nfl_user
DB_PASSWORD=SecurePassword123!
DB_NAME=nfl_platform
DB_PORT=5432
DB_ENGINE=postgresql
DB_USE_SSL=true
```

## Testing the Connection

### Test from Python

Create a test script:

```python
from app.core.database import engine
from sqlalchemy import text

try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("✅ Successfully connected to AWS RDS!")
        print(f"Database: {conn.dialect.name}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Test from Command Line

```bash
# For PostgreSQL
psql -h your-rds-endpoint.region.rds.amazonaws.com -U your_username -d nfl_platform

# For MySQL
mysql -h your-rds-endpoint.region.rds.amazonaws.com -u your_username -p nfl_ai
```

## Security Best Practices

1. **Use SSL/TLS**: Always enable `DB_USE_SSL=true` for RDS connections
2. **Secure Passwords**: Use strong, unique passwords for database users
3. **IAM Authentication**: Consider using IAM database authentication (advanced)
4. **VPC Security**: Ensure RDS is in a private subnet with proper security groups
5. **Encryption**: Enable encryption at rest and in transit
6. **Backup**: Enable automated backups with appropriate retention period
7. **Monitoring**: Enable CloudWatch monitoring and alerts

## Connection Pooling

The application automatically configures connection pooling for RDS:
- **Pool Size**: 10 connections (configurable)
- **Max Overflow**: 20 additional connections
- **Pool Recycle**: 1 hour (prevents stale connections)
- **Pool Pre-ping**: Enabled (verifies connections before use)

## Troubleshooting

### Connection Timeout

**Problem**: Cannot connect to RDS instance

**Solutions**:
1. Check security group rules (ensure port 5432/3306 is open)
2. Verify RDS endpoint URL is correct
3. Check if RDS instance is in the same VPC as your application
4. Verify credentials are correct
5. Check network connectivity (firewall, VPN, etc.)

### SSL Connection Error

**Problem**: SSL connection fails

**Solutions**:
1. Ensure `DB_USE_SSL=true` is set
2. Verify RDS instance has SSL enabled
3. Download and configure RDS CA certificate
4. Check if certificate path is correct

### Authentication Failed

**Problem**: Authentication fails

**Solutions**:
1. Verify username and password are correct
2. Check if user has necessary permissions
3. Ensure password doesn't contain unescaped special characters
4. Verify database name is correct

### Connection Pool Exhausted

**Problem**: Too many connections

**Solutions**:
1. Reduce `pool_size` in database configuration
2. Increase RDS instance `max_connections` parameter
3. Check for connection leaks in application code
4. Monitor connection usage in CloudWatch

## Cost Optimization

1. **Use Reserved Instances**: For long-term usage, consider reserved instances (up to 75% savings)
2. **Right-size Instances**: Start with smaller instances and scale up as needed
3. **Stop When Not in Use**: For development, stop RDS instances when not in use
4. **Use Multi-AZ Only When Needed**: Multi-AZ doubles the cost, use only for production
5. **Enable Automated Backups**: But adjust retention period based on needs

## Monitoring

1. **CloudWatch Metrics**: Monitor CPU, memory, connections, and storage
2. **Performance Insights**: Enable Performance Insights for query analysis
3. **Enhanced Monitoring**: Enable for more detailed metrics
4. **Alarms**: Set up CloudWatch alarms for critical metrics

## Migration from Local Database

1. **Export data** from local database:
   ```bash
   # PostgreSQL
   pg_dump -U username -d nfl_platform > backup.sql

   # MySQL
   mysqldump -u username -p nfl_ai > backup.sql
   ```

2. **Import to RDS**:
   ```bash
   # PostgreSQL
   psql -h rds-endpoint -U username -d nfl_platform < backup.sql

   # MySQL
   mysql -h rds-endpoint -u username -p nfl_ai < backup.sql
   ```

3. **Update environment variables** to point to RDS
4. **Test connection** and verify data
5. **Update application** configuration

## Additional Resources

- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)
- [RDS PostgreSQL Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_PostgreSQL.html)
- [RDS MySQL Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_MySQL.html)
- [RDS SSL/TLS Configuration](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.SSL.html)
- [RDS Security Best Practices](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.html)

## Support

For issues or questions:
1. Check AWS RDS logs in CloudWatch
2. Review application logs
3. Verify configuration in `.env` file
4. Test connection using command-line tools
5. Check AWS RDS documentation

## Quick Start Checklist

- [ ] Create RDS instance (PostgreSQL or MySQL)
- [ ] Configure security group
- [ ] Set up environment variables in `.env`
- [ ] Download RDS CA certificate (optional)
- [ ] Test connection
- [ ] Initialize database schema
- [ ] Enable SSL/TLS
- [ ] Configure backups
- [ ] Set up monitoring and alerts
- [ ] Test application with RDS

