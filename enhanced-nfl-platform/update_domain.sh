#!/bin/bash

# 🚀 NFL AI Platform - Domain Update Script

echo "🏈 NFL AI Platform - Domain Configuration"
echo "=========================================="

# Get domain from user
read -p "Enter your domain (e.g., my-nfl-platform.com): " DOMAIN

if [ -z "$DOMAIN" ]; then
    echo "❌ No domain provided. Exiting."
    exit 1
fi

echo "🔄 Updating domain to: https://$DOMAIN"

# Update all HTML files
echo "📝 Updating frontend files..."

# Update production.html
sed -i.bak "s/your-nfl-ai-platform.com/$DOMAIN/g" frontend/production.html
echo "✅ Updated production.html"

# Update comprehensive.html
sed -i.bak "s/your-nfl-ai-platform.com/$DOMAIN/g" frontend/comprehensive.html
echo "✅ Updated comprehensive.html"

# Update simple.html
sed -i.bak "s/your-nfl-ai-platform.com/$DOMAIN/g" frontend/simple.html
echo "✅ Updated simple.html"

echo ""
echo "🎉 Domain updated successfully!"
echo ""
echo "📋 Your platform will be available at:"
echo "   Backend API: https://$DOMAIN"
echo "   Frontend: https://$DOMAIN/production.html"
echo "   API Docs: https://$DOMAIN/docs"
echo ""
echo "🚀 Ready for deployment!"
