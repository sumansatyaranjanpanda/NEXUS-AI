#!/bin/bash
# Usage: bash deploy.sh
# Run this on your Vultr server to deploy or update NEXUS.
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NEXUS Deploy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Pull latest code from GitHub
echo "[1/4] Pulling latest code..."
git pull origin main

# 2. Build the React frontend (output goes to frontend/dist/)
echo "[2/4] Building frontend..."
cd frontend
npm ci --silent
npm run build
cd ..

# 3. Rebuild Docker images and restart all services
echo "[3/4] Rebuilding and restarting services..."
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# 4. Show running containers
echo "[4/4] Done!"
echo ""
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  App is live at http://$(curl -s ifconfig.me)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
