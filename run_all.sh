#!/usr/bin/env bash
set -euo pipefail

echo "=== Building backend image ==="
docker compose build backend

echo "=== Starting services (LocalStack, Postgres, Backend) ==="
docker compose up -d

echo "Waiting for Postgres to be ready..."
sleep 15

echo "Waiting for backend to be ready..."
sleep 10

BACKEND_URL="http://localhost:8000"

echo "=== Health check ==="
curl -s "${BACKEND_URL}/health" || {
  echo "Backend health check failed"
  exit 1
}

echo "=== Triggering ETL job ==="
curl -s -X POST "${BACKEND_URL}/etl/run" -H "Content-Type: application/json"

echo
echo "=== Open the dashboard at: ${BACKEND_URL}/ ==="


