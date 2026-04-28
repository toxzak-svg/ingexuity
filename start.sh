#!/bin/bash
set -e

# Python dependencies are installed during Docker build (see Dockerfile)

# Start Python Gemma service in background
if [ -f /app/python/gemma_e2b_service.py ]; then
    cd /app/python
    python gemma_e2b_service.py --port 8765 &
    GEMMA_PID=$!
    sleep 5
    for i in {1..10}; do
        if curl -s http://localhost:8765/health > /dev/null 2>&1; then
            echo "Gemma service is ready"
            break
        fi
        echo "Waiting for Gemma to start..."
        sleep 2
    done
fi

# Start Julia server
cd /app
exec julia --project=. -e "using IngExuity; IngExuity.start()"