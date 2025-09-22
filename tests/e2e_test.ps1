#!/bin/bash

set -e

echo "Running KLARNET E2E tests..."

# Start KLARNET in background
./target/release/klarnet &
KLARNET_PID=$!

# Wait for startup
sleep 5

# Test health check
curl -f http://localhost:3000/health || exit 1

# Test STT endpoint with sample audio
if [ -f "tests/fixtures/sample.wav" ]; then
    curl -X POST http://localhost:3000/stt/file \
        -H "Content-Type: audio/wav" \
        --data-binary @tests/fixtures/sample.wav
fi

# Test NLU endpoint
curl -X POST http://localhost:3000/nlu/interpret \
    -H "Content-Type: application/json" \
    -d '{"text": "джарвис включи свет"}'

# Test metrics endpoint
curl -f http://localhost:9090/metrics || exit 1

# Cleanup
kill $KLARNET_PID

echo "E2E tests passed!"