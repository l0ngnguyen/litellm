#!/bin/bash

# Start LiteLLM proxy with custom UI for development
# This script uses the LITELLM_UI_PATH environment variable to point to our custom UI build

export LITELLM_UI_PATH="_my_experimental/out"

echo "Starting LiteLLM proxy with custom UI from: $LITELLM_UI_PATH"
echo "Server will be available at: http://localhost:4000"
echo "UI will be available at: http://localhost:4000/ui"
echo ""

litellm --config config_dev.yaml --port 4000 > proxy_dev.log 2>&1 &

PROXY_PID=$!
echo "Proxy started with PID: $PROXY_PID"
echo ""
echo "To view logs: tail -f proxy_dev.log"
echo "To stop proxy: kill $PROXY_PID"
