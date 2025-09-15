#!/bin/bash

# Startup script for Render.com deployment using Poetry
echo "Starting Streamlit application with Poetry..."

# Set environment variables
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Increase timeout settings
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

echo "Environment configured:"
echo "PORT: $STREAMLIT_SERVER_PORT"
echo "ADDRESS: $STREAMLIT_SERVER_ADDRESS"
echo "HEADLESS: $STREAMLIT_SERVER_HEADLESS"

# Start the application using Poetry
poetry run streamlit run app.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=$STREAMLIT_SERVER_HEADLESS \
    --server.enableCORS=$STREAMLIT_SERVER_ENABLE_CORS \
    --server.enableXsrfProtection=$STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION \
    --browser.gatherUsageStats=$STREAMLIT_BROWSER_GATHER_USAGE_STATS
