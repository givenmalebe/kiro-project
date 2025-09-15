#!/bin/bash

# Build script for Render.com using Poetry
echo "Building application with Poetry..."

# Install Poetry if not already installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    pip install poetry
fi

# Configure Poetry for production
export POETRY_NO_INTERACTION=1
export POETRY_VENV_IN_PROJECT=1

# Install dependencies
echo "Installing dependencies with Poetry..."
poetry install --only=main

echo "Build completed successfully!"
