#!/bin/bash
# Render build script

echo "Starting build process..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p temp
mkdir -p models
mkdir -p data

echo "Build completed successfully!"