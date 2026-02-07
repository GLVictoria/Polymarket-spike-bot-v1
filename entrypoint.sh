#!/bin/bash
# Create config/.env from example.env if it doesn't exist
if [ ! -f /app/config/.env ]; then
    echo "Creating config/.env from example.env..."
    cp /app/example.env /app/config/.env
fi

# Run the main application
exec python -u main.py
