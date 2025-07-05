#!/bin/bash

# Script to build and run the ML model training in Docker

echo "🚀 Building ML Model Training Docker Image..."
docker build -f Dockerfile.model -t sentiment-ml-trainer .

echo "🏃 Running ML Model Training..."
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/sentiment_dataset.csv:/app/sentiment_dataset.csv" \
  sentiment-ml-trainer

echo "✅ ML Model Training Complete!"
echo "📊 Check the 'models' directory for:"
echo "   - Best trained model (.pkl file)"
echo "   - Model metadata (JSON file)"
echo "   - Visualization plots (PNG file)"
