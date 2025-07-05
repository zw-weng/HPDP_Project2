# PowerShell Script to build and run the ML model training in Docker

Write-Host "🚀 Building ML Model Training Docker Image..." -ForegroundColor Green
docker build -f Dockerfile.model -t sentiment-ml-trainer .

Write-Host "🏃 Running ML Model Training..." -ForegroundColor Yellow
docker run --rm `
  -v "${PWD}/models:/app/models" `
  -v "${PWD}/sentiment_dataset.csv:/app/sentiment_dataset.csv" `
  sentiment-ml-trainer

Write-Host "✅ ML Model Training Complete!" -ForegroundColor Green
Write-Host "📊 Check the 'models' directory for:" -ForegroundColor Cyan
Write-Host "   - Best trained model (.pkl file)" -ForegroundColor White
Write-Host "   - Model metadata (JSON file)" -ForegroundColor White
Write-Host "   - Visualization plots (PNG file)" -ForegroundColor White
