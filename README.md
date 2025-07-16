# YouTube Sentiment Analysis Project

## Overview
This project analyzes YouTube comments for sentiment using machine learning. It includes data extraction, model training, real-time streaming, and dashboard visualization using Docker.

## Project Structure

This project has two main parts:

### 1. `docker-compose` Folder (Docker-based)
- Contains all code and configuration for running the streaming pipeline, model training, and dashboard using Docker Compose.
- Services: Kafka, Spark, Elasticsearch, Kibana, ML trainer.
- Python dependencies for this part are listed in `docker-compose/requirements_model.txt` and installed automatically in the Docker container.

### 2. `Extract and Label` Folder (Python Virtual Environment)
- Contains scripts for extracting and labeling YouTube comments.
- Run these scripts using a Python virtual environment.
- Install dependencies with:
  ```sh
  python -m venv .venv
  source .venv/bin/activate  # On Windows: .venv\Scripts\activate
  pip install -r "requirements.txt"
  ```
- Set your YouTube API key in `.env` or your shell before running extraction scripts.

## Quick Start

### 1. Clone the Repository
```sh
git clone https://github.com/zw-weng/HPDP_Project2.git
cd HPDP_Project2
```

### 2. Set Up Environment Variables
Create a `.env` file in the project root:
```
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### 3. Build and Run with Docker Compose
```sh
cd docker-compose
# Then run:
docker-compose up --build
```
This will start all services: Kafka, Spark, Elasticsearch, Kibana, and the ML trainer.

### 4. Extract YouTube Comments
Edit and run `extract-label/youtube_extractor.py` to fetch comments. Ensure your API key is set in `.env` or your shell.

### 5. Train the Sentiment Model
The ML trainer service will automatically train the model and save results in the `model/` directory.

### 6. View Results
- Model artifacts and visualizations are saved in `model/`
- Access the dashboard at [http://localhost:5601](http://localhost:5601) (Kibana)

## Kibana Dashboard

A pre-configured Kibana dashboard is provided in `export.ndjson`.

To import the dashboard:
1. Open Kibana at [http://localhost:5601](http://localhost:5601)
2. Go to **Stack Management > Saved Objects**
3. Click **Import** and select `export.ndjson`
4. The dashboard will appear under **Dashboards** for immediate use

## Notes
- Data files are stored in `data/`
- Model files and images are stored in `docker-compose/model/`
- The `.gitignore` file ensures sensitive and large files are not committed

## Troubleshooting
- Make sure Docker is installed and running
- Set your YouTube API key before running extraction scripts
- For custom runs, use PowerShell or your preferred shell

---
For more details, see comments in each script or reach out to the project maintainer.
