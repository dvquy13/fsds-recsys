# Scalable training

# How to start

## Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker

## Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# Run
- Run the notebook 000, 001, 010, 011, 012 to get the precomputed batch recommendations as jsonl file
- Run `docker compose up -d` to start Redis which is our key-value store
- Run notebook 013 to store the recommendations into Redis
- Run notebook 014 to store the user's item sequence and popular recommendations into Redis