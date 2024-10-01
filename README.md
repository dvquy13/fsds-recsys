# fullstackdatascience.com - RecSys

# How to start

## Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- PostgreSQL (optionall for Feast setup in Lesson 2 and to install the poetry deps)
  - For Mac, run: `brew install postgresql`

## Set up
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`
- Start the Jupyterlab notebook: `poetry jupyter lab`
