# Scalable training

# How to start

## Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version

## Set up
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`
