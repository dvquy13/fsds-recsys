# Introduction to data processing pipeline with DBT and Feature Store

In this module we will explore conceptually how data is ETL from source systems into ML-specific data storeage.

We first push the interaction data into PostgreSQL to simulate transaction data.
Then we would create model using the dbt framework to extract and transform the source data into features used in ML model.
Finally we materialize the features from offline store into online store where it's stored in a Key-value manner which is optimized for inference use case.

# Prerequisite
- Poetry 1.8.3
- Miniconda or alternatives that can create new Python environment with a specified Python version
- Docker
- PostgreSQL
  - For Mac, run: `brew install postgresql`

# Set up
- Run `export ROOT_DIR=$(pwd)` for easier nagivation
- Create a new `.env` file based on `.env.example` and populate the variables there
- Create a new Python 3.11.9 environment: `conda create --prefix .venv python=3.11.9`
- Make sure Poetry use the new Python 3.11.9 environment: `poetry env use .venv/bin/python`
- Install Python dependencies with Poetry: `poetry install`

# EDA
- Start the Jupyterlab notebook: `poetry run jupyter lab`
- Experiment with the notebook [001-eda.ipynb](notebooks/001-eda.ipynb)

# Simulate transaction data
- Run `cd data_prep && make up` to start PostgreSQL service
- Execute the notebook [002-simulate-oltp.ipynb](notebooks/002-simulate-oltp.ipynb) to populate the raw data into PostgreSQL

# Build feature table
```shell
cd $ROOT_DIR
export $(cat .env | grep -v "^#")
cd data_prep/dbt/feast
# Specify credential for dbt to connect to PostgreSQL
cat <<EOF > profiles.yml
feast:
  outputs:
    dev:
      dbname: $POSTGRES_DB
      host: localhost
      pass: $POSTGRES_PASSWORD
      port: 5432
      schema: public
      threads: 1
      type: postgres
      user: $POSTGRES_USER
  target: dev
EOF
poetry run dbt build --models marts.amz_review_rating
```

# Test Feature Store
- Test feature store flow: `cd $ROOT_DIR/data_prep/fsds_feast/feature_repo && poetry run python test_workflow.py`
- Tear down after testing: `poetry run feast teardown`
