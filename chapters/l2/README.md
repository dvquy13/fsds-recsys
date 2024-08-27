# Set up

- Install Docker

# Test Feature Store
- At root: `export $(cat .env | grep -v "^#") && cd data_prep && make up`
- Run [003-l2-feature-store-pipeline.ipynb](../../notebooks/003-l2-feature-store-pipeline.ipynb) to populate the raw data into PostgreSQL
- Test feature store flow: `cd fsds_feast/feature_repo && poetry run python test_workflow.py`
- Tear down after testing: `poetry run feast teardown`
