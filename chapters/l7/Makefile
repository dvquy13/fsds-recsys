.PHONY:
.ONESHELL:

include .env
export

mlflow-up:
	poetry run mlflow server --host 0.0.0.0 --port 5003 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

notebook-up:
	poetry run jupyter lab --port 8888 --host 0.0.0.0

redis-up:
	docker compose -f compose.yml up -d

api-up:
	poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

model-server-up:
	cd model_server
	poetry run bentoml serve service:I2VService --reload
