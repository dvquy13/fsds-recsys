.PHONY:
.ONESHELL:

include .env
export

mlflow-up:
	docker compose -f compose.ml-platform.yml up -d

notebook-up:
	poetry run jupyter lab --port 8888 --host 0.0.0.0

redis-up:
	docker compose -f compose.kv-store.yml up -d

api-up:
	docker compose -f compose.api.yml up

model-server-up:
	cd model_server
	poetry run bentoml serve service:I2VService --reload

# Create the requirements.txt file to
requirements-txt:
	poetry export --without dev --without-hashes --format=requirements.txt > requirements.txt
	sed -i '' '/^torch/ s/^/# /' requirements.txt  # Commend out torch in requirements.txt to pre-install the CPU version in Docker
	sed -i '' '/^nvidia/ s/^/# /' requirements.txt
