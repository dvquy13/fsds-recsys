.PHONY:
.ONESHELL:

notebook-up:
	poetry run jupyter lab --host 0.0.0.0 --port 8891
