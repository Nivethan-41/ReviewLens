.PHONY: install test train evaluate serve help

install:
	pip install -r requirements.txt
	python -m ml.config

test:
	pytest tests/

train:
	python -m ml.train

evaluate:
	python -m ml.evaluate

serve:
	uvicorn api.main:app --reload

