#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = DS-Flight-Delay-Classification
PYTHON_VERSION = 3.11
MLFLOW_TRACKING_URI = sqlite:///$(CURDIR)/mlflow.db
MLFLOW_ARTIFACT_ROOT = file:///$(CURDIR)/mlartifacts

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	poetry env use $(PYTHON_VERSION)
	poetry lock
	poetry install


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff check
	ruff format --check

## Format source code with ruff
.PHONY: format
format:
	ruff format
	ruff check --fix


## Run tests
.PHONY: test
test:
	poetry run pytest tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	poetry run python -m flight_delay_classification.dataset

## Generate features
.PHONY: features
features: data
	poetry run python -m flight_delay_classification.features

## Train
.PHONY: train
train: features
	poetry run python -m flight_delay_classification.modeling.train

## Evaluate
.PHONY: evaluate
evaluate: train
	poetry run python -m flight_delay_classification.evaluation.evaluate

## Open MLflow UI
.PHONY: mlflow-ui
mlflow-ui:
	poetry run mlflow ui --backend-store-uri "$(MLFLOW_TRACKING_URI)" --default-artifact-root "$(MLFLOW_ARTIFACT_ROOT)"


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
