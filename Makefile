.PHONY: install train-scoreboard-detection run-ocr test clean help pull-models

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = point-by-point-tennis-score-extraction
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
install:
	uv pip install -e .

## Train RF-DETR scoreboard detection model
train-scoreboard-detection:
	$(PYTHON_INTERPRETER) -m src.training.train_scoreboard_detection

## Run end-to-end scoreboard OCR pipeline on a video
run-ocr:
	$(PYTHON_INTERPRETER) -m src.inference.scoreboard_ocr

## Run unit tests
test:
	uv run pytest tests/ -v

## Delete compiled Python files and caches
clean:
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

## Pull scoreboard detection model artifact from S3
pull-models:
	mkdir -p models/scoreboard_detection
	@echo "Set S3_URI to the model.tar.gz path, e.g.:"
	@echo "  aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/scoreboard_detection/<job>/output/model.tar.gz models/scoreboard_detection/"
	@echo "Then: tar xzf models/scoreboard_detection/model.tar.gz -C models/scoreboard_detection/"

#################################################################################
# Self-documenting Makefile                                                     #
#################################################################################

## Show this help message
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_hierarchical=6 \
		'{ \
			printf "%s%*s ", $$1, indent - length($$1), ""; \
			n = split($$2, words, " "); \
			line_length = ncol - indent; \
			for (i = 1; i <= n; i++) { \
				line_length -= length(words[i]) + 1; \
				if (line_length <= 0) { \
					line_length = ncol - indent - length(words[i]) - 1; \
					printf "\n%*s ", indent, " "; \
				} \
				printf "%s ", words[i]; \
			} \
			printf "\n"; \
		}' \
	| more

.DEFAULT_GOAL := help
