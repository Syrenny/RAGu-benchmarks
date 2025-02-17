
QA_DATASET_PATH=./data/fairy-tales/qa.json
DOCS_DATASET_PATH=./data/fairy-tales/qa.json
OUTPUT_NAME=fairy-tales
CONFIG_PATH=./configs/default_config.yaml
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct

HF_HOME=./models


setup:
	uv venv


update: setup
	uv pip compile requirements.in -o requirements.txt --quiet
	uv pip install -r requirements.txt --quiet


graph: update 
	HF_HOME=$(HF_HOME) uv run -m src.prepare --qa $(QA_DATASET_PATH) --docs $(DOCS_DATASET_PATH) --output $(OUTPUT_NAME) --config $(CONFIG_PATH) --model $(MODEL_NAME)

	

