
DATASET_PATH=./data/fairy-tails/extended.json
CONFIG_PATH=./configs/default_config.yaml
MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct

HF_HOME=./models


setup:
	uv venv


update: setup
	uv pip compile requirements.in -o requirements.txt --quiet
	uv pip install -r requirements.txt --quiet


graph: update 
	HF_HOME=$(HF_HOME) uv run -m src.prepare --source $(DATASET_PATH) --config $(CONFIG_PATH) --model $(MODEL_NAME)

	
default: update 
	uv run -m src.default_impl

