#!/bin/bash
#SBATCH --job-name=ragu-benchmarks
#SBATCH --error="/userspace/sii/logs/ragu-benchmarks.err"
#SBATCH --output="/userspace/sii/logs/ragu-benchmarks.log"
#SBATCH --partition=a100

# Variables for log and cache paths
project_root="/userspace/sii/RAGu-benchmarks"
log_file="/userspace/sii/logs/ragu-benchmarks.log"
err_file="/userspace/sii/logs/ragu-benchmarks.err"
cache_dir="/userspace/sii/cache"
logs_dir="/userspace/sii/logs"
miniconda_dir="/userspace/sii/miniconda3"
conda_env="ragu-benchmarks"


# Variables for configuring run
dataset_path="${project_root}/data/fairy-tails/extended.json"
config_path="${project_root}/configs/default_config.yaml"
model_name="Qwen/Qwen2.5-0.5B-Instruct"

echo "=== Logging variable values ===" >> "${log_file}"
echo "Project Root: ${project_root}" >> "${log_file}"
echo "Log file path: ${log_file}" >> "${log_file}"
echo "Error file path: ${err_file}" >> "${log_file}"
echo "Cache directory: ${cache_dir}" >> "${log_file}"
echo "Logs directory: ${logs_dir}" >> "${log_file}"
echo "Miniconda directory: ${miniconda_dir}" >> "${log_file}"
echo "Conda environment: ${conda_env}" >> "${log_file}"
echo "Dataset path: ${dataset_path}" >> "${log_file}"
echo "Config path: ${config_path}" >> "${log_file}"
echo "Model name: ${model_name}" >> "${log_file}"
echo "=== End of variable log ===" >> "${log_file}"

# Ensure directories for logs and cache exist
mkdir -p "${logs_dir}"
mkdir -p "${cache_dir}"

# Log script start time
echo "Script started at $(date)" >> "${log_file}"
echo "Script started at $(date)" >> "${err_file}"

# Load conda environment
echo "Initializing conda" >> "${log_file}"
source "${miniconda_dir}/etc/profile.d/conda.sh"
conda activate "${conda_env}"

# Set environment variables
echo "Setting environment variables" >> "${log_file}"
export PIP_CACHE_DIR="${cache_dir}"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"
export HF_HOME="${cache_dir}/huggingface"
export PATH="/usr/local/cuda-11/bin:${PATH}"
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export ANONYMIZED_TELEMETRY=False
export OUTLINES_CACHE_DIR="${cache_dir}/outlines"

echo "===================================================" >> "${log_file}"
echo "CUDA details" >> "${log_file}"
echo "===================================================" >> "${log_file}"
ls -d /usr/local/cuda* >> "${log_file}"
nvidia-smi -L >> "${log_file}"
nvcc -V >> "${log_file}"
python -V >> "${log_file}"
echo "===================================================" >> "${log_file}"

# Execute the task
echo "=== Executing task ===" >> "${log_file}"
python3 -m src.prepare --source $dataset_path --config $config_path --model $model_name
echo "===================================================" >> "${log_file}"

# Log script end time
echo "Script ended at $(date)" >> "${log_file}"
echo "Script ended at $(date)" >> "${err_file}"
echo "===================================================" >> "${log_file}"
echo "===================================================" >> "${err_file}"
