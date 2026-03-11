#!/bin/bash
#SBATCH --job-name=mirage
#SBATCH --account=bfox-delta-gpu
#SBATCH --partition=gpuH200x8

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=3
#SBATCH --mem=125G
#SBATCH --time=05:00:00

#SBATCH --output=mirage-%j.out
#SBATCH --error=mirage-%j.err

echo "Running on node:"
hostname
nvidia-smi

# Go to project directory
cd /u/ssingh38/MetaMirage

# Activate environment
source ~/.bashrc
conda activate mirage

echo "Starting SGLang servers..."

# GPU 0
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
  --host 127.0.0.1 \
  --port 11434 \
  --tensor-parallel-size 1 \
  --tool-call-parser llama3 \
  --enable-multimodal \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --max-total-tokens 32768 &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
  --host 127.0.0.1 \
  --port 11435 \
  --tensor-parallel-size 1 \
  --tool-call-parser llama3 \
  --enable-multimodal \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --max-total-tokens 32768 &

# GPU 2
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
  --host 127.0.0.1 \
  --port 11436 \
  --tensor-parallel-size 1 \
  --tool-call-parser llama3 \
  --enable-multimodal \
  --trust-remote-code \
  --mem-fraction-static 0.9 \
  --max-total-tokens 32768 &

echo "Waiting for servers to start..."
sleep 40

echo "Running inference..."

cd /u/ssingh38/MetaMirage/Inference
bash bash_generate.sh >> output3.txt

wait
