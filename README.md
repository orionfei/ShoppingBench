# ShoppingBench: A Real-World Intent-Grounded Shopping Benchmark for LLM-based Agents

[![Paper](https://img.shields.io/badge/Paper-arXiv:2412.12345-red)](https://arxiv.org/abs/2508.04266)
<!-- [![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/your-dataset-link) -->

## Overview

ShoppingBench is a novel end-to-end shopping benchmark designed to encompass increasingly challenging levels of grounded intent. Specifically, we propose a scalable framework to simulate user instructions based on various intents derived from sampled real-world products. To facilitate consistent and reliable evaluations, we provide a large-scale shopping sandbox that serves as an interactive simulated environment, incorporating over 2.5 million real-world products.

![](img/intro.png)
## Features

- various real-world shopping intents
- a large-scale shopping sandbox
- Comprehensive evaluation metrics

## Dataset

The ShoppingBench dataset includes:

1. **documents.jsonl.gz**: A compressed file containing product documents (located in `resources/` directory)
   - To decompress: `gunzip -c resources/documents.jsonl.gz > resources/documents.jsonl`
   - Size: ~1.4GB compressed, ~4.8GB uncompressed

2. **Test files**: Located in the `data/` directory
   - `synthesize_product_test.jsonl`: Product Intent test cases
   - `synthesize_shop_test.jsonl`: Shop Intent test cases  
   - `synthesize_voucher_test.jsonl`: Voucher Intent test cases
   - `synthesize_web_simpleqa_test.jsonl`: Web search Intent test cases

## Environment Setup

### prerequist

1. install java (jdk21 recommended)

2. install uv

3. decompress documents.jsonl.gz to get documents.jsonl in resources folder:
   ```bash
   gunzip -c resources/documents.jsonl.gz > resources/documents.jsonl
   ```

4. prepare related KEY
```bash
export OPENAI_API_KEY="your openai api key"
export OPENAI_BASE_URL="your openai base url"
export SERPER_KEY="your serper web search key"
```

### Python Environment Installation and Search Engine Preparation

Run the initialization script to set up the Python environment and start the product search engine:

```bash
./init_env.sh
```

After running the environment setup script, the search engine will be automatically started in the background. 


## Running Inference and Evaluation

To run model inference on test data and evaluate the models for different intents:
   
1. Run the inference scripts (take gpt-4.1 as example):
   
   The script will automatically create necessary directories and validate the data folder structure before running model inference and evaluation.

   ```bash
   ./run.sh product rollout gpt-4.1
   ./run.sh shop rollout gpt-4.1
   ./run.sh voucher rollout gpt-4.1
   ./run.sh web simpleqa_rollout gpt-4.1
   ```

   the inference process will be running in background, you can check the log in logs folder. you can uncomment the specific line to evaluate the inference result or kill the inference process.

2. Run the evaluation scripts (take gpt-4.1 as example):
   
   Please update run.sh by uncommenting the line for run_evaluate.py and commenting out the line for run_rollout, then rerun the scripts.
   ```bash
   ./run.sh product rollout gpt-4.1
   ./run.sh shop rollout gpt-4.1
   ./run.sh voucher rollout gpt-4.1
   ./run.sh web simpleqa_rollout gpt-4.1
   ```
   
## Training SFT and RL Models

### verl Environment Installation

Install the unified verl environment used for both SFT and RL:

```bash
cd src/rl
USE_MEGATRON=0 bash install_vllm_sglang_mcore.sh
uv pip install -e .
```

### Usage
To train the SFT and RL models:

For SFT data preparation:
```bash
python scripts/prepare_verl_shoppingbench_data.py --skip-query
```

For SFT training:
```bash
TRAIN_FILES=dataset/shoppingbench_sft_state_folded/train.parquet \
VAL_FILES=dataset/shoppingbench_sft_state_folded/test.parquet \
./src/rl/run_sft_qwen3_4b_verl_a800.sh
```

For RL training:
```bash
./src/rl/run_grpo.sh
```


## Paper

For more details about ShoppingBench, please refer to our paper
