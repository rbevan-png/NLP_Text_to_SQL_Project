# NLP_Text_to_SQL_Project
## How to get Dataset

1. Download the Spider dataset:
   [Spider Dataset on Kaggle](https://www.kaggle.com/datasets/jeromeblanchet/yale-universitys-spider-10-nlp-dataset?resource=download)
2. Upload the dataset files to your Colab environment.

# Text-to-SQL Evaluation using GPT Models

This Colab notebook evaluates large language models (LLMs) on the Spider dataset using both zero-shot prompting and fine-tuned approaches. The goal is to generate SQL queries from natural language questions and assess model performance using both structural and semantic metrics.

## Features

- Supports evaluation of:
  - Zero-shot GPT-4o
  - Fine-tuned GPT-3.5-Turbo (via OpenAI API)
- Evaluates on the full Spider development set (1,034 examples)
- Computes:
  - Exact Match Accuracy – measures if the generated SQL exactly matches the reference
  - Execution Accuracy – measures if the generated SQL produces the correct result when executed
- Saves results in CSV format for analysis
- Modular design for easily switching between models

## Cost Estimate

Evaluating the full Spider dev set (1,034 queries) will typically cost:
- **$1.50–$2.00** for GPT-4o (zero-shot)
- **$0.80–$1.20** for a fine-tuned GPT-3.5 model

If you run both models in sequence, expect a total cost between **$2.50 and $3.20**, depending on prompt and response lengths. These estimates are based on OpenAI’s April 2024 pricing.

## File Outputs

- `spider_<model>_results.csv`: Contains per-example predictions and accuracy flags
- `spider_<model>_summary.csv`: Contains summary accuracy scores

## How to Use

1. Upload the Spider dataset files to your Colab environment.
2. Set your OpenAI API key in the notebook.
3. Choose the model to evaluate by setting the `USE_MODEL` variable:
   - `"gpt-4o"` for zero-shot evaluation
   - Your fine-tuned model ID (e.g., `"ft:gpt-3.5-turbo-0125:your-model-id"`) for fine-tuned evaluation
4. Run the notebook cells to generate and evaluate SQL queries.
5. Review the accuracy metrics and inspect outputs in the saved CSV files.

## Requirements

- Python 3.x
- OpenAI Python SDK
- pandas
- sqlite3

# Fine-Tuning T5-Small on Spider with Multistage Mounting

This notebook implements a multistage fine-tuning pipeline for the T5-Small language model using a series of progressively more complex text-to-SQL datasets. The goal is to train a lightweight semantic parser capable of translating natural language questions into SQL queries for the Spider benchmark.

## Overview

Inspired by industry workflows, this notebook adopts a mounting strategy, where successive fine-tuning stages build on previous checkpoints. Early stages teach basic SQL structure, while later stages specialize the model for the Spider dataset.

## Fine-Tuning Stages

1. **SQLCreateContext** – A clean dataset with explicit schema prompts
2. **Text-to-SQL v1** – Introduces more natural and varied question structures
3. **KnowSQL** – Adds external knowledge and diverse linguistic styles
4. **Spider (train_spider)** – Final benchmark-aligned fine-tuning

Datasets are interleaved using HuggingFace's `interleave_datasets` utility. After each stage, a new checkpoint is saved and used as the starting point for the next stage.

## Key Features

- Uses the HuggingFace Transformers + Datasets library
- Schema prompts are prepended to every input
- Intermediate checkpoints saved per stage
- Designed to minimize catastrophic forgetting
- Final model evaluated on the Spider dev set

## Evaluation

The final T5-small model was evaluated on 1,034 Spider dev set examples:

- **Exact Match Accuracy**: 1.55%
- **Execution Accuracy**: Not reliably measurable due to frequent schema or syntax errors

Common failure cases included hallucinated tables and malformed tokens, suggesting overfitting and schema drift due to limited model capacity.

## Limitations

- T5-small’s limited size restricts multi-dataset retention
- Schema inconsistencies across datasets were not normalized
- Interleaving may have biased the model toward early training stages

## Refinement Directions

Future improvements may include:

- Schema linking normalization across datasets
- Layer freezing between mounting stages
- Curriculum-based dataset ordering
- Schema validation during generation

## Dataset Sources

- SQLCreateContext: [arXiv:2305.12329](https://arxiv.org/abs/2305.12329)
- Text-to-SQL v1: [arXiv:1709.00103](https://arxiv.org/abs/1709.00103)
- KnowSQL: [arXiv:2010.02850](https://arxiv.org/abs/2010.02850)
- Spider: [arXiv:1809.08887](https://arxiv.org/abs/1809.08887)

## Requirements

- Python 3.x
- Transformers >= 4.30
- Datasets >= 2.x
- PyTorch
- HuggingFace `accelerate` (optional for fast training)

## Usage

1. Upload the datasets or load via HuggingFace datasets
2. Set output paths and checkpoint save locations
3. Run the notebook cell-by-cell to perform staged fine-tuning
4. Evaluate using your custom script or HuggingFace `evaluate` utilities


