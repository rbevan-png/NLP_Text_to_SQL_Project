# NLP_Text_to_SQL_Project

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
