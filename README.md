# AP-Bots: Adaptive, Personalized Chatbots

## Table of Contents

- [Personalized Text Generation with Contrastive Examples](#personalized-text-generation-with-contrastive-examples)
  - [Description](#pers-description)  
  - [Command-Line Arguments](#command-line-arguments)
  - [Evaluation](#evaluation)
- [Requirements](#requirements)
  
## Requirements

Ensure you have the following installed:

- **Python**: `>= 3.11.3`
- **CUDA**: `>= 12`

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Personalized Text Generation with Contrastive Examples

Run an experiment with the following command:

```bash
python run_exp.py -d lamp_5_dev_user -k 5 -f WF DPF -r contriever -ce 3
```
### Command-Line Arguments

| Argument                         | Data Type    | Description                                                                                                                | Default             |
|----------------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------|---------------------|
| `-d`           | `str`        | Name of the dataset. Supported formats:                                                                                     | `amazon_Grocery_and_Gourmet_Food_2018`|
|                              |              | - **LaMP**: `lamp_{dataset_num}_{data_split}_{user/time_split}` (e.g., `lamp_4_test_user`, `lamp_5_dev_time`).              |                     |
|                              |              | - **Amazon**: `amazon_{category}_{year}` (e.g., `amazon_All_Beauty_2018`).                                                |                     |
| `-k`             | `int`        | Number of documents to retrieve for RAG. If `None`, inferred from user profiles.                                           | `None`              |
| `-f`              | `str`     | Space-separated list of features to use (WF DPF SP).                                                                        | `None`              |
| `-r`            | `str`     | Retriever model to use (`contriever`, `dpr`, or any model from [SentenceTransformers](https://www.sbert.net/)).             | `contriever`        |
| `-ce` | `int`     | Number of contrastive users to include. If `None`, this method is not applied.                                             | `None`              |
| `-rs`| `int`        | Number of times the instruction is repeated in the prompt.                                                                 | `1`                 |
|`-ob`  | `bool` | Bool for creating a batch job with the [OpenAI client](https://platform.openai.com/docs/guides/batch/getting-started?lang=node), works only with GPT-based models. | `False`

### Evaluation

Evaluate a dataset with the following command:

```bash
python -m evaluation.eval -d dataset_name
```

This command evaluates all results in the preds folder for the specified dataset and generates a CSV file in the evaluation directory.

