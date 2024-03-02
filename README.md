# Pretraining and Finetuning GPT2

This project repository contains data and code that pretrains and finetunes GPT2. The model is first pretrained on `WikiText-2`, a dataset with 2 million tokens useful for causal language modeling tasks. The pretrained model is then finetuned on a dataset for relation extraction, from which predictions for the test data are generated. 

## Provided Files

- `train.csv`: The train data file.
- `test.csv`: The test data file.
- `pretraining.ipynb`: The Jupyter notebook used to develop the pretraining code.
- `finetuning.ipynb`: The Jupyter notebook used to develop the finetuning code.
- `test_preds.ipynb`: The Jupyter notebook used to develop the code for generating predictions on the test data
- `main.py`: The python file that contains all the code for pretraining, finetuning, and generating test predictions. Contains argument parsing code.
- `requirements.txt`: Contains a list of all the required packages to setup the environment to run the code for this project.

## Setup
Run the following command in the cli:

```
pip install -r requirements.txt
```

***Note: I do use models uploaded to my HuggingFace account for the finetuning code, if you wish to use a model that you've trained separately, make sure to change the model path manually inside `main.py` in the `finetuning()` function where I configure my model.***

## Example Run Commands

```
python3 main.py --train --data ./train.csv --save_model ./test

python3 main.py --test --data ./test.csv --model_path ./test --output ./predictions.csv
```
