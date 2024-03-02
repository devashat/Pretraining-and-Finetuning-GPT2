import pandas as pd
import torch
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import transformers
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config, Trainer, TrainingArguments, pipeline
from huggingface_hub import HfApi, HfFolder
import pickle as pkl
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import datasets
from datasets import load_dataset
import math
from huggingface_hub import HfApi, HfFolder


def pretraining():
    train_dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
    val_dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="validation")
    test_dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="test")


    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token


    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )

    model = GPT2LMHeadModel(config)


    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples["text"],
                                    truncation=True,
                                    max_length=128,
                                    return_overflowing_tokens=False,
                                    return_length=True)

        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()


        return tokenized_inputs


    def format_data(batch):
        all_tokens = sum(batch['input_ids'], [])

        num_full_chunks = len(all_tokens) // 128

        formatted_data = [all_tokens[i * 128: (i + 1) * 128] for i in range(num_full_chunks)]

        return {'input_ids': formatted_data}



    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    formatted_train_dataset = tokenized_train_dataset.map(format_data, batched=True, remove_columns=tokenized_train_dataset.column_names)
    formatted_val_dataset = tokenized_val_dataset.map(format_data, batched=True, remove_columns=tokenized_val_dataset.column_names)
    formatted_test_dataset = tokenized_test_dataset.map(format_data, batched=True, remove_columns=tokenized_test_dataset.column_names)


    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_steps=400,
        save_steps=800,
        warmup_steps=500,
        learning_rate=1e-3,
        fp16=True,
        prediction_loss_only=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_test_dataset,
        tokenizer=tokenizer,
    )


    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    new_model = "244-pretrained"

    trainer.save_model(new_model)

    # HfFolder.save_token('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')
    # api = HfApi()


    # try:
    #     model.push_to_hub(new_model, use_temp_dir=False)
    #     tokenizer.push_to_hub(new_model, use_temp_dir=False)
    #     print("Model and tokenizer successfully pushed to the Hugging Face Hub.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")



# Helpers for finetuning

class CoreRelationsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item
    
    def __len__(self):
        return len(self.labels)



def one_hot_encoding(data, unique_core_relations):
    one_hot_vectors = []
    
    for _, row in data.iterrows():
        one_hot_vector = dict.fromkeys(unique_core_relations, 0)
        for relation in row['Core Relations'].split():
            if relation in one_hot_vector:
                one_hot_vector[relation] = 1
        one_hot_vectors.append(one_hot_vector)
    
    return one_hot_vectors



def convert_to_matrix(vectors):
    return np.array([list(vec.values()) for vec in vectors])


def compute_metrics(pred):
    logits = pred.predictions
    true_labels = pred.label_ids
    
    preds = np.where(torch.sigmoid(torch.tensor(logits)).numpy() > 0.5, 1, 0)
    
    f1 = f1_score(true_labels, preds, average='weighted')
    accuracy = accuracy_score(true_labels, preds)
    
    return {
        'f1': f1,
        'accuracy': accuracy,
    }


# Finetuning function

def finetuning(data_path, save_model_path):

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token


    config = GPT2Config.from_pretrained('devashat/244-pretrained', num_labels=18)
    model = GPT2ForSequenceClassification.from_pretrained('devashat/244-pretrained', config=config)
    model.config.pad_token_id = tokenizer.eos_token_id


    train_data_path = data_path
    train_data = pd.read_csv(train_data_path)

    train_data.drop('IOB Slot tags', axis=1)
    train_data = train_data.dropna(subset=['Core Relations'])
    train_data['Core Relations'].fillna('none', inplace=True)
    train_data['Core Relations'] = train_data['Core Relations'].astype(str)
    train_data.reset_index(drop=True, inplace=True)


    unique_core_relations = set()
    for relations in train_data['Core Relations']:
        unique_core_relations.update(relations.split())
    
    unique_core_relations = sorted(list(unique_core_relations))

    with open('unique_core_relations.pkl', 'wb') as file:
        pkl.dump(unique_core_relations, file)
        
    print("unique_core_relations has been successfully dumped into unique_core_relations.pkl")

    train_split, validation_split = train_test_split(train_data, test_size=0.1, train_size=0.9)

    train_texts = train_split['utterances'].tolist()
    validation_texts = validation_split['utterances'].tolist()

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    validation_encodings = tokenizer(validation_texts, truncation=True, padding=True, max_length=128)

    train_labels = convert_to_matrix(one_hot_encoding(train_split, unique_core_relations))
    validation_labels = convert_to_matrix(one_hot_encoding(validation_split, unique_core_relations))
    
    train_dataset = CoreRelationsDataset(train_encodings, train_labels)
    validation_dataset = CoreRelationsDataset(validation_encodings, validation_labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=1e-4,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )


    trainer.train()

    trainer.save_model(save_model_path)


def train_model(data_path, save_model_path):
    print("Running pretraining")
    pretraining()
    print("Running finetuning...")
    finetuning(data_path, save_model_path)

def test_model(data_path, model_path, output_path):

    test_data = pd.read_csv(data_path)    
    
    with open('unique_core_relations.pkl', 'rb') as file:
        unique_core_relations = pkl.load(file)    
    
    test_texts = test_data['utterances'].tolist()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    config = GPT2Config.from_pretrained(model_path, num_labels=18)
    model = GPT2ForSequenceClassification.from_pretrained(model_path, config=config)
    model.config.pad_token_id = tokenizer.eos_token_id
    
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)
    
    predictions = classifier(test_texts)
    
    threshold = 0.5
    
    predicted_labels = []
    for prediction in predictions:
        # Convert LABEL_X to actual label using unique_core_relations
        labels = [unique_core_relations[int(pred['label'].split('_')[-1])] for pred in prediction if pred['score'] > threshold]
        predicted_labels.append(labels)
    
    predicted_labels_joined = [" ".join(labels) for labels in predicted_labels]
    
    predictions_df = pd.DataFrame({
        "utterances": test_texts,
        "Core Relations": predicted_labels_joined
    })
    
    predictions_csv_path = output_path
    predictions_df.to_csv(predictions_csv_path, index=False)
    print("Generated predictions.csv")


if __name__ == "__main__":
    parser = ArgumentParser("homework CLI")

    parser.add_argument('--train', action="store_true", help="indicator to train model")
    parser.add_argument('--test', action="store_true", help="indicator to test model")

    parser.add_argument('--data', help="path to data file")
    parser.add_argument('--save_model', help="ouput path of trained model")
    parser.add_argument('--model_path', help="path to load trained model from")

    parser.add_argument('--output', help="output path of predictions")

    args = parser.parse_args()

    if args.train:
        train_model(args.data, args.save_model)
    if args.test:
        print("Generating test predictions...")
        test_model(args.data, args.model_path, args.output)
