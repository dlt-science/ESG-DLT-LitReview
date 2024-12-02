import os
import evaluate
import numpy as np
import pandas as pd
import pickle
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset

# from nervaluate import Evaluator
from sklearn.model_selection import KFold, GroupKFold
from datasets import Dataset, DatasetDict, Features, ClassLabel, Sequence, Value
from transformers import Trainer, DataCollatorForTokenClassification, AutoModelForTokenClassification, AdamW, \
    TrainingArguments, AutoConfig


# Function to compute metrics for NER task
def compute_metrics(p):
    # Metrics using the HuggingFace datasets library (https://huggingface.co/docs/evaluate/index)
    metric = evaluate.load("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str,
                        default="./../../models")
    parser.add_argument("--data_dir", type=str,
                        default="./../../data")
    parser.add_argument("--model_name", type=str,
                        default="bert-base-cased")

    args = parser.parse_args()

    print(f"Is CUDA available: {torch.cuda.is_available()}")

    # Clean up the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Type of device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPUs: {torch.cuda.device_count()}")

    # Load the dataset
    data_dir = "./../data"
    model_dir = "./../models"

    dataset = pd.read_parquet("hf://datasets/DLTScienceFoundation/ESG-DLT-NER/data/train-00000-of-00001.parquet")

    print(f"Loading the label_to_id and id_to_label jsons from {args.data_dir}...")
    # Load the label_to_id and id_to_label jsons
    with open(os.path.join(args.data_dir, "label_to_id.json"), "r") as f:
        label_to_id = json.load(f)

    # with open(os.path.join(args.data_dir, "id_to_label.json"), "r") as f:
    #     id_to_label = json.load(f)

    id_to_label = {v: k for k, v in label_to_id.items()}

    # Get the unique labels
    unique_labels = list(set(label_to_id.keys()))

    # Get the label_list
    label_list = list(label_to_id.keys())

    # Set the hyperparameters
    # model_name = "bert-base-cased"
    num_epochs = 20  # for each k-fold
    learning_rate = 5e-5
    # train_batch_size = 32
    # eval_batch_size = 64

    train_batch_size = 16  # Reduced from 32
    eval_batch_size = 32  # Reduced from 64

    max_seq_length = 256
    early_stopping_patience = 3

    # use some warmup steps to increase the learning rate up to a certain point
    # and then use your normal learning rate afterwards
    warmup_steps = 500
    logging_dir = os.path.join(args.model_dir, "logs")

    print(f"Loading the tokenizer for {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    print(f"Defining the training args...")
    # define training args
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=logging_dir,
        learning_rate=float(learning_rate),
        save_total_limit=5,
        do_train=True,
        do_eval=True,
        gradient_accumulation_steps=2,
        # allows to effectively have a larger batch size while using less memory per step
        fp16=False if torch.backends.mps.is_available() else True,  # Enable mixed precision training
        local_rank=0,  # Enable distributed training. It is
        use_mps_device=True if torch.backends.mps.is_available() else False
    )

    print(f"Setting up the config for the model...")
    # Set up config for the model
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=len(unique_labels),
        label2id=label_to_id,
        id2label=id_to_label,
    )

    # Load the model
    print(f"Loading the model {args.model_name}...")
    model = AutoModelForTokenClassification.from_pretrained(args.model_name, config=config)

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Clear the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize the GroupKFold class
    n_splits = 5
    group_kfold = GroupKFold(n_splits=n_splits)

    print(f"Defining the features...")
    # Define your features
    features = Features({
        'input_ids': Sequence(Value('int64')),
        'attention_mask': Sequence(feature=Value(dtype='int64')),
        'labels': Sequence(ClassLabel(num_classes=len(unique_labels), names=unique_labels)),
        'ner_tags': Sequence(ClassLabel(num_classes=len(unique_labels), names=unique_labels)),
        'tokens': Sequence(feature=Value(dtype='string')),
        'attention_mask': Sequence(Value('int64')),
        'paper_name': Value(dtype='string'),
        '__index_level_0__': Value(dtype='int64')
    })

    # Store results from each fold
    results = []
    evaluations = []

    # Clear the GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # Store results from each fold
    results = []
    evaluations = []
    scores = []

    print(f"Starting the training loop...")
    for train_index, val_index in tqdm(group_kfold.split(dataset, groups=dataset['paper_name']), total=n_splits):

        # Split data into training and validation
        train_data, val_data = dataset.iloc[train_index], dataset.iloc[val_index]

        # Create training and validation datasets
        train_dataset = Dataset.from_pandas(train_data, features=features)
        eval_dataset = Dataset.from_pandas(val_data, features=features)

        # Create Trainer instance
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train model
        trainer.train()

        # Make predictions
        predictions, labels, metrics = trainer.predict(eval_dataset)
        predictions = np.argmax(predictions, axis=2)

        # Save the scores for the models:
        scores.append(metrics)

    # Save the model to local storage
    print(f"Saving the model to {model_dir}...")
    model.save_pretrained(model_dir)

    # Save the tokenizer to local storage
    print(f"Saving the tokenizer to {model_dir}...")
    tokenizer.save_pretrained(model_dir)
