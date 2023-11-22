from typing import Tuple
import argparse
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline
)
from transformers.utils.logging import set_verbosity_error as set_transformers_verbosity_error


def main():
    """Use transformer model to help you annotate textcat data."""
    parser = argparse.ArgumentParser(description='Retrain ')
    parser.add_argument('-m', '--model', help='Specify the model name', required=True)
    parser.add_argument('-o', '--output', help='Specify the output file path', required=True)
    parser.add_argument('-tx', '--trainx', help='Specify the train_x file path', required=True)
    parser.add_argument('-ty', '--trainy', help='Specify the train_y file path', required=True)
    parser.add_argument("-es", "--eval-split", help="Split off a a percentage of the training "
                                                    "examples for evaluation.", required=True)
    parser.add_argument("-e", "--epochs", help="Number of epochs to train for.", required=True)

    args = parser.parse_args()

    train_x = pd.read_csv(args.trainx)[["text"]]
    train_y = pd.read_csv(args.trainy)
    eval_split = float(args.eval_split)
    label_names = train_y.columns.tolist()
    train_examples, valid_examples, train_labels, valid_labels = train_valid_split(train_x, train_y, eval_split)

    id2label = {i: n for i, n in enumerate(label_names)}
    label2id = {n: i for i, n in enumerate(label_names)}

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)


    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    train_dataset = Dataset.from_pandas(train_examples)
    valid_dataset = Dataset.from_pandas(valid_examples)
    tokenized_dataset_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_dataset_valid = valid_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_names), id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=int(args.epochs),
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_train,
        eval_dataset=tokenized_dataset_valid,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_metrics_func(),
    )

    trainer.train()



def build_metrics_func():
    roc_auc = evaluate.load("roc_auc", "multiclass")
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        """Taken from https://huggingface.co/docs/transformers/tasks/sequence_classification#evaluate"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    return compute_metrics


def train_valid_split(train_x, train_y, eval_split) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Make a train validation split. No using sklearn because we don't want to import it."""
    train_examples = train_x.sample(frac=eval_split, random_state=42)
    valid_examples = train_x.drop(train_examples.index)
    train_labels = train_y[train_y.index.isin(train_examples.index)]
    valid_labels = train_y.drop(train_labels.index)

    return train_examples, valid_examples, train_labels, valid_labels


if __name__ == "__main__":
    main()
