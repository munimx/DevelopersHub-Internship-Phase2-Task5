from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from utils import (
    ensure_dir,
    load_config,
    load_dataset_csv,
    save_csv,
    save_json,
    set_seed,
    setup_logging,
)


def build_label_maps(labels: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, text_col: str, max_length: int) -> Dataset:
    def tokenize(batch: Dict[str, List[str]]) -> Dict[str, List[int]]:
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True)


def compute_metrics(
    eval_pred: Tuple[np.ndarray, np.ndarray], labels: List[str]
) -> Dict[str, float]:
    logits, targets = eval_pred
    preds = np.argmax(logits, axis=1)
    y_true = [labels[idx] for idx in targets]
    y_pred = [labels[idx] for idx in preds]
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def plot_training_curves(log_history: List[Dict[str, float]], output_path: str) -> None:
    train_points = [
        (entry["epoch"], entry["loss"])
        for entry in log_history
        if "loss" in entry and "eval_loss" not in entry
    ]
    eval_points = [
        (entry["epoch"], entry["eval_loss"])
        for entry in log_history
        if "eval_loss" in entry
    ]
    if not train_points and not eval_points:
        return

    plt.figure(figsize=(8, 5))
    if train_points:
        epochs, losses = zip(*train_points)
        plt.plot(epochs, losses, label="train_loss")
    if eval_points:
        epochs, losses = zip(*eval_points)
        plt.plot(epochs, losses, label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_predictions_df(
    df: pd.DataFrame,
    logits: np.ndarray,
    labels: List[str],
    id_col: str,
    text_col: str,
    label_col: str,
    top_k: int,
) -> pd.DataFrame:
    records = []
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    top_indices = np.argsort(probs, axis=1)[:, ::-1][:, :top_k]

    for row, idxs, row_probs in zip(df.itertuples(index=False), top_indices, probs):
        record = {
            "ticket_id": getattr(row, id_col),
            "ticket_text": getattr(row, text_col),
            "true_label": getattr(row, label_col),
        }
        for rank, label_idx in enumerate(idxs, start=1):
            record[f"pred_{rank}"] = labels[label_idx]
            record[f"score_{rank}"] = float(row_probs[label_idx])
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a ticket classifier.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("finetune")
    config = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_seed(seed)

    data_cfg = config["data"]
    df = load_dataset_csv(
        data_cfg["processed_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
        data_cfg["id_col"],
    )
    labels = list(config["labels"])
    label_to_id, id_to_label = build_label_maps(labels)
    if set(df[data_cfg["label_col"]]) - set(labels):
        raise ValueError("Dataset contains labels not in config.")

    df = df.copy()
    df["label_id"] = df[data_cfg["label_col"]].map(label_to_id)
    train_cfg = config["training"]
    temp_size = float(train_cfg["test_size"]) + float(train_cfg["val_size"])
    if temp_size <= 0 or temp_size >= 1:
        raise ValueError("test_size + val_size must be between 0 and 1.")

    stratify_labels = df["label_id"] if df["label_id"].value_counts().min() >= 2 else None
    if stratify_labels is None:
        logger.warning("Skipping stratification for train split due to limited samples per class.")

    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        stratify=stratify_labels,
        random_state=seed,
    )
    test_ratio = float(train_cfg["test_size"]) / temp_size
    temp_stratify = (
        temp_df["label_id"] if temp_df["label_id"].value_counts().min() >= 2 else None
    )
    if temp_stratify is None:
        logger.warning("Skipping stratification for validation/test split due to limited samples per class.")

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio,
        stratify=temp_stratify,
        random_state=seed,
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    model_name = config["models"]["finetune"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    max_length = int(train_cfg["max_length"])
    train_ds = tokenize_dataset(train_ds, tokenizer, data_cfg["text_col"], max_length)
    val_ds = tokenize_dataset(val_ds, tokenizer, data_cfg["text_col"], max_length)
    test_ds = tokenize_dataset(test_ds, tokenizer, data_cfg["text_col"], max_length)

    train_ds = train_ds.rename_column("label_id", "labels")
    val_ds = val_ds.rename_column("label_id", "labels")
    test_ds = test_ds.rename_column("label_id", "labels")

    train_ds.set_format("torch")
    val_ds.set_format("torch")
    test_ds.set_format("torch")

    output_dir = "models/finetuned"
    ensure_dir(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        per_device_eval_batch_size=int(train_cfg["batch_size"]),
        num_train_epochs=float(train_cfg["epochs"]),
        learning_rate=float(train_cfg["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(pred, labels),
    )

    trainer.train()
    trainer.save_model(output_dir)
    with open(f"{output_dir}/label_map.json", "w", encoding="utf-8") as handle:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, handle, indent=2)

    eval_metrics = trainer.evaluate(test_ds)
    metrics_path = "reports/finetune_metrics.json"
    save_json(eval_metrics, metrics_path)
    logger.info("Saved fine-tune metrics to %s", metrics_path)

    predictions = trainer.predict(test_ds)
    top_k = int(config["inference"]["top_k"])
    test_df = test_ds.to_pandas()
    output_df = build_predictions_df(
        test_df,
        predictions.predictions,
        labels,
        data_cfg["id_col"],
        data_cfg["text_col"],
        data_cfg["label_col"],
        top_k,
    )
    pred_path = f"{config['outputs']['predictions_dir']}/finetune_predictions.csv"
    save_csv(output_df, pred_path)
    logger.info("Saved predictions to %s", pred_path)

    plots_dir = config["outputs"]["plots_dir"]
    ensure_dir(plots_dir)
    plot_training_curves(trainer.state.log_history, f"{plots_dir}/training_curves.png")
    save_json({"log_history": trainer.state.log_history}, "reports/training_log.json")


if __name__ == "__main__":
    main()
