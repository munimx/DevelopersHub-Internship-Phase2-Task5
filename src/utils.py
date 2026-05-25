from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_csv(
    path: str, text_col: str, label_col: str, id_col: str
) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    missing = {text_col, label_col, id_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def save_json(data: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def top_k_accuracy(y_true: Sequence[str], top_k_preds: Sequence[Sequence[str]]) -> float:
    hits = [true in preds for true, preds in zip(y_true, top_k_preds)]
    return float(np.mean(hits)) if hits else 0.0


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    top_k_preds: Sequence[Sequence[str]],
) -> Dict[str, float]:
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    top3 = top_k_accuracy(y_true, top_k_preds)
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "precision": float(precision),
        "recall": float(recall),
        "top3_accuracy": float(top3),
    }


def format_topk(
    labels: Sequence[str],
    scores: Sequence[float],
    top_k: int,
) -> Dict[str, List[Any]]:
    pairs = sorted(zip(labels, scores), key=lambda item: item[1], reverse=True)
    top_pairs = pairs[:top_k]
    return {
        "labels": [label for label, _ in top_pairs],
        "scores": [float(score) for _, score in top_pairs],
    }


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_fewshot_prompt(
    examples: Iterable[Dict[str, Any]], labels: Sequence[str], ticket_text: str
) -> str:
    label_list = ", ".join(labels)
    lines = [
        "You are a support ticket classifier.",
        f"Possible labels: {label_list}.",
        "Return only the labels separated by spaces.",
        "",
    ]
    for idx, example in enumerate(examples, start=1):
        tags = " ".join(example["tags"])
        lines.extend(
            [
                f"Example {idx}:",
                f"Ticket: {example['ticket']}",
                f"Tags: {tags}",
                "",
            ]
        )
    lines.extend([f"Ticket: {ticket_text}", "Tags:"])
    return "\n".join(lines)
