from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import (
    build_fewshot_prompt,
    get_device,
    load_config,
    load_dataset_csv,
    save_csv,
    set_seed,
    setup_logging,
)


def score_labels(
    prompt: str,
    labels: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
) -> List[float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    scores: List[float] = []
    with torch.no_grad():
        for label in labels:
            target_ids = tokenizer(label, return_tensors="pt").input_ids.to(device)
            output = model(**inputs, labels=target_ids)
            scores.append(-output.loss.item())
    probs = torch.softmax(torch.tensor(scores), dim=0).tolist()
    return [float(score) for score in probs]


def predict_fewshot(
    texts: List[str],
    labels: List[str],
    model_name: str,
    examples: List[Dict[str, List[str]]],
    top_k: int,
) -> List[Dict[str, List[float]]]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    predictions: List[Dict[str, List[float]]] = []
    for text in tqdm(texts, desc="Few-shot predictions"):
        prompt = build_fewshot_prompt(examples, labels, text)
        probs = score_labels(prompt, labels, tokenizer, model, device)
        pairs = sorted(zip(labels, probs), key=lambda item: item[1], reverse=True)[:top_k]
        predictions.append(
            {
                "labels": [label for label, _ in pairs],
                "scores": [float(score) for _, score in pairs],
            }
        )
    return predictions


def build_predictions_df(
    df: pd.DataFrame,
    predictions: List[Dict[str, List[float]]],
    top_k: int,
    text_col: str,
    label_col: str,
    id_col: str,
) -> pd.DataFrame:
    records = []
    for row, pred in zip(df.itertuples(index=False), predictions):
        record = {
            "ticket_id": getattr(row, id_col),
            "ticket_text": getattr(row, text_col),
            "true_label": getattr(row, label_col),
        }
        for idx in range(top_k):
            record[f"pred_{idx + 1}"] = pred["labels"][idx]
            record[f"score_{idx + 1}"] = pred["scores"][idx]
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot ticket tagging.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("fewshot")
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))

    data_cfg = config["data"]
    df = load_dataset_csv(
        data_cfg["processed_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
        data_cfg["id_col"],
    )

    labels = list(config["labels"])
    model_name = config["models"]["fewshot"]
    top_k = int(config["inference"]["top_k"])
    examples = list(config["fewshot"]["examples"])

    predictions = predict_fewshot(
        df[data_cfg["text_col"]].tolist(), labels, model_name, examples, top_k
    )
    output_path = f"{config['outputs']['predictions_dir']}/fewshot_predictions.csv"
    output_df = build_predictions_df(
        df,
        predictions,
        top_k,
        data_cfg["text_col"],
        data_cfg["label_col"],
        data_cfg["id_col"],
    )
    save_csv(output_df, output_path)
    logger.info("Saved predictions to %s", output_path)


if __name__ == "__main__":
    main()
