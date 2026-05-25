from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
from transformers import pipeline

from utils import (
    format_topk,
    load_config,
    load_dataset_csv,
    save_csv,
    set_seed,
    setup_logging,
)


def predict_zero_shot(
    texts: List[str],
    labels: List[str],
    model_name: str,
    top_k: int,
) -> List[Dict[str, List[float]]]:
    classifier = pipeline("zero-shot-classification", model=model_name)
    predictions: List[Dict[str, List[float]]] = []
    for text in tqdm(texts, desc="Zero-shot predictions"):
        result = classifier(text, labels)
        topk = format_topk(result["labels"], result["scores"], top_k)
        predictions.append(topk)
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
        labels = pred["labels"]
        scores = pred["scores"]
        record = {
            "ticket_id": getattr(row, id_col),
            "ticket_text": getattr(row, text_col),
            "true_label": getattr(row, label_col),
        }
        for idx in range(top_k):
            record[f"pred_{idx + 1}"] = labels[idx]
            record[f"score_{idx + 1}"] = scores[idx]
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot ticket tagging.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("zeroshot")
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
    model_name = config["models"]["zeroshot"]
    top_k = int(config["inference"]["top_k"])

    predictions = predict_zero_shot(df[data_cfg["text_col"]].tolist(), labels, model_name, top_k)
    output_path = f"{config['outputs']['predictions_dir']}/zeroshot_predictions.csv"
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
