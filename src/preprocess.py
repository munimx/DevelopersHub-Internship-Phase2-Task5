from __future__ import annotations

import argparse

import pandas as pd

from utils import ensure_dir, load_config, load_dataset_csv, save_csv, set_seed, setup_logging


def preprocess(config_path: str) -> str:
    logger = setup_logging("preprocess")
    config = load_config(config_path)
    seed = int(config["project"]["seed"])
    set_seed(seed)

    data_cfg = config["data"]
    df = load_dataset_csv(
        data_cfg["raw_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
        data_cfg["id_col"],
    )

    df = df.dropna(subset=[data_cfg["text_col"], data_cfg["label_col"]]).copy()
    df[data_cfg["text_col"]] = df[data_cfg["text_col"]].astype(str).str.strip()
    df[data_cfg["label_col"]] = df[data_cfg["label_col"]].astype(str).str.strip()

    valid_labels = set(config["labels"])
    unknown_labels = set(df[data_cfg["label_col"]]) - valid_labels
    if unknown_labels:
        raise ValueError(f"Unknown labels in data: {sorted(unknown_labels)}")

    processed_path = data_cfg["processed_path"]
    ensure_dir(processed_path.rsplit("/", 1)[0])
    save_csv(df, processed_path)
    logger.info("Saved processed data to %s", processed_path)
    return processed_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess support ticket data.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess(args.config)


if __name__ == "__main__":
    main()
