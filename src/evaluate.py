from __future__ import annotations

import argparse
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils import (
    compute_metrics,
    ensure_dir,
    load_config,
    load_dataset_csv,
    save_csv,
    save_json,
    setup_logging,
)


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"true_label", "pred_1", "pred_2", "pred_3", "score_1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing prediction columns in {path}: {sorted(missing)}")
    return df


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    output_path: str,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_probability_hist(scores: List[float], output_path: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=20, color="#4c78a8", alpha=0.8)
    plt.xlabel("Top-1 Probability")
    plt.ylabel("Count")
    plt.title("Probability Histogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_class_distribution(labels: List[str], output_path: str) -> None:
    plt.figure(figsize=(10, 4))
    sns.barplot(x=labels.index, y=labels.values)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_method_comparison(metrics: Dict[str, Dict[str, float]], output_path: str) -> None:
    rows = []
    for method, scores in metrics.items():
        rows.append(
            {
                "method": method,
                "accuracy": scores["accuracy"],
                "macro_f1": scores["macro_f1"],
                "top3_accuracy": scores["top3_accuracy"],
            }
        )
    df = pd.DataFrame(rows)
    df = df.melt(id_vars="method", var_name="metric", value_name="value")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="metric", y="value", hue="method")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Method Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ticket tagging methods.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging("evaluate")
    config = load_config(args.config)

    data_cfg = config["data"]
    labels = list(config["labels"])
    predictions_dir = config["outputs"]["predictions_dir"]
    plots_dir = config["outputs"]["plots_dir"]
    ensure_dir(plots_dir)

    data_df = load_dataset_csv(
        data_cfg["processed_path"],
        data_cfg["text_col"],
        data_cfg["label_col"],
        data_cfg["id_col"],
    )
    class_counts = data_df[data_cfg["label_col"]].value_counts()
    plot_class_distribution(class_counts, f"{plots_dir}/class_distribution.png")

    methods = {
        "zero_shot": f"{predictions_dir}/zeroshot_predictions.csv",
        "few_shot": f"{predictions_dir}/fewshot_predictions.csv",
        "fine_tuned": f"{predictions_dir}/finetune_predictions.csv",
    }

    metrics: Dict[str, Dict[str, float]] = {}
    for method, path in methods.items():
        df = load_predictions(path)
        y_true = df["true_label"].tolist()
        y_pred = df["pred_1"].tolist()
        top_k = df[["pred_1", "pred_2", "pred_3"]].values.tolist()
        scores = compute_metrics(y_true, y_pred, top_k)
        metrics[method] = scores

        plot_confusion_matrix(
            y_true,
            y_pred,
            labels,
            f"{plots_dir}/{method}_confusion_matrix.png",
        )
        plot_probability_hist(
            df["score_1"].tolist(),
            f"{plots_dir}/{method}_probability_hist.png",
        )

    metrics_path = config["outputs"]["metrics_path"]
    save_json(metrics, metrics_path)
    logger.info("Saved metrics to %s", metrics_path)

    comparison = pd.DataFrame(
        [
            {
                "Method": method.replace("_", " ").title(),
                "Accuracy": metrics[method]["accuracy"],
                "F1": metrics[method]["macro_f1"],
                "Top3 Accuracy": metrics[method]["top3_accuracy"],
            }
            for method in metrics
        ]
    )
    comparison_path = config["outputs"]["comparison_path"]
    save_csv(comparison, comparison_path)
    comparison_md = comparison.to_markdown(index=False)
    with open("reports/comparison.md", "w", encoding="utf-8") as handle:
        handle.write(comparison_md)

    plot_method_comparison(metrics, f"{plots_dir}/method_comparison.png")


if __name__ == "__main__":
    main()
