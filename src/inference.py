from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fewshot import predict_fewshot
from utils import get_device, load_config, set_seed
from zeroshot import predict_zero_shot


def predict_finetuned(
    text: str, model_dir: str, labels: List[str], top_k: int
) -> Dict[str, List[float]]:
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    top_indices = np.argsort(probs)[::-1][:top_k]
    return {
        "labels": [labels[idx] for idx in top_indices],
        "scores": [float(probs[idx]) for idx in top_indices],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ticket tagging inference.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    parser.add_argument(
        "--method",
        choices=["zeroshot", "fewshot", "finetuned"],
        default="finetuned",
        help="Inference method.",
    )
    parser.add_argument("--text", required=True, help="Ticket text to classify.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    labels = list(config["labels"])
    top_k = int(config["inference"]["top_k"])

    if args.method == "zeroshot":
        preds = predict_zero_shot([args.text], labels, config["models"]["zeroshot"], top_k)[0]
    elif args.method == "fewshot":
        preds = predict_fewshot(
            [args.text],
            labels,
            config["models"]["fewshot"],
            list(config["fewshot"]["examples"]),
            top_k,
        )[0]
    else:
        preds = predict_finetuned(args.text, "models/finetuned", labels, top_k)

    output = {
        "ticket": args.text,
        "top_predictions": [
            {"label": label, "score": score}
            for label, score in zip(preds["labels"], preds["scores"])
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
