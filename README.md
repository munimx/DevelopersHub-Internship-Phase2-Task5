# Task 5 — Auto Tagging Support Tickets Using LLM

## 1. Objective
Build a system that tags support tickets with top-3 predicted categories using zero-shot, few-shot, and fine-tuned approaches, then compare performance.

## 2. Dataset
The pipeline expects a CSV with columns: `ticket_id`, `ticket_text`, `category`. A sample dataset is provided at `data/support_tickets.csv`. Replace it with your Kaggle/Zendesk/synthetic dataset if needed.

## 3. Methods
- **Zero-shot**: `facebook/bart-large-mnli` with candidate labels.
- **Few-shot**: `google/flan-t5-base` with prompt examples, scored by label likelihoods.
- **Fine-tuning**: `distilbert-base-uncased` sequence classifier.

## 4. Zero Shot
Run preprocessing and zero-shot prediction:
```bash
python src/preprocess.py --config config.yaml
python src/zeroshot.py --config config.yaml
```
Outputs: `reports/predictions/zeroshot_predictions.csv`

## 5. Few Shot
```bash
python src/fewshot.py --config config.yaml
```
Outputs: `reports/predictions/fewshot_predictions.csv`

## 6. Fine Tuning
```bash
python src/finetune.py --config config.yaml
```
Outputs:
- Model: `models/finetuned/`
- Metrics: `reports/finetune_metrics.json`
- Predictions: `reports/predictions/finetune_predictions.csv`
- Training curves: `reports/plots/training_curves.png`

## 7. Results
Run evaluation to compute metrics and generate plots:
```bash
python src/evaluate.py --config config.yaml
```
This writes `reports/metrics.json`, `reports/comparison.csv`, and charts in `reports/plots/`.

## 8. Comparison
The comparison table is saved to `reports/comparison.md`:

| Method | Accuracy | F1 | Top3 Accuracy |
| --- | --- | --- | --- |
| Zero Shot | ... | ... | ... |
| Few Shot | ... | ... | ... |
| Fine Tuned | ... | ... | ... |

## 9. Inference
```bash
python src/inference.py --config config.yaml --method finetuned --text "My payment failed after checkout"
```
Example output:
```json
{
  "ticket": "My payment failed after checkout",
  "top_predictions": [
    { "label": "billing", "score": 0.81 },
    { "label": "refund", "score": 0.12 },
    { "label": "subscription", "score": 0.05 }
  ]
}
```
