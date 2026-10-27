
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modernbert_guardrail"
VALIDATION_FILE = ROOT / "validation-00000-of-00001.parquet"

MAX_LENGTH = 256
BATCH_SIZE = 8

df = pd.read_parquet(VALIDATION_FILE)
texts = df["text"].fillna("").astype(str).tolist()
y_true = df["label"].astype(int).tolist()

print("Loading saved model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to("cpu")
model.eval()

scores = []

print(f"Scoring {len(texts)} validation prompts...")

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        scores.extend(probabilities.tolist())

        done = min(start + len(batch), len(texts))
        print(f"Progress: {done}/{len(texts)}", end="\r")

print("\n\nThreshold comparison")
print("Threshold | Precision | Recall | F1     | FPR    | Flagged")

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
                   0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:
    predictions = [int(s >= threshold) for s in scores]
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    recall = recall_score(
        y_true, predictions, zero_division=0
    )
    f1 = f1_score(y_true, predictions, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    flagged = sum(
        threshold <= s < 0.80 for s in scores
    ) if threshold < 0.80 else 0

    print(
        f"{threshold:8.2f} | {precision:.4f}    | "
        f"{recall:.4f} | {f1:.4f} | {fpr:.4f} | {flagged}"
    )

print("\nScore examples by actual class:")
for label, name in [(0, "Benign"), (1, "Malicious")]:
    class_scores = [
        s for s, y in zip(scores, y_true) if y == label
    ]
    class_scores.sort()
    mid = len(class_scores) // 2

    print(
        f"{name}: count={len(class_scores)}, "
        f"min={class_scores[0]:.4f}, "
        f"median={class_scores[mid]:.4f}, "
        f"max={class_scores[-1]:.4f}"
    )

print("\nCalibration analysis complete. Model was not modified.")
