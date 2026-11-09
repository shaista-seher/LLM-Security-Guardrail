
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "guardrail_datasets_csv"
MODEL_DIR = ROOT / "modernbert_guardrail_candidate"

TEST_FILE = DATA_DIR / "test-00000-of-00001.csv"
OUTPUT_FILE = MODEL_DIR / "test_metrics.json"

MAX_LENGTH = 256
BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Load calibrated threshold
with open(MODEL_DIR / "calibration_results.json") as f:
    calibration = json.load(f)

threshold = calibration["best_validation_threshold"]

# Load test data
df = pd.read_csv(TEST_FILE)

if df["text"].isna().any() or df["label"].isna().any():
    raise ValueError("Test data contains missing text or labels.")

texts = df["text"].astype(str).tolist()
labels = df["label"].astype(int).to_numpy()

print("Test samples:", len(df))
print("Threshold:", threshold)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
).to(device)

model.eval()

# Predict
all_scores = []
inference_times = []

for start in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[start:start + BATCH_SIZE]

    inputs = tokenizer(
        batch_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = model(**inputs).logits
        else:
            logits = model(**inputs).logits

        scores = torch.softmax(
            logits.float(), dim=1
        )[:, 1]

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    inference_times.append(
        elapsed / len(batch_texts) * 1000
    )

    all_scores.extend(scores.cpu().tolist())

scores = np.array(all_scores)
predictions = (scores >= threshold).astype(int)

# Metrics
accuracy = accuracy_score(labels, predictions)

precision = precision_score(
    labels, predictions, zero_division=0
)

recall = recall_score(
    labels, predictions, zero_division=0
)

f1 = f1_score(
    labels, predictions, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    labels, predictions, labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) else 0.0

# Results
results = {
    "model": "ModernBERT Candidate",
    "test_samples": len(df),
    "threshold": float(threshold),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1": float(f1),
    "fpr": float(fpr),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "average_batch_amortized_latency_ms": float(
        np.mean(inference_times)
    ),
}

print("\n========== TEST RESULTS ==========")

for key, value in results.items():
    print(f"{key}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nClassification Report:")
print(
    classification_report(
        labels,
        predictions,
        target_names=["BENIGN", "INJECTION"],
        zero_division=0,
    )
)

# Category-wise evaluation
df["true_label"] = labels
df["predicted_label"] = predictions

print("\n========== CATEGORY RESULTS ==========")

category_results = {}

for category, group in df.groupby("category"):
    y_true = group["true_label"].to_numpy()
    y_pred = group["predicted_label"].to_numpy()

    category_results[category] = {
        "samples": len(group),
        "precision": float(precision_score(
            y_true, y_pred, zero_division=0
        )),
        "recall": float(recall_score(
            y_true, y_pred, zero_division=0
        )),
        "f1": float(f1_score(
            y_true, y_pred, zero_division=0
        )),
    }

    print(category, category_results[category])

results["category_results"] = category_results

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:", OUTPUT_FILE)
