
import json
import time
from pathlib import Path

import pandas as pd
import torch
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
MODEL_DIR = ROOT / "modernbert_guardrail"
TEST_FILE = ROOT / "test-00000-of-00001.parquet"
MAX_LENGTH = 256
BATCH_SIZE = 4

# Check that the saved model and test data exist
if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Saved model not found: {MODEL_DIR}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_FILE}")

print("Loading saved ModernBERT model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.to("cpu")
model.eval()

print("Loading unseen test dataset...")
df = pd.read_parquet(TEST_FILE)
texts = df["text"].fillna("").astype(str).tolist()
y_true = df["label"].astype(int).tolist()

predictions = []
total_inference_time = 0.0

print(f"Evaluating {len(texts)} test examples...")

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        batch_start = time.perf_counter()

        inputs = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = model(**inputs)
        batch_predictions = torch.argmax(
            outputs.logits, dim=1
        ).tolist()

        total_inference_time += time.perf_counter() - batch_start
        predictions.extend(batch_predictions)

        completed = min(start + len(batch_texts), len(texts))
        print(f"Progress: {completed}/{len(texts)}", end="\r")

# Label mapping: 0 = benign, 1 = malicious
tn, fp, fn, tp = confusion_matrix(
    y_true, predictions, labels=[0, 1]
).ravel()

precision = precision_score(
    y_true, predictions, pos_label=1, zero_division=0
)
recall = recall_score(
    y_true, predictions, pos_label=1, zero_division=0
)
f1 = f1_score(
    y_true, predictions, pos_label=1, zero_division=0
)
accuracy = accuracy_score(y_true, predictions)
fpr = fp / (fp + tn) if (fp + tn) else 0.0
avg_latency_ms = total_inference_time / len(texts) * 1000

print("\n\n========== FINAL TEST RESULTS ==========")
print(f"Test examples:             {len(texts)}")
print(f"Accuracy:                  {accuracy:.4f}")
print(f"Precision:                 {precision:.4f}")
print(f"Recall:                    {recall:.4f}")
print(f"F1-score:                  {f1:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"Avg. latency per prompt:   {avg_latency_ms:.2f} ms")
print(f"Total evaluation time:     {total_inference_time / 60:.2f} min")

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print("                 Predicted Benign  Predicted Malicious")
print(f"Actual Benign          {tn:5d}               {fp:5d}")
print(f"Actual Malicious       {fn:5d}               {tp:5d}")

print("\nClassification Report:")
print(classification_report(
    y_true,
    predictions,
    labels=[0, 1],
    target_names=["Benign", "Malicious"],
    digits=4,
    zero_division=0,
))

# Save the final results for the project report
results = {
    "test_examples": len(texts),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "false_positive_rate": fpr,
    "average_latency_ms_per_prompt": avg_latency_ms,
    "total_evaluation_time_seconds": total_inference_time,
    "confusion_matrix": {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    },
}

output_file = ROOT / "test_metrics.json"
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print(f"\nMetrics saved to: {output_file}")
print("Evaluation complete. The trained model was not modified.")
