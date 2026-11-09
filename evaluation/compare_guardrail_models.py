
import json
import csv
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from guardrail import ModernBERTGuardrail
from evaluation.baseline import BaselineDetector


ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    ROOT / "data" / "guardrail_datasets_csv_audited"
    / "test-00000-of-00001.csv"
)

OUTPUT_DIR = (
    ROOT / "artifacts" / "research_results" / "comparison"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "accuracy": round(
            accuracy_score(y_true, y_pred), 4
        ),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        "specificity": round(
            tn / (tn + fp), 4
        ) if (tn + fp) else None,
        "false_positive_rate": round(
            fp / (fp + tn), 4
        ) if (fp + tn) else None,
        "false_negative_rate": round(
            fn / (fn + tp), 4
        ) if (fn + tp) else None,
    }


def evaluate_models():
    print("\nLoading audited test dataset...")

    df = pd.read_csv(DATASET_PATH)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain text and label columns.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    print("Total samples:", len(df))
    print("Benign samples:", int((df["label"] == 0).sum()))
    print("Attack samples:", int((df["label"] == 1).sum()))

    print("\nLoading ModernBERT V2...")
    modernbert = ModernBERTGuardrail()

    print("\nLoading ProtectAI...")
    protectai = BaselineDetector()

    y_true = df["label"].tolist()

    modern_predictions = []
    protect_predictions = []

    modern_latencies = []
    protect_latencies = []

    modern_records = []
    protect_records = []

    print("\nStarting model evaluation...")

    for index, row in enumerate(
        df.itertuples(index=False), start=1
    ):
        text = str(row.text)
        actual = int(row.label)

        # ModernBERT V2
        modern_result = modernbert.evaluate(
            user_prompt=text
        )

        modern_prediction = int(
            modern_result["action"] in ("BLOCK", "FLAG")
        )

        # ProtectAI baseline
        protect_result = protectai.predict(text)

        protect_prediction = int(
            protect_result["attack_detected"]
        )

        modern_predictions.append(modern_prediction)
        protect_predictions.append(protect_prediction)

        modern_latencies.append(modern_result["latency_ms"])
        protect_latencies.append(
            protect_result["inference_latency_ms"]
        )

        modern_records.append({
            "sample_id": index,
            "actual": actual,
            "predicted": modern_prediction,
            "result": modern_result,
        })

        protect_records.append({
            "sample_id": index,
            "actual": actual,
            "predicted": protect_prediction,
            "result": protect_result,
        })

        if index % 50 == 0 or index == len(df):
            print(f"Evaluated {index}/{len(df)} samples")

    # Calculate the same seven metrics
    modern_metrics = calculate_metrics(
        y_true, modern_predictions
    )

    protect_metrics = calculate_metrics(
        y_true, protect_predictions
    )

    comparison = {
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "total_samples": len(df),
        "benign_samples": int((df["label"] == 0).sum()),
        "attack_samples": int((df["label"] == 1).sum()),
        "metrics": {
            "ModernBERT_V2": modern_metrics,
            "ProtectAI": protect_metrics,
        },
        "mean_inference_latency_ms": {
            "ModernBERT_V2": round(
                sum(modern_latencies) / len(modern_latencies), 2
            ),
            "ProtectAI": round(
                sum(protect_latencies) / len(protect_latencies), 2
            ),
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save individual predictions
    files = {
        f"modernbert_predictions_{timestamp}.json": modern_records,
        f"protectai_predictions_{timestamp}.json": protect_records,
    }

    for filename, records in files.items():
        with open(
            OUTPUT_DIR / filename,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(records, file, indent=4)

    # Save comparison JSON
    comparison_file = OUTPUT_DIR / f"comparison_{timestamp}.json"

    with open(comparison_file, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    # Save comparison CSV
    csv_file = OUTPUT_DIR / f"comparison_{timestamp}.csv"

    with open(
        csv_file, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Metric", "ModernBERT V2", "ProtectAI"
        ])

        for metric in modern_metrics:
            writer.writerow([
                metric,
                modern_metrics[metric],
                protect_metrics[metric],
            ])

    # Display results
    print("\n========== MODEL COMPARISON ==========")
    print(f"{'Metric':<27}{'ModernBERT V2':<20}{'ProtectAI'}")

    for metric in modern_metrics:
        print(
            f"{metric:<27}"
            f"{str(modern_metrics[metric]):<20}"
            f"{protect_metrics[metric]}"
        )

    print("\nMean inference latency:")
    print("ModernBERT V2:", comparison["mean_inference_latency_ms"]["ModernBERT_V2"], "ms")
    print("ProtectAI:", comparison["mean_inference_latency_ms"]["ProtectAI"], "ms")

    print("\nSaved results:")
    print("Directory:", OUTPUT_DIR)
    print("Comparison JSON:", comparison_file)
    print("Comparison CSV:", csv_file)


if __name__ == "__main__":
    evaluate_models()
