
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "artifacts" / "research_results" / "comparison"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def latest_file(pattern):
    files = list(RESULT_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    return max(files, key=lambda p: p.stat().st_mtime)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(RESULT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", filename)


def load_results():
    comparison_file = latest_file("comparison_*.json")
    modern_file = latest_file("modernbert_predictions_*.json")
    protect_file = latest_file("protectai_predictions_*.json")

    with open(comparison_file, encoding="utf-8") as f:
        comparison = json.load(f)

    with open(modern_file, encoding="utf-8") as f:
        modern = json.load(f)

    with open(protect_file, encoding="utf-8") as f:
        protect = json.load(f)

    if len(modern) != len(protect):
        raise ValueError("Prediction counts do not match.")

    for m, p in zip(modern, protect):
        if m["actual"] != p["actual"]:
            raise ValueError("Ground-truth labels do not match.")

    return comparison, modern, protect


def metric_comparison(metrics):
    names = ["Accuracy", "Precision", "Recall", "F1-score", "Specificity"]
    keys = ["accuracy", "precision", "recall", "f1_score", "specificity"]

    modern = [metrics["ModernBERT_V2"][k] * 100 for k in keys]
    protect = [metrics["ProtectAI"][k] * 100 for k in keys]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, modern, width, label="ModernBERT V2")
    plt.bar(x + width / 2, protect, width, label="ProtectAI")

    plt.xticks(x, names)
    plt.ylabel("Score (%)")
    plt.ylim(0, 105)
    plt.title("Overall Classification Performance Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure("01_metrics_comparison.png")


def error_rate_comparison(metrics):
    names = ["False Positive Rate", "False Negative Rate"]
    keys = ["false_positive_rate", "false_negative_rate"]

    modern = [metrics["ModernBERT_V2"][k] * 100 for k in keys]
    protect = [metrics["ProtectAI"][k] * 100 for k in keys]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, modern, width, label="ModernBERT V2")
    plt.bar(x + width / 2, protect, width, label="ProtectAI")

    plt.xticks(x, names)
    plt.ylabel("Error Rate (%)")
    plt.title("False Positive and False Negative Rate Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure("02_error_rate_comparison.png")


def confusion_matrix_plot(records, model_name, filename):
    actual = [r["actual"] for r in records]
    predicted = [r["predicted"] for r in records]

    matrix = confusion_matrix(actual, predicted, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks([0, 1], ["Benign", "Attack"])
    plt.yticks([0, 1], ["Benign", "Attack"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(matrix[i, j]),
                ha="center", va="center", fontsize=13
            )

    plt.colorbar(label="Number of Samples")
    save_figure(filename)


def precision_recall_comparison(metrics):
    names = ["Precision", "Recall"]

    modern = [
        metrics["ModernBERT_V2"]["precision"] * 100,
        metrics["ModernBERT_V2"]["recall"] * 100,
    ]

    protect = [
        metrics["ProtectAI"]["precision"] * 100,
        metrics["ProtectAI"]["recall"] * 100,
    ]

    x = np.arange(2)
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, modern, width, label="ModernBERT V2")
    plt.bar(x + width / 2, protect, width, label="ProtectAI")

    plt.xticks(x, names)
    plt.ylabel("Score (%)")
    plt.ylim(0, 105)
    plt.title("Precision and Recall Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure("05_precision_recall_comparison.png")


def latency_comparison(comparison):
    latency = comparison["mean_inference_latency_ms"]

    names = ["ModernBERT V2", "ProtectAI"]
    values = [
        latency["ModernBERT_V2"],
        latency["ProtectAI"],
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values)

    plt.ylabel("Mean Inference Latency (ms)")
    plt.title("Model Inference Latency Comparison")

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f} ms",
            ha="center",
            va="bottom",
        )

    plt.grid(axis="y", alpha=0.25)
    save_figure("06_latency_comparison.png")


def detection_outcomes(modern, protect):
    def counts(records):
        actual = np.array([r["actual"] for r in records])
        predicted = np.array([r["predicted"] for r in records])

        return [
            int(((actual == 0) & (predicted == 0)).sum()),
            int(((actual == 1) & (predicted == 1)).sum()),
            int(((actual == 0) & (predicted == 1)).sum()),
            int(((actual == 1) & (predicted == 0)).sum()),
        ]

    labels = ["True Negatives", "True Positives",
              "False Positives", "False Negatives"]

    modern_counts = counts(modern)
    protect_counts = counts(protect)

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, modern_counts, width, label="ModernBERT V2")
    plt.bar(x + width / 2, protect_counts, width, label="ProtectAI")

    plt.xticks(x, labels)
    plt.ylabel("Number of Samples")
    plt.title("Correct and Incorrect Detection Outcomes")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_figure("07_detection_outcomes.png")


def radar_comparison(metrics):
    labels = [
        "Accuracy", "Precision", "Recall",
        "F1-score", "Specificity"
    ]

    keys = [
        "accuracy", "precision", "recall",
        "f1_score", "specificity"
    ]

    modern = [metrics["ModernBERT_V2"][k] for k in keys]
    protect = [metrics["ProtectAI"][k] for k in keys]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    modern += [modern[0]]
    protect += [protect[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    ax.plot(angles, modern, marker="o", label="ModernBERT V2")
    ax.fill(angles, modern, alpha=0.10)

    ax.plot(angles, protect, marker="o", label="ProtectAI")
    ax.fill(angles, protect, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Overall Model Performance Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    save_figure("08_model_performance_radar.png")


def main():
    print("Loading saved results...")

    comparison, modern, protect = load_results()
    metrics = comparison["metrics"]

    print("ModernBERT records:", len(modern))
    print("ProtectAI records:", len(protect))

    metric_comparison(metrics)
    error_rate_comparison(metrics)

    confusion_matrix_plot(
        modern,
        "ModernBERT V2",
        "03_modernbert_confusion_matrix.png",
    )

    confusion_matrix_plot(
        protect,
        "ProtectAI",
        "04_protectai_confusion_matrix.png",
    )

    precision_recall_comparison(metrics)
    latency_comparison(comparison)
    detection_outcomes(modern, protect)
    radar_comparison(metrics)

    print("\nAll 8 research graphs generated successfully.")
    print("Output directory:", RESULT_DIR)


if __name__ == "__main__":
    main()
