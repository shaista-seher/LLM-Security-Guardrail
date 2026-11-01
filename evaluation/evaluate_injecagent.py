
import json
import os

from evaluation.baseline import BaselineDetector

DATA_FILES = [
    "datasets/injecagent/raw/test_cases_dh_base.json",
    "datasets/injecagent/raw/test_cases_ds_base.json",
]

def main():
    detector = BaselineDetector()
    records = []

    for file_path in DATA_FILES:
        print(f"\nReading: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            cases = json.load(file)

        for index, case in enumerate(cases):
            text = str(case.get("Tool Response", ""))

            if not text.strip():
                print(f"Skipping empty case {index + 1}")
                continue

            prediction = detector.predict(text)

            records.append({
                "dataset": os.path.basename(file_path),
                "case_id": index + 1,
                "attack_type": case.get("Attack Type", "Unknown"),
                "attack_detected": prediction["attack_detected"],
                "classifier_score": prediction["classifier_score"],
                "inference_latency_ms": prediction["inference_latency_ms"],
            })

        print(f"Loaded {len(cases)} cases")

    total = len(records)
    detected = sum(r["attack_detected"] for r in records)

    detection_rate = (detected / total * 100) if total else 0

    os.makedirs("artifacts", exist_ok=True)

    with open("artifacts/injecagent_baseline_results.json", "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)

    summary = {
        "total_cases": total,
        "attacks_detected": detected,
        "attacks_missed": total - detected,
        "attack_detection_rate_percent": round(detection_rate, 2),
    }

    with open("artifacts/injecagent_baseline_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\nEVALUATION COMPLETE")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
