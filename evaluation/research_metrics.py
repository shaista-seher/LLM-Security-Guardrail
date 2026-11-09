"""
Research Evaluation Metrics
Adaptive Dual-Agent LLM Security Framework

Evaluates:
1. ModernBERT guardrail performance
2. ProtectAI baseline performance, when available
3. Attack success rate
4. Adaptive Red-Team generation statistics
5. Guardrail and end-to-end latency

Input:
    artifacts/m3_m4_evaluation.json

Output:
    artifacts/research_results/
"""

import json
import csv
import re
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, median

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
)


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT_DIR / "artifacts" / "m3_m4_evaluation.json"

OUTPUT_DIR = ROOT_DIR / "artifacts" / "research_results"


# --------------------------------------------------
# BASIC METRIC CALCULATIONS
# --------------------------------------------------

def calculate_binary_metrics(actual, predicted):

    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths must match.")

    if not actual:
        raise ValueError("No evaluation records available.")

    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))

    total = len(actual)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    specificity = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    accuracy = (tp + tn) / total

    balanced_accuracy = (
        (recall + specificity) / 2
    )

    mcc = matthews_corrcoef(actual, predicted)

    return {
        "sample_count": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation_coefficient": float(mcc),
    }


# --------------------------------------------------
# SCORE-BASED METRICS
# --------------------------------------------------

def calculate_auc_metrics(actual, scores):

    if len(set(actual)) < 2:
        return {
            "auroc": None,
            "auprc": None,
            "note": "Both classes are required for AUC metrics."
        }

    return {
        "auroc": float(roc_auc_score(actual, scores)),
        "auprc": float(average_precision_score(actual, scores)),
    }


# --------------------------------------------------
# LATENCY STATISTICS
# --------------------------------------------------

def calculate_latency(values):

    values = [
        float(value)
        for value in values
        if isinstance(value, (int, float))
        and value >= 0
    ]

    if not values:
        return {
            "sample_count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "minimum_ms": None,
            "maximum_ms": None,
        }

    ordered = sorted(values)

    # Nearest-rank 95th percentile
    p95_index = max(0, int(0.95 * len(ordered) + 0.999999) - 1)

    return {
        "sample_count": len(values),
        "mean_ms": mean(values),
        "median_ms": median(values),
        "p95_ms": ordered[p95_index],
        "minimum_ms": min(values),
        "maximum_ms": max(values),
    }


# --------------------------------------------------
# GUARDRAIL LATENCY EXTRACTION
# --------------------------------------------------

def extract_guardrail_latency(record):

    guardrail = record.get("guardrail_result", {})

    for signal in guardrail.get("signals", []):
        details = signal.get("details", "")

        match = re.search(
            r"Evaluation latency:\s*([\d.]+)\s*ms",
            details,
        )

        if match:
            return float(match.group(1))

    return None


# --------------------------------------------------
# ATTACK SUCCESS RATE
# --------------------------------------------------

def calculate_attack_success(records):

    attack_records = [
        record for record in records
        if record.get("ground_truth_attack") is True
    ]

    outcomes = [
        record.get("attack_objective_achieved")
        for record in attack_records
    ]

    known = [
        outcome for outcome in outcomes
        if isinstance(outcome, bool)
    ]

    successful = sum(known)

    return {
        "total_attack_cases": len(attack_records),
        "known_objective_outcomes": len(known),
        "unknown_objective_outcomes": len(outcomes) - len(known),
        "successful_attacks": successful,
        "blocked_or_unsuccessful_attacks": len(known) - successful,
        "attack_success_rate": (
            successful / len(known) if known else None
        ),
        "objective_evaluation_coverage": (
            len(known) / len(outcomes) if outcomes else None
        ),
        "note": (
            "ASR is calculated only from known objective outcomes. "
            "Allowed attacks with unknown outcomes are not assumed "
            "to have succeeded or failed."
        ),
    }


# --------------------------------------------------
# BASELINE PREDICTION EXTRACTION
# --------------------------------------------------

def extract_binary_prediction(result):

    if not isinstance(result, dict):
        return None

    possible_keys = (
        "attack_detected",
        "is_attack",
        "predicted_attack",
        "prediction",
        "predicted_label",
        "label",
    )

    for key in possible_keys:

        value = result.get(key)

        if isinstance(value, bool):
            return int(value)

        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)

        if isinstance(value, str):

            normalized = value.strip().lower()

            if normalized in ("attack", "malicious", "block", "flag"):
                return 1

            if normalized in ("benign", "normal", "safe", "allow"):
                return 0

    return None


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_modernbert(records):

    actual = []
    predicted = []
    scores = []

    end_to_end_latency = []
    guardrail_latency = []

    for record in records:

        ground_truth = record.get("ground_truth_attack")

        if not isinstance(ground_truth, bool):
            continue

        guardrail = record.get("guardrail_result", {})

        decision = str(
            guardrail.get("decision", "")
        ).upper()

        if decision not in ("ALLOW", "FLAG", "BLOCK"):
            continue

        actual.append(int(ground_truth))

        # FLAG and BLOCK both count as detected attacks.
        predicted.append(
            1 if decision in ("FLAG", "BLOCK") else 0
        )

        score = guardrail.get("classifier_score")

        if isinstance(score, (int, float)):
            scores.append(float(score))

        latency = record.get("end_to_end_latency_ms")

        if isinstance(latency, (int, float)):
            end_to_end_latency.append(float(latency))

        guardrail_time = extract_guardrail_latency(record)

        if guardrail_time is not None:
            guardrail_latency.append(guardrail_time)

    results = calculate_binary_metrics(actual, predicted)

    if len(scores) == len(actual):
        results["score_metrics"] = calculate_auc_metrics(
            actual, scores
        )
    else:
        results["score_metrics"] = {
            "auroc": None,
            "auprc": None,
            "note": "Complete classifier scores unavailable."
        }

    results["end_to_end_latency"] = calculate_latency(
        end_to_end_latency
    )

    results["guardrail_latency"] = calculate_latency(
        guardrail_latency
    )

    return results


# --------------------------------------------------
# BASELINE EVALUATION
# --------------------------------------------------

def evaluate_baseline(records):

    actual = []
    predicted = []

    for record in records:

        ground_truth = record.get("ground_truth_attack")

        if not isinstance(ground_truth, bool):
            continue

        baseline_result = record.get("baseline_result", {})

        prediction = extract_binary_prediction(baseline_result)

        if prediction is None:
            continue

        actual.append(int(ground_truth))
        predicted.append(prediction)

    if not actual:
        return {
            "available": False,
            "note": (
                "No usable baseline predictions found. "
                "Run the evaluation pipeline with use_baseline=True."
            )
        }

    results = calculate_binary_metrics(actual, predicted)
    results["available"] = True

    return results


# --------------------------------------------------
# ADAPTIVE RED-TEAM STATISTICS
# --------------------------------------------------

def evaluate_red_team(records):

    generation_counts = {}
    adaptive_count = 0
    total_attacks = 0
    attack_ids = set()

    for record in records:

        method = record.get(
            "generation_method",
            record.get("dataset_source", "unknown")
        )

        generation_counts[method] = (
            generation_counts.get(method, 0) + 1
        )

        if record.get("ground_truth_attack") is True:

            total_attacks += 1
            attack_ids.add(record.get("case_id"))

            if method == "gemma_adaptive":
                adaptive_count += 1

    return {
        "total_evaluated_cases": len(records),
        "total_attack_cases": total_attacks,
        "adaptive_attack_cases": adaptive_count,
        "generation_methods": generation_counts,
        "unique_attack_case_ids": len(attack_ids),
        "note": (
            "These are generation and execution counts. "
            "Attack novelty, semantic diversity and feedback "
            "effectiveness require additional multi-round analysis."
        ),
    }


# --------------------------------------------------
# COMPLETE EVALUATION
# --------------------------------------------------

def run_research_evaluation():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if isinstance(records, dict):
        records = records.get("records", [])

    if not isinstance(records, list) or not records:
        raise ValueError("No valid evaluation records found.")

    print("\nLoading evaluation records...")
    print("Total records:", len(records))

    modernbert_results = evaluate_modernbert(records)

    baseline_results = evaluate_baseline(records)

    attack_results = calculate_attack_success(records)

    red_team_results = evaluate_red_team(records)

    results = {
        "experiment": "Adaptive Dual-Agent LLM Security Framework",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "input_file": str(INPUT_FILE),

        "total_records": len(records),

        "modernbert_v2": modernbert_results,

        "protectai_baseline": baseline_results,

        "attack_objective_evaluation": attack_results,

        "red_team_evaluation": red_team_results,
    }

    # ----------------------------------------------
    # SAVE RESULTS
    # ----------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = OUTPUT_DIR / f"research_metrics_{timestamp}.json"
    csv_path = OUTPUT_DIR / f"research_metrics_{timestamp}.csv"

    latest_json = OUTPUT_DIR / "research_metrics_latest.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    with latest_json.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Flatten top-level metric results into CSV.
    rows = []

    def flatten(data, prefix=""):

        for key, value in data.items():

            name = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flatten(value, name)

            elif isinstance(value, list):
                rows.append({
                    "metric": name,
                    "value": json.dumps(value),
                })

            else:
                rows.append({
                    "metric": name,
                    "value": value,
                })

    flatten(results)

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "value"]
        )

        writer.writeheader()
        writer.writerows(rows)

    # ----------------------------------------------
    # DISPLAY SUMMARY
    # ----------------------------------------------

    print("\n========== RESEARCH EVALUATION ==========")

    print("\nMODERNBERT V2")
    for key, value in modernbert_results.items():
        print(f"{key}: {value}")

    print("\nPROTECTAI BASELINE")
    for key, value in baseline_results.items():
        print(f"{key}: {value}")

    print("\nATTACK OBJECTIVE EVALUATION")
    for key, value in attack_results.items():
        print(f"{key}: {value}")

    print("\nRED-TEAM EVALUATION")
    for key, value in red_team_results.items():
        print(f"{key}: {value}")

    print("\n========== SAVED RESULTS ==========")
    print("JSON:", json_path)
    print("CSV:", csv_path)
    print("Latest:", latest_json)

    return results


if __name__ == "__main__":
    run_research_evaluation()