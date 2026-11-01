
"""Member 3 + Member 4 evaluation pipeline.

This stage generates and validates attack cases and prepares the
evaluation workflow. It does not invent guardrail or target results.
"""

import uuid
from pathlib import Path

from red_team.attack_generator import generate_attack_cases
from common.schemas import (
    validate_attack_case,
    validate_guardrail_result,
    validate_target_result,
)
from evaluation.records import validate_evaluation_record, save_records
from evaluation.metrics import calculate_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_validated_cases(limit=2, use_gemma=False):
    """Generate attack cases using Member 3 and validate their contracts."""

    cases = generate_attack_cases(
        limit=limit,
        use_gemma=use_gemma,
    )

    for case in cases:
        validate_attack_case(case)

    return cases


def build_evaluation_record(
    case,
    guardrail_result,
    target_result,
    baseline_result=None,
    attack_objective_achieved=None,
    latency_ms=None,
):
    """Build one Member 4 record from supplied execution results."""

    if guardrail_result is None or target_result is None:
        raise ValueError("Actual guardrail and target results are required.")

    validate_guardrail_result(guardrail_result)
    validate_target_result(target_result)

    if guardrail_result["case_id"] != case["case_id"]:
        raise ValueError("Guardrail case_id does not match attack case.")

    if target_result["case_id"] != case["case_id"]:
        raise ValueError("Target case_id does not match attack case.")

    record = {
        "run_id": str(uuid.uuid4()),
        "case_id": case["case_id"],
        "dataset_source": case["generation_method"],
        "ground_truth_attack": case["expected_attack"],
        "guardrail_result": guardrail_result,
        "target_result": target_result,
        "baseline_result": baseline_result or {},
        "attack_objective_achieved": attack_objective_achieved,
        "end_to_end_latency_ms": latency_ms,
        "notes": "Results supplied by integration runner; execution mode depends on the configured application.",
    }

    validate_evaluation_record(record)
    return record


def calculate_guardrail_metrics(records):
    """Compare Member 3 ground truth with Member 1 guardrail predictions."""

    actual = [
        int(record["ground_truth_attack"])
        for record in records
    ]

    predicted = [
        int(record["guardrail_result"]["attack_detected"])
        for record in records
    ]

    return calculate_metrics(actual, predicted)


def save_evaluation(records, output_name="m3_m4_evaluation.json"):
    """Save validated evaluation records."""

    output_path = PROJECT_ROOT / "artifacts" / output_name
    save_records(records, output_path)
    return output_path

