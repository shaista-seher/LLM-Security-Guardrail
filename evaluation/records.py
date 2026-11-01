
"""Save evaluation records as JSON."""

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "run_id",
    "case_id",
    "dataset_source",
    "ground_truth_attack",
    "guardrail_result",
    "target_result",
    "baseline_result",
    "attack_objective_achieved",
    "end_to_end_latency_ms",
    "notes",
}


def validate_evaluation_record(record: dict[str, Any]) -> None:
    """Check required fields and basic evaluation-record types."""
    if not isinstance(record, dict):
        raise ValueError("Evaluation record must be a dictionary")

    missing = REQUIRED_FIELDS - record.keys()
    extra = record.keys() - REQUIRED_FIELDS

    if missing or extra:
        raise ValueError(
            f"Record fields mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    for field in ("run_id", "case_id", "dataset_source", "notes"):
        if not isinstance(record[field], str):
            raise ValueError(f"{field} must be a string")

    for field in ("ground_truth_attack", "attack_objective_achieved"):
        value = record[field]
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{field} must be Boolean or None")

    latency = record["end_to_end_latency_ms"]
    if latency is not None:
        if isinstance(latency, bool) or not isinstance(
            latency, (int, float)
        ):
            raise ValueError("Latency must be a number or None")
        if latency < 0:
            raise ValueError("Latency cannot be negative")

    if not isinstance(record["baseline_result"], dict):
        raise ValueError("baseline_result must be a dictionary")


def save_records(
    records: list[dict[str, Any]],
    output_file: str = "artifacts/evaluation_results.json",
) -> None:
    """Validate and save evaluation records to a JSON file."""
    for record in records:
        validate_evaluation_record(record)

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} evaluation records to {path}")
    