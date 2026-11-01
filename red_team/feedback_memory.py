
"""Member 3: Store and retrieve red-team evaluation feedback."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_FEEDBACK_PATH = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
    / "attack_feedback.jsonl"
)

# Optional evaluation details: retained only when supplied.
OPTIONAL_FIELDS = {
    "classifier_score",
    "risk_score",
    "target_response",
    "execution_status",
    "failure_info",
    "feedback_reference",
    "generation_method",
    "evaluation_run_id",
    "test_inputs",
}


def record_attack_outcome(
    feedback: dict,
    path=DEFAULT_FEEDBACK_PATH,
) -> str:
    """Validate and save one observed evaluation outcome."""

    if not isinstance(feedback, dict):
        raise ValueError("feedback must be a dictionary.")

    required_fields = {
        "case_id",
        "attack_category",
        "expected_attack",
        "guardrail_decision",
        "attack_objective_achieved",
        "notes",
    }

    if not required_fields.issubset(feedback):
        missing = required_fields - set(feedback)
        raise ValueError(f"Missing feedback fields: {sorted(missing)}")

    if not isinstance(feedback["case_id"], str) or not feedback["case_id"].strip():
        raise ValueError("case_id must be a non-empty string.")

    if (
        not isinstance(feedback["attack_category"], str)
        or not feedback["attack_category"].strip()
    ):
        raise ValueError("attack_category must be a non-empty string.")

    if not isinstance(feedback["expected_attack"], bool):
        raise ValueError("expected_attack must be a boolean.")

    decision = feedback["guardrail_decision"]
    if decision not in (None, "allow", "flag", "block"):
        raise ValueError(
            "guardrail_decision must be allow, flag, block, or None."
        )

    achieved = feedback["attack_objective_achieved"]
    if achieved is not None and not isinstance(achieved, bool):
        raise ValueError(
            "attack_objective_achieved must be true, false, or None."
        )

    if not isinstance(feedback["notes"], str):
        raise ValueError("notes must be a string.")

    record = {
        "feedback_id": str(uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": feedback["case_id"],
        "attack_category": feedback["attack_category"],
        "expected_attack": feedback["expected_attack"],
        "guardrail_decision": decision,
        "attack_objective_achieved": achieved,
        "notes": feedback["notes"],
    }

    # Keep additional evaluation details when the caller supplies them.
    for field in OPTIONAL_FIELDS:
        if field in feedback:
            record[field] = feedback[field]

    # Validate optional fields without requiring them from the caller.
    
    for field in ("classifier_score", "risk_score"):
        value = record.get(field)

        if value is not None:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ValueError(f"{field} must be a number or None.")

            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between 0.0 and 1.0.")


 
    
    valid_execution_statuses = {
        "NOT_STARTED",
        "COMPLETED",
        "FLAGGED",
        "BLOCKED",
        "ERROR",
    }

    execution_status = record.get("execution_status")
    if (
        execution_status is not None
        and execution_status not in valid_execution_statuses
    ):
        raise ValueError(
            "execution_status must be NOT_STARTED, COMPLETED, "
            "FLAGGED, BLOCKED, ERROR, or None."
        )



    
    
    if "test_inputs" in record and not isinstance(record["test_inputs"], dict):
        raise ValueError("test_inputs must be a dictionary.")


    # failure_info can contain a string or structured JSON details.
    if "failure_info" in record and not isinstance(
        record["failure_info"], (str, dict, list, type(None))
    ):
        raise ValueError("failure_info must be text, an object, a list, or None.")

    try:
        line = json.dumps(record, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Feedback contains unsupported values.") from exc

    feedback_path = Path(path)
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    with feedback_path.open("a", encoding="utf-8") as file:
        file.write(line + "\n")

    return record["feedback_id"]


def retrieve_relevant_feedback(
    attack_category: str | None = None,
    limit: int = 5,
    path=DEFAULT_FEEDBACK_PATH,
) -> list[dict]:
    """Return the newest saved outcomes, optionally by attack category."""

    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer.")

    if attack_category is not None and not isinstance(attack_category, str):
        raise ValueError("attack_category must be a string or None.")

    if limit == 0:
        return []

    feedback_path = Path(path)

    if not feedback_path.exists():
        return []

    records = []

    with feedback_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on feedback line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Feedback line {line_number} must contain an object."
                )

            if (
                attack_category is None
                or record.get("attack_category") == attack_category
            ):
                records.append(record)

    # Newest records first; apply the limit after filtering.
    records.reverse()
    return records[:limit]
