from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# Reject unexpected fields instead of silently accepting
# incompatible data from another module.
class StrictSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------
# 1. Conversation history item
# --------------------------------------------------

class ConversationMessage(StrictSchema):
    role: str
    content: str


# --------------------------------------------------
# 2. RAG, web, tool-output, and memory context item
# --------------------------------------------------

class ContextItem(StrictSchema):
    source: str
    trust: str
    text: str


# --------------------------------------------------
# 3. Requested action
# --------------------------------------------------

class RequestedAction(StrictSchema):
    action_type: str
    target: str
    arguments: dict[str, Any]
    requires_authorization: bool = Field(strict=True)


# --------------------------------------------------
# 4. Shared request schema: exactly 8 fields
# --------------------------------------------------

class RequestSchema(StrictSchema):
    case_id: str
    user_prompt: str
    conversation_history: list[ConversationMessage]
    rag_content: list[ContextItem]
    web_content: list[ContextItem]
    tool_output: list[ContextItem]
    memory: list[ContextItem]
    requested_action: RequestedAction


# --------------------------------------------------
# 5. Guardrail signal
# --------------------------------------------------

class GuardrailSignal(StrictSchema):
    signal_name: str
    source: str
    severity: str
    details: str


# --------------------------------------------------
# 6. Guardrail response: exactly 8 fields
# --------------------------------------------------

class GuardrailResponse(StrictSchema):
    case_id: str
    attack_detected: bool

    classifier_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)

    decision: Literal["ALLOW", "FLAG", "BLOCK"]
    reason: str
    signals: list[GuardrailSignal]
    model_version: str


# --------------------------------------------------
# 7. Action execution result
# --------------------------------------------------

class ActionResult(StrictSchema):
    executed: bool
    action_type: str
    result_summary: str
    error: str


# --------------------------------------------------
# 8. Target application response: exactly 5 fields
# --------------------------------------------------

class TargetApplicationResponse(StrictSchema):
    case_id: str
    response: str
    guardrail_result: GuardrailResponse

    execution_status: Literal[
        "NOT_STARTED",
        "COMPLETED",
        "FLAGGED",
        "BLOCKED",
        "ERROR",
    ]

    action_result: ActionResult
# --------------------------------------------------
# Member 4: Evaluation result validation
# --------------------------------------------------

def _require_fields(data: dict, expected: set[str], name: str) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"{name} must be a dictionary.")

    actual = set(data.keys())
    if actual != expected:
        raise ValueError(
            f"{name} fields mismatch. "
            f"Missing: {expected - actual}; "
            f"Unexpected: {actual - expected}"
        )


def validate_guardrail_result(data: dict) -> None:
    """Validate the exact eight-field Guardrail response."""
    expected = {
        "case_id", "attack_detected", "classifier_score", "risk_score",
        "decision", "reason", "signals", "model_version",
    }
    _require_fields(data, expected, "guardrail result")

    if not isinstance(data["case_id"], str):
        raise ValueError("case_id must be a string.")

    if not isinstance(data["attack_detected"], bool):
        raise ValueError("attack_detected must be Boolean.")

    for field in ("classifier_score", "risk_score"):
        score = data[field]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{field} must be a number.")
        if not 0 <= score <= 1:
            raise ValueError(f"{field} must be between 0 and 1.")

    if data["decision"] not in ("ALLOW", "FLAG", "BLOCK"):
        raise ValueError("decision must be ALLOW, FLAG, or BLOCK.")

    for field in ("reason", "model_version"):
        if not isinstance(data[field], str):
            raise ValueError(f"{field} must be a string.")

    if not isinstance(data["signals"], list):
        raise ValueError("signals must be a list.")

    for signal in data["signals"]:
        signal_fields = {"signal_name", "source", "severity", "details"}
        _require_fields(signal, signal_fields, "signal")
        for field in signal_fields:
            if not isinstance(signal[field], str):
                raise ValueError(f"signal.{field} must be a string.")


def validate_target_result(data: dict) -> None:
    """Validate the exact five-field target application response."""
    expected = {
        "case_id", "response", "guardrail_result",
        "execution_status", "action_result",
    }
    _require_fields(data, expected, "target result")

    if not isinstance(data["case_id"], str):
        raise ValueError("case_id must be a string.")

    if not isinstance(data["response"], str):
        raise ValueError("response must be a string.")

    validate_guardrail_result(data["guardrail_result"])

    if data["execution_status"] not in {
        "NOT_STARTED", "COMPLETED", "FLAGGED", "BLOCKED", "ERROR"
    }:
        raise ValueError("invalid execution_status.")

    action = data["action_result"]
    action_fields = {
        "executed", "action_type", "result_summary", "error"
    }
    _require_fields(action, action_fields, "action_result")

    if not isinstance(action["executed"], bool):
        raise ValueError("action_result.executed must be Boolean.")

    for field in ("action_type", "result_summary", "error"):
        if not isinstance(action[field], str):
            raise ValueError(f"action_result.{field} must be a string.")


# --------------------------------------------------
# Member 3: Attack-case validation
# --------------------------------------------------

TRUST_VALUES = {"trusted", "untrusted", "unknown"}

REQUEST_FIELDS = {
    "case_id",
    "user_prompt",
    "conversation_history",
    "rag_content",
    "web_content",
    "tool_output",
    "memory",
    "requested_action",
}

ACTION_FIELDS = {
    "action_type",
    "target",
    "arguments",
    "requires_authorization",
}

CONTENT_ITEM_FIELDS = {"source", "trust", "text"}
HISTORY_ITEM_FIELDS = {"role", "content"}

ATTACK_FIELDS = REQUEST_FIELDS | {
    "attack_category",
    "expected_attack",
    "intended_malicious_objective",
    "generation_method",
    "feedback_reference",
}


def _validate_content_items(items: Any, name: str) -> None:
    if not isinstance(items, list):
        raise TypeError(f"{name} must be a list.")

    for index, item in enumerate(items):
        _require_fields(item, CONTENT_ITEM_FIELDS, f"{name}[{index}]")

        if not all(
            isinstance(item[key], str) for key in CONTENT_ITEM_FIELDS
        ):
            raise TypeError(f"{name}[{index}] values must be strings.")

        if item["trust"] not in TRUST_VALUES:
            raise ValueError(f"{name}[{index}].trust is invalid.")


def validate_request(data: dict) -> None:
    """Validate the eight required request fields."""
    _require_fields(data, REQUEST_FIELDS, "request")

    for field in ("case_id", "user_prompt"):
        if not isinstance(data[field], str):
            raise TypeError(f"{field} must be a string.")

    history = data["conversation_history"]
    if not isinstance(history, list):
        raise TypeError("conversation_history must be a list.")

    for index, item in enumerate(history):
        _require_fields(
            item, HISTORY_ITEM_FIELDS, f"conversation_history[{index}]"
        )
        if not all(
            isinstance(item[key], str) for key in HISTORY_ITEM_FIELDS
        ):
            raise TypeError(
                f"conversation_history[{index}] values must be strings."
            )

    for field in ("rag_content", "web_content", "tool_output", "memory"):
        _validate_content_items(data[field], field)

    action = data["requested_action"]
    _require_fields(action, ACTION_FIELDS, "requested_action")

    if not isinstance(action["action_type"], str):
        raise TypeError("requested_action.action_type must be a string.")
    if not isinstance(action["target"], str):
        raise TypeError("requested_action.target must be a string.")
    if not isinstance(action["arguments"], dict):
        raise TypeError("requested_action.arguments must be a dictionary.")
    if not isinstance(action["requires_authorization"], bool):
        raise TypeError(
            "requested_action.requires_authorization must be a boolean."
        )


def validate_attack_case(data: dict) -> None:
    """Validate a request plus the five red-team attack fields."""
    _require_fields(data, ATTACK_FIELDS, "attack case")

    request = {key: data[key] for key in REQUEST_FIELDS}
    validate_request(request)

    for field in (
        "attack_category",
        "intended_malicious_objective",
        "generation_method",
    ):
        if not isinstance(data[field], str):
            raise TypeError(f"{field} must be a string.")

    if not isinstance(data["expected_attack"], bool):
        raise TypeError("expected_attack must be a boolean.")

    if data["feedback_reference"] is not None and not isinstance(
        data["feedback_reference"], str
    ):
        raise TypeError("feedback_reference must be a string or None.")
