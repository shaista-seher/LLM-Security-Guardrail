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