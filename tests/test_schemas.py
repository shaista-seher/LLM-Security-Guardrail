import pytest
from pydantic import ValidationError

from common.schemas import (
    ActionResult,
    ContextItem,
    ConversationMessage,
    GuardrailResponse,
    GuardrailSignal,
    RequestSchema,
    RequestedAction,
    TargetApplicationResponse,
)


def make_valid_request():
    return {
        "case_id": "test-001",
        "user_prompt": "Summarize the provided document.",
        "conversation_history": [
            {
                "role": "user",
                "content": "Summarize the provided document.",
            }
        ],
        "rag_content": [
            {
                "source": "test_document",
                "trust": "untrusted",
                "text": "This is sample document content.",
            }
        ],
        "web_content": [],
        "tool_output": [],
        "memory": [],
        "requested_action": {
            "action_type": "",
            "target": "",
            "arguments": {},
            "requires_authorization": False,
        },
    }


def make_valid_guardrail():
    return {
        "case_id": "test-001",
        "attack_detected": False,
        "classifier_score": 0.1,
        "risk_score": 0.1,
        "decision": "ALLOW",
        "reason": "Dummy test response.",
        "signals": [],
        "model_version": "dummy-test",
    }


def test_valid_request():
    request = RequestSchema.model_validate(make_valid_request())

    assert request.case_id == "test-001"
    assert request.user_prompt == "Summarize the provided document."
    assert len(request.conversation_history) == 1
    assert len(request.rag_content) == 1
    assert request.web_content == []
    assert request.tool_output == []
    assert request.memory == []


def test_request_has_exactly_eight_fields():
    request = RequestSchema.model_validate(make_valid_request())

    assert len(request.model_dump()) == 8


def test_request_rejects_missing_required_field():
    data = make_valid_request()
    del data["user_prompt"]

    with pytest.raises(ValidationError):
        RequestSchema.model_validate(data)


def test_request_rejects_unexpected_field():
    data = make_valid_request()
    data["unexpected_field"] = "not allowed"

    with pytest.raises(ValidationError):
        RequestSchema.model_validate(data)


def test_request_rejects_wrong_collection_type():
    data = make_valid_request()
    data["rag_content"] = "This should be a list"

    with pytest.raises(ValidationError):
        RequestSchema.model_validate(data)


def test_request_rejects_wrong_authorization_type():
    data = make_valid_request()
    data["requested_action"]["requires_authorization"] = "yes"

    with pytest.raises(ValidationError):
        RequestSchema.model_validate(data)


def test_guardrail_response():
    result = GuardrailResponse.model_validate(make_valid_guardrail())

    assert result.decision == "ALLOW"
    assert result.classifier_score == 0.1
    assert result.risk_score == 0.1
    assert len(result.model_dump()) == 8


@pytest.mark.parametrize(
    "field,value",
    [
        ("classifier_score", -0.1),
        ("classifier_score", 1.1),
        ("risk_score", -0.1),
        ("risk_score", 1.1),
    ],
)
def test_guardrail_rejects_invalid_scores(field, value):
    data = make_valid_guardrail()
    data[field] = value

    with pytest.raises(ValidationError):
        GuardrailResponse.model_validate(data)


@pytest.mark.parametrize("decision", ["ALLOW", "FLAG", "BLOCK"])
def test_guardrail_accepts_all_documented_decisions(decision):
    data = make_valid_guardrail()
    data["decision"] = decision

    result = GuardrailResponse.model_validate(data)

    assert result.decision == decision


def test_guardrail_rejects_unknown_decision():
    data = make_valid_guardrail()
    data["decision"] = "IGNORE"

    with pytest.raises(ValidationError):
        GuardrailResponse.model_validate(data)


def test_target_response_has_exactly_five_fields():
    response = TargetApplicationResponse.model_validate(
        {
            "case_id": "test-001",
            "response": "This is a dummy response.",
            "guardrail_result": make_valid_guardrail(),
            "execution_status": "COMPLETED",
            "action_result": {
                "executed": False,
                "action_type": "",
                "result_summary": "No action executed.",
                "error": "",
            },
        }
    )

    assert len(response.model_dump()) == 5
    assert response.case_id == "test-001"
    assert response.action_result.executed is False


def test_target_response_rejects_invalid_execution_status():
    data = {
        "case_id": "test-001",
        "response": "Dummy response.",
        "guardrail_result": make_valid_guardrail(),
        "execution_status": "SUCCESS",
        "action_result": {
            "executed": False,
            "action_type": "",
            "result_summary": "No action executed.",
            "error": "",
        },
    }

    with pytest.raises(ValidationError):
        TargetApplicationResponse.model_validate(data)


def test_guardrail_signal_schema():
    signal = GuardrailSignal.model_validate(
        {
            "signal_name": "test_signal",
            "source": "rag_content",
            "severity": "low",
            "details": "Dummy signal for schema testing.",
        }
    )

    assert signal.signal_name == "test_signal"


def test_context_item_schema():
    item = ContextItem.model_validate(
        {
            "source": "test_source",
            "trust": "untrusted",
            "text": "Sample content.",
        }
    )

    assert item.trust == "untrusted"


def test_conversation_message_schema():
    message = ConversationMessage.model_validate(
        {
            "role": "user",
            "content": "Hello",
        }
    )

    assert message.role == "user"


def test_action_result_schema():
    result = ActionResult.model_validate(
        {
            "executed": False,
            "action_type": "",
            "result_summary": "No action executed.",
            "error": "",
        }
    )

    assert result.executed is False


def test_requested_action_schema():
    action = RequestedAction.model_validate(
        {
            "action_type": "search",
            "target": "mock_search",
            "arguments": {"query": "example"},
            "requires_authorization": False,
        }
    )

    assert action.action_type == "search"
    assert action.arguments == {"query": "example"}