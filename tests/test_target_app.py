from unittest.mock import Mock

import pytest

from target.app import TargetApplication
from common.schemas import GuardrailResponse
from target.guardrail_adapter import ModernBERTGuardrailAdapter

class StubModernBERT:
    """Controlled test double for testing application behavior."""

    def __init__(self, decision="ALLOW"):
        if decision not in {"ALLOW", "FLAG", "BLOCK"}:
            raise ValueError("Invalid guardrail decision")
        self.decision = decision

    def evaluate(
        self,
        user_prompt,
        conversation_history="",
        retrieved_context="",
        tool_output="",
        memory="",
    ):
        score = {
            "ALLOW": 0.0,
            "FLAG": 0.6,
            "BLOCK": 1.0,
        }[self.decision]

        return {
            "action": self.decision,
            "malicious_score": score,
            "reason": "Controlled test decision",
            "latency_ms": 1.0,
        }


def make_test_guardrail(decision="ALLOW"):
    return ModernBERTGuardrailAdapter(StubModernBERT(decision))

@pytest.fixture
def requested_action():
    return {
        "action_type": "none",
        "target": "test",
        "arguments": {},
        "requires_authorization": False,
    }


def test_allow_generates_model_response(requested_action):
    model_client = Mock()
    model_client.generate.return_value = "Qwen generated response."

    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
    )

    result = app.handle_request(
        case_id="test-allow",
        user_prompt="Hello",
        requested_action=requested_action,
    )

    assert result.execution_status == "COMPLETED"
    assert result.guardrail_result.decision == "ALLOW"
    assert result.response == "Qwen generated response."
    assert result.action_result.executed is False

    model_client.generate.assert_called_once_with("Hello")


@pytest.mark.parametrize("decision", ["FLAG", "BLOCK"])
def test_flag_and_block_do_not_call_model(decision, requested_action):
    model_client = Mock()

    app = TargetApplication(
        guardrail=make_test_guardrail(decision),
        model_client=model_client,
    )

    result = app.handle_request(
        case_id=f"test-{decision.lower()}",
        user_prompt="Test request",
        requested_action=requested_action,
    )

    expected_status = "FLAGGED" if decision == "FLAG" else "BLOCKED"

    assert result.execution_status == expected_status
    assert result.guardrail_result.decision == decision
    assert result.action_result.executed is False

    model_client.generate.assert_not_called()


def test_model_failure_returns_error(requested_action):
    model_client = Mock()
    model_client.generate.side_effect = RuntimeError("Model unavailable")

    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
    )

    result = app.handle_request(
        case_id="test-model-error",
        user_prompt="Hello",
        requested_action=requested_action,
    )

    assert result.execution_status == "ERROR"
    assert result.response == "Unable to generate a response."
    assert result.action_result.executed is False
    assert result.action_result.error == "Model unavailable"


def test_invalid_guardrail_decision_is_rejected():
    with pytest.raises(ValueError):
        make_test_guardrail("INVALID")

def test_accepts_complete_shared_request():
    model_client = Mock()
    model_client.generate.return_value = "Complete request accepted."

    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
    )

    request_data = {
        "case_id": "full-request-001",
        "user_prompt": "Summarize the information.",
        "conversation_history": [
            {"role": "user", "content": "Previous question"}
        ],
        "rag_content": [
            {
                "source": "local_document",
                "trust": "trusted",
                "text": "Local information.",
            }
        ],
        "web_content": [],
        "tool_output": [],
        "memory": [],
        "requested_action": {
            "action_type": "none",
            "target": "",
            "arguments": {},
            "requires_authorization": False,
        },
    }

    result = app.handle_request(request_data=request_data)

    assert result.case_id == "full-request-001"
    assert result.response == "Complete request accepted."
    assert result.execution_status == "COMPLETED"
    assert result.action_result.executed is False

    model_client.generate.assert_called_once()

    generated_prompt = model_client.generate.call_args.args[0]

    assert "Summarize the information." in generated_prompt
    assert "Previous question" in generated_prompt
    assert "local_document" in generated_prompt
    assert "Local information." in generated_prompt
    assert "trusted" in generated_prompt

def test_blocked_request_does_not_execute_tool():
    from unittest.mock import Mock

    tool_executor = Mock()
    app = TargetApplication(
        guardrail=make_test_guardrail("BLOCK"),
        model_client=Mock(),
        tool_executor=tool_executor,
    )

    result = app.handle_request(
        case_id="blocked-tool",
        user_prompt="Search for something",
        requested_action={
            "action_type": "search_mock",
            "target": "mock",
            "arguments": {"query": "test"},
            "requires_authorization": False,
        },
    )

    assert result.execution_status == "BLOCKED"
    assert result.action_result.executed is False
    tool_executor.execute.assert_not_called()


def test_flagged_request_does_not_execute_tool():
    from unittest.mock import Mock

    tool_executor = Mock()
    app = TargetApplication(
        guardrail=make_test_guardrail("FLAG"),
        model_client=Mock(),
        tool_executor=tool_executor,
    )

    result = app.handle_request(
        case_id="flagged-tool",
        user_prompt="Search for something",
        requested_action={
            "action_type": "search_mock",
            "target": "mock",
            "arguments": {"query": "test"},
            "requires_authorization": False,
        },
    )

    assert result.execution_status == "FLAGGED"
    assert result.action_result.executed is False
    tool_executor.execute.assert_not_called()


def test_unauthorized_tool_request_is_rejected():
    from unittest.mock import Mock

    model_client = Mock()
    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
    )

    result = app.handle_request(
        case_id="unauthorized-tool",
        user_prompt="Search for something",
        requested_action={
            "action_type": "search_mock",
            "target": "mock",
            "arguments": {"query": "test"},
            "requires_authorization": True,
        },
    )

    assert result.execution_status == "ERROR"
    assert result.action_result.executed is False
    assert "Authorization is required" in result.action_result.error
    model_client.generate.assert_not_called()


def test_invalid_tool_name_is_rejected():
    from unittest.mock import Mock

    model_client = Mock()
    tool_executor = Mock()

    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
        tool_executor=tool_executor,
    )

    result = app.handle_request(
        case_id="invalid-tool",
        user_prompt="Run a tool",
        requested_action={
            "action_type": "unknown_tool",
            "target": "external",
            "arguments": {},
            "requires_authorization": False,
        },
    )

    assert result.execution_status == "ERROR"
    assert result.action_result.executed is False
    assert "Unsupported action type" in result.action_result.error

    model_client.generate.assert_not_called()
    tool_executor.execute.assert_not_called()


def test_successful_mock_tool_is_reported():
    from unittest.mock import Mock

    model_client = Mock()
    model_client.generate.return_value = "The calculation is complete."

    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
    )

    result = app.handle_request(
        case_id="calculator-tool",
        user_prompt="Calculate 2 plus 3",
        requested_action={
            "action_type": "calculator_mock",
            "target": "mock",
            "arguments": {
                "a": 2,
                "b": 3,
                "operation": "add",
            },
            "requires_authorization": False,
        },
    )

    assert result.execution_status == "COMPLETED"
    assert result.action_result.executed is True
    assert result.action_result.action_type == "calculator_mock"
    assert result.action_result.result_summary == "Mock calculation result: 5"
    assert result.response == "The calculation is complete."


def test_application_response_has_expected_fields():
    from unittest.mock import Mock

    model_client = Mock()
    model_client.generate.return_value = "Test response."

    app = TargetApplication(
        guardrail=make_test_guardrail("ALLOW"),
        model_client=model_client,
    )

    result = app.handle_request(
        case_id="response-schema",
        user_prompt="Hello",
        requested_action={
            "action_type": "none",
            "target": "",
            "arguments": {},
            "requires_authorization": False,
        },
    )

    assert set(result.model_dump().keys()) == {
        "case_id",
        "response",
        "guardrail_result",
        "execution_status",
        "action_result",
    }