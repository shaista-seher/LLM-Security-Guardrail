from target.app import TargetApplication
from target.dummy_guardrail import DummyGuardrail


def test_target_application_with_real_qwen():
    app = TargetApplication(
        guardrail=DummyGuardrail("ALLOW")
    )

    result = app.handle_request(
        case_id="live-test-001",
        user_prompt="Reply with exactly: Target application connected to Qwen",
        requested_action={
            "action_type": "none",
            "target": "test",
            "arguments": {},
            "requires_authorization": False,
        },
    )

    print("\nGenerated response:", result.response)
    print("Execution status:", result.execution_status)
    print("Guardrail decision:", result.guardrail_result.decision)

    assert result.execution_status == "COMPLETED"
    assert result.guardrail_result.decision == "ALLOW"
    assert isinstance(result.response, str)
    assert len(result.response) > 0
    assert result.action_result.executed is False