from guardrail import ModernBERTGuardrail
from target.app import TargetApplication
from target.guardrail_adapter import ModernBERTGuardrailAdapter


def test_target_application_with_real_qwen():
    real_model = ModernBERTGuardrail()
    real_guardrail = ModernBERTGuardrailAdapter(real_model)

    app = TargetApplication(guardrail=real_guardrail)

    result = app.handle_request(
        case_id="live-test-001",
        user_prompt="What is the capital of France?",
        requested_action={
            "action_type": "none",
            "target": "test",
            "arguments": {},
            "requires_authorization": False,
        },
    )

    print("\nGuardrail model:", result.guardrail_result.model_version)
    print("Generated response:", result.response)
    print("Execution status:", result.execution_status)
    print("Guardrail decision:", result.guardrail_result.decision)

    assert result.guardrail_result.model_version == "modernbert_guardrail"
    assert result.guardrail_result.decision == "ALLOW"
    assert result.execution_status == "COMPLETED"
    assert isinstance(result.response, str)
    assert len(result.response) > 0
    assert result.action_result.executed is False
