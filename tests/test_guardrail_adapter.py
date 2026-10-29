import pytest

from common.schemas import GuardrailResponse
from target.guardrail_adapter import ModernBERTGuardrailAdapter


class FakeModernBERTGuardrail:
    """Mock the real Guardrail without loading its model files."""

    def __init__(self, action="ALLOW", malicious_score=0.1):
        self.action = action
        self.malicious_score = malicious_score
        self.last_arguments = None

    def evaluate(
        self,
        user_prompt,
        conversation_history="",
        retrieved_context="",
        tool_output="",
        memory="",
    ):
        self.last_arguments = {
            "user_prompt": user_prompt,
            "conversation_history": conversation_history,
            "retrieved_context": retrieved_context,
            "tool_output": tool_output,
            "memory": memory,
        }

        return {
            "action": self.action,
            "malicious_score": self.malicious_score,
            "benign_score": round(1.0 - self.malicious_score, 4),
            "reason": "Mock Guardrail decision.",
            "latency_ms": 1.0,
        }


@pytest.mark.parametrize("decision", ["ALLOW", "FLAG", "BLOCK"])
def test_adapter_preserves_guardrail_decision(decision):
    fake = FakeModernBERTGuardrail(action=decision)
    adapter = ModernBERTGuardrailAdapter(fake)

    result = adapter.evaluate(
        case_id="test-001",
        user_prompt="Test prompt",
    )

    assert isinstance(result, GuardrailResponse)
    assert result.case_id == "test-001"
    assert result.decision == decision
    assert result.attack_detected == (decision != "ALLOW")


def test_adapter_maps_scores_and_reason():
    fake = FakeModernBERTGuardrail(
        action="FLAG",
        malicious_score=0.65,
    )
    adapter = ModernBERTGuardrailAdapter(fake)

    result = adapter.evaluate(
        case_id="test-002",
        user_prompt="Test prompt",
    )

    assert result.classifier_score == pytest.approx(0.65)
    assert result.risk_score == pytest.approx(0.65)
    assert result.reason == "Mock Guardrail decision."
    assert result.model_version == "modernbert_guardrail"
    assert len(result.signals) == 1


def test_adapter_formats_context_for_guardrail():
    fake = FakeModernBERTGuardrail()
    adapter = ModernBERTGuardrailAdapter(fake)

    adapter.evaluate(
        case_id="test-003",
        user_prompt="Summarize this",
        conversation_history=[
            {"role": "user", "content": "Earlier message"},
            {"role": "assistant", "content": "Earlier reply"},
        ],
        rag_content=[
            {
                "source": "document.txt",
                "trust": "untrusted",
                "text": "Retrieved content",
            }
        ],
        web_content=[
            {
                "source": "website",
                "trust": "unknown",
                "text": "Web content",
            }
        ],
        tool_output=[
            {
                "source": "calculator",
                "trust": "trusted",
                "text": "2",
            }
        ],
        memory=[
            {
                "source": "memory",
                "trust": "unknown",
                "text": "Stored context",
            }
        ],
    )

    arguments = fake.last_arguments

    assert "user: Earlier message" in arguments["conversation_history"]
    assert "assistant: Earlier reply" in arguments["conversation_history"]
    assert "Retrieved content" in arguments["retrieved_context"]
    assert "Web content" in arguments["retrieved_context"]
    assert "Trust: untrusted" in arguments["retrieved_context"]
    assert "2" in arguments["tool_output"]
    assert "Stored context" in arguments["memory"]


@pytest.mark.parametrize(
    "action,score",
    [
        ("INVALID", 0.1),
        ("ALLOW", -0.1),
        ("BLOCK", 1.1),
    ],
)
def test_adapter_rejects_invalid_guardrail_output(action, score):
    fake = FakeModernBERTGuardrail(
        action=action,
        malicious_score=score,
    )
    adapter = ModernBERTGuardrailAdapter(fake)

    with pytest.raises(ValueError):
        adapter.evaluate(
            case_id="test-004",
            user_prompt="Test prompt",
        )