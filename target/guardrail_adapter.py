from typing import Any

from common.schemas import GuardrailResponse, GuardrailSignal


class ModernBERTGuardrailAdapter:
    """Adapt Member 1's ModernBERT Guardrail to the shared response schema."""

    def __init__(self, guardrail: Any):
        if guardrail is None or not callable(getattr(guardrail, "evaluate", None)):
            raise ValueError("A valid ModernBERT Guardrail instance is required.")

        self.guardrail = guardrail

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> str:
        """Format validated conversation history for the Guardrail."""
        return "\n".join(
            f"{message['role']}: {message['content']}"
            for message in messages
        )

    @staticmethod
    def _format_context(items: list[dict[str, Any]]) -> str:
        """Preserve source and trust metadata when formatting context."""
        return "\n".join(
            f"Source: {item['source']} | Trust: {item['trust']}\n"
            f"{item['text']}"
            for item in items
        )

    def evaluate(
        self,
        case_id: str,
        user_prompt: str,
        conversation_history: list[dict[str, Any]] | None = None,
        rag_content: list[dict[str, Any]] | None = None,
        web_content: list[dict[str, Any]] | None = None,
        tool_output: list[dict[str, Any]] | None = None,
        memory: list[dict[str, Any]] | None = None,
    ) -> GuardrailResponse:
        """Evaluate the request and convert the result to the shared schema."""

        conversation_history = conversation_history or []
        rag_content = rag_content or []
        web_content = web_content or []
        tool_output = tool_output or []
        memory = memory or []

        retrieved_context = "\n\n".join(
            section
            for section in (
                self._format_context(rag_content),
                self._format_context(web_content),
            )
            if section
        )

        tool_output_text = self._format_context(tool_output)
        memory_text = self._format_context(memory)

        result = self.guardrail.evaluate(
            user_prompt=user_prompt,
            conversation_history=self._format_messages(conversation_history),
            retrieved_context=retrieved_context,
            tool_output=tool_output_text,
            memory=memory_text,
        )

        action = result["action"]
        malicious_score = float(result["malicious_score"])

        if action not in {"ALLOW", "FLAG", "BLOCK"}:
            raise ValueError(f"Unexpected Guardrail decision: {action}")

        if not 0.0 <= malicious_score <= 1.0:
            raise ValueError("Guardrail malicious_score must be between 0 and 1.")

        return GuardrailResponse(
            case_id=case_id,
            attack_detected=action != "ALLOW",
            classifier_score=malicious_score,
            risk_score=malicious_score,
            decision=action,
            reason=result["reason"],
            signals=[
                GuardrailSignal(
                    signal_name="modernbert_maliciousness_score",
                    source="ModernBERT",
                    severity=action,
                    details=(
                        f"Malicious score: {malicious_score:.4f}. "
                        f"Evaluation latency: {result.get('latency_ms', 'unknown')} ms."
                    ),
                )
            ],
            model_version="modernbert_guardrail",
        )