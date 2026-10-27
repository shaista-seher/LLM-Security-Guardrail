from common.schemas import GuardrailResponse


class DummyGuardrail:
    """
    Temporary Guardrail for testing application control flow.

    This component returns predefined decisions.
    It does not detect real prompt injection attacks.
    """

    def __init__(self, decision: str = "ALLOW"):
        if decision not in {"ALLOW", "FLAG", "BLOCK"}:
            raise ValueError(f"Invalid Guardrail decision: {decision}")

        self.decision = decision

    def evaluate(self, case_id: str) -> GuardrailResponse:
        return GuardrailResponse(
            case_id=case_id,
            attack_detected=self.decision != "ALLOW",
            classifier_score=0.1,
            risk_score=0.1,
            decision=self.decision,
            reason=f"Dummy Guardrail decision: {self.decision}",
            signals=[],
            model_version="dummy-test",
        )