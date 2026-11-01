import pytest

from common.schemas import (
    ActionResult,
    GuardrailResponse,
    GuardrailSignal,
    TargetApplicationResponse,
)
from evaluation.run_m3_m4 import run_pipeline


class MockTargetApplication:
    def handle_request(self, request_data):
        case_id = request_data["case_id"]
        is_attack = case_id == "RT-DUMMY-001"

        guardrail = GuardrailResponse(
            case_id=case_id,
            attack_detected=is_attack,
            classifier_score=0.99 if is_attack else 0.05,
            risk_score=0.99 if is_attack else 0.05,
            decision="BLOCK" if is_attack else "ALLOW",
            reason="Synthetic integration test result.",
            signals=[
                GuardrailSignal(
                    signal_name="mock_signal",
                    source="mock_test",
                    severity="high" if is_attack else "low",
                    details="Synthetic test signal.",
                )
            ],
            model_version="mock-test-only",
        )

        return TargetApplicationResponse(
            case_id=case_id,
            response="Mock response.",
            guardrail_result=guardrail,
            execution_status="BLOCKED" if is_attack else "COMPLETED",
            action_result=ActionResult(
                executed=False,
                action_type=request_data["requested_action"]["action_type"],
                result_summary="No action executed in test.",
                error="",
            ),
        )


def test_m3_m4_pipeline_with_mock_application():
    result = run_pipeline(
        limit=2,
        use_gemma=False,
        application=MockTargetApplication(),
    )

    assert len(result["cases"]) == 2
    assert len(result["records"]) == 2
    assert result["metrics"]["true_positives"] == 1
    assert result["metrics"]["true_negatives"] == 1
    assert result["metrics"]["false_positives"] == 0
    assert result["metrics"]["false_negatives"] == 0
    assert result["output_path"]


def test_pipeline_preserves_case_ids():
    result = run_pipeline(
        limit=2,
        use_gemma=False,
        application=MockTargetApplication(),
    )

    case_ids = [case["case_id"] for case in result["cases"]]
    record_ids = [record["case_id"] for record in result["records"]]

    assert case_ids == record_ids


def test_pipeline_rejects_mismatched_case_id():
    class MismatchedApplication:
        def handle_request(self, request_data):
            guardrail = GuardrailResponse(
                case_id="WRONG-ID",
                attack_detected=False,
                classifier_score=0.05,
                risk_score=0.05,
                decision="ALLOW",
                reason="Test result.",
                signals=[],
                model_version="mock-test-only",
            )

            return TargetApplicationResponse(
                case_id="WRONG-ID",
                response="Mock response.",
                guardrail_result=guardrail,
                execution_status="COMPLETED",
                action_result=ActionResult(
                    executed=False,
                    action_type="",
                    result_summary="No action executed.",
                    error="",
                ),
            )

    with pytest.raises(ValueError, match="case_id"):
        run_pipeline(
            limit=1,
            use_gemma=False,
            application=MismatchedApplication(),
        )
