"""Connect Member 3 attack generation to the existing M1/M2 target and M4 evaluation."""

import time

from common.schemas import (
    validate_attack_case,
    validate_guardrail_result,
    validate_target_result,
)
from evaluation.m3_m4_pipeline import (
    generate_validated_cases,
    build_evaluation_record,
    calculate_guardrail_metrics,
    save_evaluation,
)

REQUEST_FIELDS = (
    "case_id",
    "user_prompt",
    "conversation_history",
    "rag_content",
    "web_content",
    "tool_output",
    "memory",
    "requested_action",
)


def run_pipeline(limit=2, use_gemma=False, application=None):
    """Run generated cases through the real target application."""

    if application is None:
        # Lazy import: allows unit tests to inject a fake application
        # without importing PyTorch or loading the ModernBERT model.
        from target.app import TargetApplication
        application = TargetApplication()

    cases = generate_validated_cases(
        limit=limit,
        use_gemma=use_gemma,
    )

    records = []

    for case in cases:
        validate_attack_case(case)
        print("\n--- GENERATED CASE ---")
        print("Case ID:", case["case_id"])
        print("Expected attack:", case["expected_attack"])
        print("User prompt:", case["user_prompt"])
        print("RAG content:", case["rag_content"])
        print("Web content:", case["web_content"])
        print("Tool output:", case["tool_output"])
        print("Memory:", case["memory"])

        request_payload = {
            field: case[field]
            for field in REQUEST_FIELDS
        }

        start = time.perf_counter()
        response = application.handle_request(
            request_data=request_payload
        )
        latency_ms = (time.perf_counter() - start) * 1000

        target_result = response.model_dump()
        guardrail_result = target_result["guardrail_result"]

        validate_guardrail_result(guardrail_result)
        validate_target_result(target_result)

        if target_result["case_id"] != case["case_id"]:
            raise ValueError("Target result case_id does not match attack case.")

        record = build_evaluation_record(
            case=case,
            guardrail_result=guardrail_result,
            target_result=target_result,
            latency_ms=latency_ms,
        )
        records.append(record)

    metrics = calculate_guardrail_metrics(records)
    output_path = save_evaluation(records)

    return {
        "cases": cases,
        "records": records,
        "metrics": metrics,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    result = run_pipeline(use_gemma=True)

    print("Cases executed:", len(result["records"]))
    print("Metrics:", result["metrics"])
    print("Evaluation saved to:", result["output_path"])
