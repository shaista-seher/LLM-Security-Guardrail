
"""Connect M3 attack generation to M1/M2 and M4 evaluation."""

import json
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
    calculate_baseline_metrics,
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


def build_baseline_input(case):
    """Combine the request and supplied context for baseline evaluation."""
    return json.dumps(
        {
            "user_prompt": case["user_prompt"],
            "conversation_history": case["conversation_history"],
            "rag_content": case["rag_content"],
            "web_content": case["web_content"],
            "tool_output": case["tool_output"],
            "memory": case["memory"],
        },
        ensure_ascii=False,
    )


def run_pipeline(
    limit=2,
    use_gemma=False,
    application=None,
    use_baseline=False,
):
    """Run M3 cases through M1/M2 and optionally compare a baseline."""

    if application is None:
        from target.app import TargetApplication
        application = TargetApplication()

    baseline = None
    if use_baseline:
        from evaluation.baseline import BaselineDetector
        baseline = BaselineDetector()

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
            raise ValueError(
                "Target result case_id does not match attack case."
            )

        baseline_result = {}
        if baseline is not None:
            baseline_result = baseline.predict(
                build_baseline_input(case)
            )

        record = build_evaluation_record(
            case=case,
            guardrail_result=guardrail_result,
            target_result=target_result,
            latency_ms=latency_ms,
            baseline_result=baseline_result,
        )
        records.append(record)

    metrics = calculate_guardrail_metrics(records)

    baseline_metrics = None
    if use_baseline:
        baseline_metrics = calculate_baseline_metrics(records)

    output_path = save_evaluation(records)

    return {
        "cases": cases,
        "records": records,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    result = run_pipeline(
        limit=2,
        use_gemma=True,
        use_baseline=False,
    )

    print("Cases executed:", len(result["records"]))
    print("Metrics:", result["metrics"])
    print("Evaluation saved to:", result["output_path"])
