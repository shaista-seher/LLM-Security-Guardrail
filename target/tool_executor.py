from typing import Any

from common.schemas import ActionResult


class SafeToolExecutor:
    """
    Executes allowlisted mock tools only.

    No real external actions are performed.
    """

    def __init__(self):
        self.allowed_tools = {
            "search_mock",
            "calculator_mock",
        }

    def execute(
        self,
        action_type: str,
        target: str,
        arguments: dict[str, Any],
        requires_authorization: bool,
        authorized: bool = False,
    ) -> ActionResult:
        """
        Validate and execute a permitted mock tool.

        Execution is denied when authorization is required
        but has not been granted.
        """

        if not isinstance(arguments, dict):
            return self._failure(
                action_type,
                "Tool arguments must be a dictionary.",
            )

        if action_type not in self.allowed_tools:
            return self._failure(
                action_type,
                f"Tool '{action_type}' is not allowlisted.",
            )

        if requires_authorization and not authorized:
            return self._failure(
                action_type,
                "Authorization is required but has not been granted.",
            )

        if action_type == "search_mock":
            return self._mock_search(target, arguments)

        if action_type == "calculator_mock":
            return self._mock_calculator(target, arguments)

        return self._failure(
            action_type,
            "Unsupported tool.",
        )

    def _mock_search(
        self,
        target: str,
        arguments: dict[str, Any],
    ) -> ActionResult:
        if target != "mock":
            return self._failure(
                "search_mock",
                "Search target must be 'mock'.",
            )

        if set(arguments.keys()) != {"query"}:
            return self._failure(
                "search_mock",
                "Search requires exactly one argument: 'query'.",
            )

        query = arguments["query"]

        if not isinstance(query, str) or not query.strip():
            return self._failure(
                "search_mock",
                "The search query must be a non-empty string.",
            )

        return ActionResult(
            executed=True,
            action_type="search_mock",
            result_summary=(
                f"Mock search completed for query: {query.strip()}"
            ),
            error="",
        )

    def _mock_calculator(
        self,
        target: str,
        arguments: dict[str, Any],
    ) -> ActionResult:
        if target != "mock":
            return self._failure(
                "calculator_mock",
                "Calculator target must be 'mock'.",
            )

        if set(arguments.keys()) != {"a", "b", "operation"}:
            return self._failure(
                "calculator_mock",
                "Calculator requires exactly 'a', 'b', and 'operation'.",
            )

        a = arguments["a"]
        b = arguments["b"]
        operation = arguments["operation"]

        if (
            isinstance(a, bool)
            or not isinstance(a, (int, float))
            or isinstance(b, bool)
            or not isinstance(b, (int, float))
        ):
            return self._failure(
                "calculator_mock",
                "'a' and 'b' must be numbers.",
            )

        if operation not in {"add", "subtract", "multiply"}:
            return self._failure(
                "calculator_mock",
                "Operation must be add, subtract, or multiply.",
            )

        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        else:
            result = a * b

        return ActionResult(
            executed=True,
            action_type="calculator_mock",
            result_summary=f"Mock calculation result: {result}",
            error="",
        )

    @staticmethod
    def _failure(
        action_type: str,
        error: str,
    ) -> ActionResult:
        return ActionResult(
            executed=False,
            action_type=action_type,
            result_summary="No action was executed.",
            error=error,
        )