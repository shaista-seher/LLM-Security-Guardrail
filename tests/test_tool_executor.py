import pytest

from target.tool_executor import SafeToolExecutor


@pytest.fixture
def executor():
    return SafeToolExecutor()


def test_mock_search_executes_successfully(executor):
    result = executor.execute(
        action_type="search_mock",
        target="mock",
        arguments={"query": "machine learning"},
        requires_authorization=False,
    )

    assert result.executed is True
    assert result.action_type == "search_mock"
    assert "machine learning" in result.result_summary
    assert result.error == ""


def test_mock_calculator_adds_numbers(executor):
    result = executor.execute(
        action_type="calculator_mock",
        target="mock",
        arguments={
            "a": 10,
            "b": 5,
            "operation": "add",
        },
        requires_authorization=False,
    )

    assert result.executed is True
    assert "15" in result.result_summary
    assert result.error == ""


def test_mock_calculator_multiplies_numbers(executor):
    result = executor.execute(
        action_type="calculator_mock",
        target="mock",
        arguments={
            "a": 4,
            "b": 3,
            "operation": "multiply",
        },
        requires_authorization=False,
    )

    assert result.executed is True
    assert "12" in result.result_summary


def test_unknown_tool_is_rejected(executor):
    result = executor.execute(
        action_type="delete_files",
        target="mock",
        arguments={},
        requires_authorization=False,
    )

    assert result.executed is False
    assert "not allowlisted" in result.error


def test_unauthorized_action_is_rejected(executor):
    result = executor.execute(
        action_type="search_mock",
        target="mock",
        arguments={"query": "test"},
        requires_authorization=True,
        authorized=False,
    )

    assert result.executed is False
    assert "Authorization is required" in result.error


def test_authorized_action_can_execute(executor):
    result = executor.execute(
        action_type="search_mock",
        target="mock",
        arguments={"query": "test"},
        requires_authorization=True,
        authorized=True,
    )

    assert result.executed is True


def test_empty_search_query_is_rejected(executor):
    result = executor.execute(
        action_type="search_mock",
        target="mock",
        arguments={"query": "   "},
        requires_authorization=False,
    )

    assert result.executed is False
    assert "non-empty string" in result.error


def test_invalid_calculator_operation_is_rejected(executor):
    result = executor.execute(
        action_type="calculator_mock",
        target="mock",
        arguments={
            "a": 5,
            "b": 2,
            "operation": "divide",
        },
        requires_authorization=False,
    )

    assert result.executed is False
    assert "Operation must be" in result.error


def test_calculator_rejects_non_numeric_arguments(executor):
    result = executor.execute(
        action_type="calculator_mock",
        target="mock",
        arguments={
            "a": "five",
            "b": 2,
            "operation": "add",
        },
        requires_authorization=False,
    )

    assert result.executed is False
    assert "must be numbers" in result.error


def test_invalid_target_is_rejected(executor):
    result = executor.execute(
        action_type="search_mock",
        target="real_website",
        arguments={"query": "test"},
        requires_authorization=False,
    )

    assert result.executed is False
    assert "target must be 'mock'" in result.error


def test_unexpected_arguments_are_rejected(executor):
    result = executor.execute(
        action_type="search_mock",
        target="mock",
        arguments={
            "query": "test",
            "execute_real_search": True,
        },
        requires_authorization=False,
    )

    assert result.executed is False
    assert "exactly one argument" in result.error