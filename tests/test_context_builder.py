
import pytest

from target.context_builder import ContextBuilder


@pytest.fixture
def valid_request():
    return {
        "case_id": "case-001",
        "user_prompt": "Summarize the supplied information.",
        "conversation_history": [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ],
        "rag_content": [
            {
                "source": "local_document",
                "trust": "trusted",
                "text": "Information from a local document.",
            }
        ],
        "web_content": [
            {
                "source": "external_website",
                "trust": "untrusted",
                "text": "Information from an external website.",
            }
        ],
        "tool_output": [
            {
                "source": "mock_search_tool",
                "trust": "unknown",
                "text": "Mock search result.",
            }
        ],
        "memory": [
            {
                "source": "application_memory",
                "trust": "unknown",
                "text": "Stored context.",
            }
        ],
        "requested_action": {
            "action_type": "none",
            "target": "",
            "arguments": {},
            "requires_authorization": False,
        },
    }


def test_build_valid_request(valid_request):
    builder = ContextBuilder()

    result = builder.build(valid_request)

    assert result["case_id"] == "case-001"
    assert result["user_prompt"] == valid_request["user_prompt"]
    assert len(result["conversation_history"]) == 2


def test_preserves_context_provenance(valid_request):
    builder = ContextBuilder()

    result = builder.build(valid_request)

    assert result["rag_content"][0]["trust"] == "trusted"
    assert result["web_content"][0]["trust"] == "untrusted"
    assert result["tool_output"][0]["source"] == "mock_search_tool"
    assert result["memory"][0]["source"] == "application_memory"


def test_context_categories_remain_separate(valid_request):
    result = ContextBuilder().build(valid_request)

    assert "external_website" not in result["rag_content"][0]["source"]
    assert result["web_content"][0]["source"] == "external_website"
    assert isinstance(result["tool_output"], list)
    assert isinstance(result["memory"], list)


def test_rejects_missing_required_field(valid_request):
    del valid_request["user_prompt"]

    with pytest.raises(ValueError, match="Invalid request data"):
        ContextBuilder().build(valid_request)


def test_rejects_unexpected_field(valid_request):
    valid_request["unexpected_field"] = "not allowed"

    with pytest.raises(ValueError, match="Invalid request data"):
        ContextBuilder().build(valid_request)


def test_rejects_empty_prompt(valid_request):
    valid_request["user_prompt"] = "   "

    with pytest.raises(ValueError, match="cannot be empty"):
        ContextBuilder().build(valid_request)


def test_rejects_oversized_prompt(valid_request):
    builder = ContextBuilder(max_prompt_chars=10)

    with pytest.raises(ValueError, match="character limit"):
        builder.build(valid_request)


def test_limits_context_text(valid_request):
    valid_request["web_content"][0]["text"] = "A" * 100

    builder = ContextBuilder(max_context_chars_per_item=20)
    result = builder.build(valid_request)

    assert len(result["web_content"][0]["text"]) == 20
    assert result["web_content"][0]["trust"] == "untrusted"


def test_limits_context_item_count(valid_request):
    valid_request["memory"] = [
        {
            "source": f"source-{i}",
            "trust": "unknown",
            "text": f"Memory item {i}",
        }
        for i in range(5)
    ]

    builder = ContextBuilder(max_context_items_per_source=2)
    result = builder.build(valid_request)

    assert len(result["memory"]) == 2


def test_limits_conversation_history(valid_request):
    valid_request["conversation_history"] = [
        {"role": "user", "content": f"Message {i}"}
        for i in range(5)
    ]

    builder = ContextBuilder(max_history_items=2)
    result = builder.build(valid_request)

    assert len(result["conversation_history"]) == 2
    assert result["conversation_history"][0]["content"] == "Message 3"
    assert result["conversation_history"][1]["content"] == "Message 4"


def test_rejects_invalid_requested_action(valid_request):
    valid_request["requested_action"]["requires_authorization"] = "false"

    with pytest.raises(ValueError, match="Invalid request data"):
        ContextBuilder().build(valid_request)


def test_rejects_empty_case_id(valid_request):
    valid_request["case_id"] = " "

    with pytest.raises(ValueError, match="case_id cannot be empty"):
        ContextBuilder().build(valid_request)
def test_rejects_invalid_trust_value(valid_request):
    valid_request["web_content"][0]["trust"] = "maybe_trusted"

    with pytest.raises(ValueError, match="Invalid trust value"):
        ContextBuilder().build(valid_request)


def test_normalizes_trust_value(valid_request):
    valid_request["web_content"][0]["trust"] = " UNTRUSTED "

    result = ContextBuilder().build(valid_request)

    assert result["web_content"][0]["trust"] == "untrusted"    

def test_estimates_token_count():
    builder = ContextBuilder()

    assert builder.estimate_tokens("") == 0
    assert builder.estimate_tokens("test") == 1
    assert builder.estimate_tokens("12345678") == 2

def test_context_respects_token_budget(valid_request):
    builder = ContextBuilder(max_context_tokens=10)

    result = builder.build(valid_request)

    total_text = result["user_prompt"]
    total_text += "".join(
        message["content"]
        for message in result["conversation_history"]
    )

    for field_name in ("rag_content", "web_content", "tool_output", "memory"):
        total_text += "".join(
            item["text"] for item in result[field_name]
        )

    assert builder.estimate_tokens(total_text) <= 10


def test_rejects_nonpositive_token_budget():
    with pytest.raises(ValueError, match="max_context_tokens must be positive"):
        ContextBuilder(max_context_tokens=0)