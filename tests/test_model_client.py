
import requests
import pytest
from unittest.mock import Mock, patch

from target.model_client import OllamaModelClient


def test_generate_returns_model_response():
    client = OllamaModelClient()
    mock_response = Mock()
    mock_response.json.return_value = {
        "response": "This is a test response."
    }

    with patch(
        "target.model_client.requests.post",
        return_value=mock_response,
    ) as mock_post:
        result = client.generate("Hello")

    assert result == "This is a test response."

    mock_post.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
    "model": "qwen3.5:9b",
    "prompt": "Hello",
    "stream": False,
    "think": False,
    "options": {
        "temperature": 0.2,
        "num_predict": 512,
    },
},
        timeout=120,
    )

    assert client.last_latency_seconds >= 0


def test_generate_rejects_empty_prompt():
    with pytest.raises(ValueError, match="non-empty string"):
        OllamaModelClient().generate("")


def test_generate_rejects_whitespace_prompt():
    with pytest.raises(ValueError, match="non-empty string"):
        OllamaModelClient().generate("   ")


def test_generate_raises_for_http_error():
    client = OllamaModelClient()
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError(
        "Server error"
    )

    with patch(
        "target.model_client.requests.post",
        return_value=mock_response,
    ):
        with pytest.raises(RuntimeError, match="HTTP error"):
            client.generate("Hello")


def test_generate_rejects_invalid_response_format():
    client = OllamaModelClient()
    mock_response = Mock()
    mock_response.json.return_value = {"unexpected": "data"}

    with patch(
        "target.model_client.requests.post",
        return_value=mock_response,
    ):
        with pytest.raises(RuntimeError, match="invalid response format"):
            client.generate("Hello")


def test_invalid_temperature_is_rejected():
    with pytest.raises(ValueError, match="temperature"):
        OllamaModelClient(temperature=3.0)


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError, match="timeout"):
        OllamaModelClient(timeout=0)


def test_invalid_generation_limit_is_rejected():
    with pytest.raises(ValueError, match="num_predict"):
        OllamaModelClient(num_predict=0)


def test_connection_error_is_handled():
    client = OllamaModelClient()

    with patch(
        "target.model_client.requests.post",
        side_effect=requests.ConnectionError("Connection failed"),
    ):
        with pytest.raises(RuntimeError, match="Could not connect to Ollama"):
            client.generate("Hello")


def test_timeout_error_is_handled():
    client = OllamaModelClient()

    with patch(
        "target.model_client.requests.post",
        side_effect=requests.Timeout("Request timed out"),
    ):
        with pytest.raises(RuntimeError, match="timed out"):
            client.generate("Hello")
