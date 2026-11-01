
"""Local Ollama client for the Member 3 Red-Team Agent."""

import json
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e4b"


def generate_with_gemma(
    prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 600,
) -> str:
    """Send a prompt to Gemma through Ollama and return its response."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string.")

    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("timeout must be a positive integer.")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama returned HTTP {exc.code}: {details}"
        ) from exc

    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Check that Ollama is running."
        ) from exc

    if not isinstance(result, dict):
        raise RuntimeError("Ollama returned an unexpected response.")

    generated_text = result.get("response")

    if not isinstance(generated_text, str):
        raise RuntimeError("Ollama response did not contain generated text.")

    if not generated_text.strip():
        raise RuntimeError("Gemma returned an empty response.")

    return generated_text.strip()


if __name__ == "__main__":
    test_prompt = (
        "You are testing a red-team research system. "
        "Reply with exactly: RED_TEAM_MODEL_READY"
    )

    print(generate_with_gemma(test_prompt))
