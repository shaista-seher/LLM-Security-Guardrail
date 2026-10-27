import requests


def test_real_ollama_generation():
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "qwen3.5:9b",
        "prompt": "Reply with exactly these words: Qwen connection successful",
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
        },
    }

    response = requests.post(url, json=payload, timeout=120)
    print("\nHTTP status:", response.status_code)
    response.raise_for_status()

    data = response.json()

    print("\nGenerated response:", repr(data.get("response")))
    print("Thinking:", repr(data.get("thinking")))
    print("Done reason:", data.get("done_reason"))
    print("Generation tokens:", data.get("eval_count"))

    assert isinstance(data.get("response"), str)
    assert data["response"].strip() != ""