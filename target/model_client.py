import time
import requests
from target.config import TargetConfig

class OllamaModelClient:
    """Client for running a local language model through Ollama."""

    def __init__(
    self,
    model: str = TargetConfig.MODEL_NAME,
    base_url: str = TargetConfig.OLLAMA_BASE_URL,
    timeout: int = TargetConfig.MODEL_TIMEOUT,
    temperature: float = TargetConfig.MODEL_TEMPERATURE,
    num_predict: int = TargetConfig.MODEL_MAX_OUTPUT_TOKENS,
):
        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        if num_predict <= 0:
            raise ValueError("num_predict must be positive.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_predict = num_predict
        self.last_latency_seconds = None

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            generated_text = data.get("response")

            if not isinstance(generated_text, str):
                raise RuntimeError(
                    "Ollama returned an invalid response format."
                )

            if not generated_text.strip():
                raise RuntimeError(
                    "Ollama returned an empty response. "
                    "The model may have stopped before producing a final answer."
                )

            return generated_text.strip()

        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds."
            ) from exc

        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Check that Ollama is running."
            ) from exc

        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned an HTTP error: {exc}"
            ) from exc

        finally:
            self.last_latency_seconds = time.perf_counter() - start_time