
import os


class TargetConfig:
    """Configuration for the target application."""

    # Local Qwen model served through Ollama
    MODEL_NAME = os.getenv("TARGET_MODEL", "qwen3.5:9b")
    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    # Model generation settings
    MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "120"))
    MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    MODEL_MAX_OUTPUT_TOKENS = int(
        os.getenv("MODEL_MAX_OUTPUT_TOKENS", "512")
    )

    # Context limits
    MAX_PROMPT_CHARS = 8000
    MAX_HISTORY_ITEMS = 20
    MAX_CONTEXT_ITEMS_PER_SOURCE = 10
    MAX_CONTEXT_CHARS_PER_ITEM = 3000
    MAX_CONTEXT_TOKENS = 6000
