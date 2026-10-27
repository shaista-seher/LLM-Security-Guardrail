# Target Application — Member 2

## 1. Overview

The target application is part of the Adaptive Dual-Agent Framework for Automated Red-Teaming and Context-Aware Runtime Guardrails Against Evolving Prompt Injection Attacks in LLM Agents.

It uses the Qwen3.5-9B model locally through Ollama and prepares structured request context for processing. A dummy Guardrail is currently used for testing until the actual Guardrail component is ready for integration.

## 2. Architecture

The target application consists of the following components:

- `config.py` — Central configuration for the model, Ollama connection, generation settings, and context limits.
- `model_client.py` — Sends generation requests to the locally running Ollama model.
- `context_builder.py` — Validates request data, preserves context provenance and trust labels, and applies context limits.
- `app.py` — Coordinates request processing, Guardrail decisions, model responses, and mock-tool execution.
- `tool_executor.py` — Validates and executes allowlisted mock tools.
- `dummy_guardrail.py` — Provides simulated ALLOW, FLAG, and BLOCK decisions for testing.

The shared request and response schemas are defined in `common/schemas.py`.

## 3. Environment Requirements

- Python 3.11 or a compatible supported Python version
- Ollama installed and running
- Qwen3.5-9B available through Ollama
- Python dependencies listed in the root `requirements.txt`

### Verified Model Details

- **Model:** Qwen3.5-9B
- **Ollama tag:** `qwen3.5:9b`
- **Installed model ID:** `56671c2ab938`
- **Installed model size:** 6.6 GB
- **Quantization:** Q4_K_M
- **Official model-card license:** Apache-2.0

Official references:

- Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Ollama model registry: https://ollama.com/library/qwen3.5
- Ollama model tags: https://ollama.com/library/qwen3.5/tags

The model is loaded locally through Ollama. The installed model's 6.6 GB size is not the same as its total runtime memory requirement; inference also requires memory for the context and other runtime operations.

The application currently uses an approximate 6,000-token context budget. This is an application-level limit and is not the model's maximum context window.

## 4. Setup

From the project root, activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

Check that Ollama is responding:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

Ensure the configured Qwen model is available before running live model tests.

## 5. Configuration

The default settings are defined in `config.py`:

- Model: `qwen3.5:9b`
- Ollama URL: `http://localhost:11434`
- Model timeout: 120 seconds
- Temperature: 0.2
- Maximum generated tokens: 512
- Maximum user prompt length: 8,000 characters
- Maximum conversation history items: 20
- Maximum context items per source: 10
- Maximum characters per context item: 3,000
- Approximate context token budget: 6,000

Some model settings can be overridden using environment variables. See `config.py` for the supported variable names.

The context token limit is currently estimated from character length, approximately four characters per token. It is not an exact count from the Qwen tokenizer.

## 6. Request Processing

The current request-processing flow is:

1. Validate and prepare the incoming request.
2. Apply context length and item-count limits.
3. Obtain a decision from the currently configured dummy Guardrail.
4. If the decision is BLOCK, stop processing.
5. If the decision is FLAG, return a flagged result without executing the requested action.
6. If the decision is ALLOW, process the supported request.
7. Execute only supported mock tools when requested and permitted by the executor.
8. Generate a response using the local Qwen model when appropriate.
9. Return a structured response matching the shared response schema.

## 7. Supported Mock Tools

The current implementation supports:

- `search_mock`
- `calculator_mock`

Tool names, targets, and arguments are validated. Real external actions and real-world side effects are not enabled.

The request's `requires_authorization` field does not, by itself, prove that authorization was granted. Authorization must be enforced separately before any sensitive action is introduced.

## 8. Running Tests

Run the complete test suite from the project root:

```powershell
python -m pytest -v
```

The live Qwen tests require Ollama to be running and the configured model to be available.

Unit tests use mock components where appropriate so that application behavior can be tested independently of live inference.

## 9. Current Limitations

- The real Guardrail implementation has not yet been integrated.
- The dummy Guardrail is for development and testing; it does not detect actual prompt injection.
- Context token usage is estimated rather than calculated with the model tokenizer.
- The initial tools are mocked and do not perform real external operations.
- Context formatting and provenance labels do not, on their own, prevent prompt injection.
- Production deployment requires additional validation, authorization controls, monitoring, and security testing.

## 10. Integration Status

The application currently uses the shared schemas and a dummy Guardrail. Integration with Member 1's real Guardrail will be performed once its agreed interface is available.

The model choice remains Qwen3.5-9B, and the target model is used without fine-tuning in this implementation.