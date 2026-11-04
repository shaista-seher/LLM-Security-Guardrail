# Adaptive Dual-Agent Framework for LLM Security

**Project:** Adaptive Dual-Agent Framework for Automated Red-Teaming and Context-Aware Runtime Guardrails Against Evolving Prompt Injection Attacks in LLM Agents.

## 1. Project overview

This repository contains a prompt-injection guardrail, a target LLM application, an attack-generation component, feedback storage, shared request/response schemas, and evaluation utilities. The goal is to generate adversarial test cases and evaluate how a guardrail handles untrusted prompts and context.

## 2. Architecture and implemented components

| Component | Main files | Responsibility |
|---|---|---|
| Guardrail classifier | `guardrail.py`, `modernbert_guardrail/` | Scores the user prompt and supplied context; returns ALLOW, FLAG, or BLOCK. |
| Context and target application | `target/app.py`, `target/context_builder.py` | Builds structured context, invokes the guardrail, and coordinates the target application. |
| Target model and safe tools | `target/model_client.py`, `target/tool_executor.py` | Uses local Ollama Qwen and allowlisted mock tools. Real external actions are disabled. |
| Red-team generation | `red_team/attack_generator.py`, `red_team/gemma_client.py` | Generates attack cases using fixed examples or local Gemma. |
| Feedback memory | `red_team/feedback_memory.py` | Stores and retrieves attack-outcome feedback for feedback-informed generation. |
| Shared contracts | `common/schemas.py` | Validates attack, guardrail, target, and evaluation records. |
| Evaluation | `evaluation/` | Contains baseline, metric, integration-record, and comparison utilities. |

**Important integration boundary:** the live `TargetApplication` calls the ModernBERT guardrail adapter, context builder, Ollama client, and safe tool executor. The red-team generator and feedback memory are used by the attack-generation workflow. The current code does not demonstrate that feedback memory updates the live guardrail's classifier or runtime decision policy. Therefore, feedback-informed generation and runtime guardrail detection must be reported as separate capabilities unless a controlled experiment proves a runtime adaptation effect.

## 3. Models

- **Guardrail encoder:** `answerdotai/ModernBERT-base`, configured for two classes (0 = benign, 1 = injection).
- **Target LLM:** `qwen3.5:9b` through Ollama (configurable with `TARGET_MODEL`).
- **Attack generator:** local Ollama Gemma when `use_gemma=True`; fixed/test generation can be used without live generation.
- **Baseline detector:** `protectai/deberta-v3-base-prompt-injection-v2` in `evaluation/baseline.py`.

The ModernBERT classifier must be fine-tuned on the supplied training split before its predictions are treated as meaningful evaluation results. A newly initialized classification head is not a trained detector.

## 4. Dataset

The supplied Parquet dataset uses these fields:

- `text`: input prompt.
- `label`: 0 = benign, 1 = attack.
- `category`: attack/benign category.
- `source`, `severity`, `group_id`, `augmented`, `tags`: supporting metadata.

Observed split sizes:

| Split | Rows | Benign (0) | Attack (1) |
|---|---:|---:|---:|
| Train | 4,391 | 1,741 | 2,650 |
| Validation | 941 | 407 | 534 |
| Test | 942 | 390 | 552 |

Keep the test split separate from training and model-selection decisions. Use validation data for threshold selection. Do not mix test examples into training.

## 5. Environment setup (Windows PowerShell)

From the repository root:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pandas pyarrow scikit-learn
```

Install and start Ollama separately, then pull the configured target and generator models as needed:

```powershell
ollama pull qwen3.5:9b
ollama pull gemma4:latest
Invoke-RestMethod http://localhost:11434/api/tags
```

For the baseline comparison, the ProtectAI model is downloaded by Hugging Face Transformers on first use. Internet access is required for first-time model downloads.

## 6. Train the ModernBERT guardrail

Place the three Parquet files in the repository root, or update the paths in `train_modernbert.py` to match their location. The script expects files named `train-*.parquet` and `validation-*.parquet` in the project root.

```powershell
python train_modernbert.py
```

The script trains for two epochs on CPU by default and saves the best validation-F1 checkpoint under `modernbert_guardrail/`. CPU training may take a substantial amount of time. Do not run the final test evaluation until training finishes successfully and the saved checkpoint is confirmed to be the trained checkpoint.

## 7. Run tests

```powershell
python -m pytest tests -v
```

Live tests require Ollama to be running and the configured model to be available. A passing unit/integration test suite verifies software contracts and execution paths; it does not by itself establish classifier accuracy or research effectiveness.

## 8. Evaluation protocol

A valid baseline-versus-proposed comparison must:

1. Use the exact same held-out test cases and labels for both systems.
2. Use the same prompt/context content and a documented detection rule.
3. Evaluate the trained guardrail checkpoint, not a newly initialized classifier head.
4. Report inference latency separately from model-loading and offline attack-generation time.
5. Include attack categories and benign controls so category recall and false-positive rate can be calculated.
6. Keep attack-generation validity/feedback-reference rates separate from runtime detection metrics.
7. Report feedback adaptation improvement only after a controlled feedback-disabled versus feedback-enabled experiment using the same evaluation protocol.
8. Report attack success rate only when objective outcomes were independently observed and labelled.

The comparison utility currently imports `ModernBERTGuardrail` directly. It therefore evaluates that guardrail detector, not the full `TargetApplication` request path. Do not describe its output as a complete four-module end-to-end evaluation unless the runner is updated to call the full target application and its outputs are validated.

The `evaluation/run_m3_m4.py` runner generates attack cases and sends them through `TargetApplication`; it can optionally run the baseline. It is a generated-case integration workflow, not a substitute for a large fixed held-out test-set comparison. The `artifacts/m3_m4_evaluation.json` fixture created by tests may contain dummy records; inspect its provenance before reporting it as experimental data.

## 9. Metrics and graphs

For the labelled test set, report:

- Attack detection rate (recall) and attack miss rate.
- Detected and missed attack counts.
- Detection rate by attack category.
- Precision, F1, accuracy, and false-positive rate when benign labels are included.
- Mean, median, P95, minimum, and maximum inference latency.
- Throughput, measured as evaluated cases divided by batch wall-clock time.

For the red-team generator, report requested/generated case counts, schema-validity rate, generation latency, and feedback-reference utilization. A non-null feedback reference measures use of stored feedback during generation; it does not by itself prove improved guardrail detection.

The comparison script can create plots and JSON/CSV outputs under `artifacts/comparison/` when run on a valid prepared JSON dataset and a trained model. The current script's proposed detector is ModernBERT-only; the end-to-end comparison needs an explicit target-application evaluation path before it can be called a complete-agent comparison.

## 10. Safety and limitations

- The target application only exposes allowlisted mock tools; it does not perform real external actions.
- Context provenance labels are metadata and are not, by themselves, a security guarantee.
- The current feedback store supports feedback-informed generation; runtime model adaptation is not established by the code described above.
- Thresholds in `guardrail.py` are provisional and should be calibrated on validation data.
- Research screening results are not a guarantee of security. Evaluate false positives, false negatives, and failure cases before deployment.

## 11. Current verification status

The project test suite was reported as **85 tests passed** in the local development environment. The Parquet splits were also inspected locally: 4,391 training, 941 validation, and 942 test records, with 390 benign and 552 attack labels in the test split. These checks establish dataset readability and software-test status; they do not establish final baseline-versus-complete-agent performance.

## 12. Repository layout

```text
.
├── app.py
├── guardrail.py
├── train_modernbert.py
├── modernbert_guardrail/
├── common/
├── target/
├── red_team/
├── evaluation/
├── datasets/
└── tests/
```
