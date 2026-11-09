# Adaptive Dual-Agent Framework for LLM Security

**Project Title:** Adaptive Dual-Agent Framework for Automated Red-Teaming and Context-Aware Runtime Guardrails Against Evolving Prompt Injection Attacks in LLM Agents.

## 1. Project Overview

This project proposes an adaptive dual-agent security framework designed to detect and prevent prompt injection attacks in Large Language Model (LLM)-based applications.

The framework combines a ModernBERT-based runtime guardrail, a target LLM, an automated red-team attack generator, and a feedback memory mechanism. It evaluates how effectively the guardrail identifies malicious prompts and untrusted contextual instructions before they reach the target LLM.

The framework consists of four main modules:

- **M1 – Guardrail Agent:** Uses fine-tuned ModernBERT V2 to classify prompts and identify potentially malicious instructions.
- **M2 – Target LLM:** Uses Qwen 3.5 9B through Ollama to process requests permitted by the guardrail.
- **M3 – Red-Team Agent:** Uses a local Gemma model to generate adversarial prompts and test the security framework.
- **M4 – Evaluation and Feedback Memory:** Evaluates attack outcomes, calculates performance metrics, and stores feedback for subsequent attack generation.

The primary objective is to provide a structured security mechanism that detects malicious inputs, prevents unsafe execution, and supports feedback-informed red-team testing.

## 2. System Architecture and Implemented Components

| Component | Main Files | Responsibility |
|---|---|---|
| Guardrail Agent (M1) | `guardrail.py`, `modernbert_guardrail_candidate_v2/` | Classifies user prompts and detects malicious instructions in supplied context. Produces ALLOW, FLAG, or BLOCK decisions. |
| Target Application (M2) | `target/app.py`, `target/context_builder.py` | Coordinates guardrail evaluation, context construction, and target LLM execution. |
| Target LLM | `target/model_client.py` | Connects to the local Qwen model through Ollama and generates responses. |
| Safe Tool Executor | `target/tool_executor.py` | Provides allowlisted mock tools and prevents unauthorized real-world actions. |
| Red-Team Agent (M3) | `red_team/attack_generator.py`, `red_team/gemma_client.py` | Generates adversarial test cases using fixed examples or local Gemma. |
| Feedback Memory (M4) | `red_team/feedback_memory.py` | Stores attack outcomes and retrieves relevant feedback for subsequent attack generation. |
| Shared Schemas | `common/schemas.py` | Validates request structures, attack cases, guardrail results, and target results. |
| Evaluation Pipeline | `evaluation/run_m3_m4.py`, `evaluation/m3_m4_pipeline.py` | Integrates generated attack cases with the target application and records evaluation outcomes. |
| Model Comparison | `evaluation/compare_guardrail_models.py` | Compares ModernBERT V2 with the ProtectAI baseline detector. |
| Graph Generation | `evaluation/plot_model_comparison.py` | Generates performance comparison graphs from saved evaluation results. |
| Research Metrics | `evaluation/research_metrics.py` | Provides additional evaluation and research metric utilities. |

### Important Integration Boundary

The live target application integrates the ModernBERT guardrail adapter, context builder, Ollama model client, and safe tool executor.

The red-team generator and feedback memory support feedback-informed attack generation. The current implementation does not establish that feedback memory directly modifies the ModernBERT classifier or its runtime decision policy.

Therefore, feedback-informed attack generation and runtime guardrail detection are evaluated as separate capabilities unless a controlled experiment demonstrates a runtime adaptation effect.

## 3. Models and Technologies

### 3.1 Guardrail Model

- Model: `answerdotai/ModernBERT-base`
- Architecture: ModernBERT encoder
- Classification: Binary classification
- Class 0: Benign
- Class 1: Prompt injection / malicious input
- Implementation: Fine-tuned ModernBERT V2
- Output: Maliciousness score and guardrail decision.

The trained ModernBERT checkpoint is used by the runtime guardrail to evaluate incoming requests.

### 3.2 Target LLM

- Model: `qwen3.5:9b`
- Runtime: Ollama
- Purpose: Generates responses for requests permitted by the guardrail.

### 3.3 Red-Team Model

- Model: Local Gemma through Ollama.
- Purpose: Generates adversarial prompts and attack scenarios.
- Generation modes: Fixed/test cases, few-shot generation, and feedback-informed adaptive generation.

### 3.4 Baseline Model

- Model: `protectai/deberta-v3-base-prompt-injection-v2`
- Runtime: Hugging Face Transformers.
- Purpose: Provides a baseline for prompt-injection detection comparison.

## 4. Dataset

The project uses a Parquet-based dataset for training, validation, and testing the ModernBERT guardrail.

The dataset contains the following fields:

| Field | Description |
|---|---|
| `text` | Input prompt or text. |
| `label` | Classification label: 0 = benign, 1 = attack. |
| `category` | Attack or benign category. |
| `source` | Dataset source. |
| `severity` | Attack severity information. |
| `group_id` | Group identifier. |
| `augmented` | Indicates whether the sample was augmented. |
| `tags` | Additional sample metadata. |

### Dataset Split

| Split | Total Rows | Benign (0) | Attack (1) |
|---|---:|---:|---:|
| Training | 4,391 | 1,741 | 2,650 |
| Validation | 941 | 407 | 534 |
| Testing | 942 | 390 | 552 |
| **Total** | **6,274** | **2,538** | **3,736** |

The training split is used for model fine-tuning, while validation data is used for model selection and threshold calibration.

The test split must remain separate from training and model-selection decisions to ensure an unbiased evaluation.

## 5. System Workflow

The framework follows the workflow below:

1. The Red-Team Agent generates adversarial test cases or receives predefined test inputs.
2. The generated request is passed to the target application.
3. The context builder organizes the user prompt, conversation history, retrieved content, web content, tool output, and memory.
4. The ModernBERT guardrail evaluates the user prompt and checks the supplied context for explicit malicious instructions.
5. The guardrail produces an ALLOW, FLAG, or BLOCK decision.
6. BLOCK and FLAG requests are prevented from proceeding to target LLM execution.
7. ALLOW requests are passed to Qwen 3.5 through Ollama.
8. The target model generates a response.
9. The evaluation pipeline records the guardrail decision, execution status, response, and attack outcome.
10. Feedback memory stores attack outcomes for future feedback-informed attack generation.

### Guardrail Decisions

| Decision | Description |
|---|---|
| ALLOW | Request is permitted to proceed to the target LLM. |
| FLAG | Request is identified as suspicious and prevented from proceeding. |
| BLOCK | Request is identified as malicious and execution is stopped. |

## 6. Environment Setup

### Prerequisites

- Python 3.11.9
- Git
- Ollama
- NVIDIA GPU recommended for local model execution and training.
- Sufficient RAM and storage for local LLMs and model checkpoints.

### Windows PowerShell Setup

Clone the repository:

```powershell
git clone https://github.com/shaista-seher/LLM-Security-Guardrail.git
cd LLM-Security-Guardrail
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install Ollama separately and start its local service.

Pull the target and red-team models:

```powershell
ollama pull qwen3.5:9b
ollama pull gemma4:latest
```

Verify the Ollama service:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

The configured model names must match the models available in the local Ollama installation.

For baseline evaluation, the ProtectAI model is downloaded through Hugging Face Transformers on first use. Internet access is required for initial model downloads.

## 7. Train the ModernBERT Guardrail

The repository includes training scripts for ModernBERT fine-tuning and candidate evaluation.

Main scripts:

- `train_modernbert.py`
- `train_modernbert_audited.py`
- `train_modernbert_original.py`
- `evaluate_candidate.py`

The training script expects the training and validation Parquet files at the configured dataset paths.

Run the training script:

```powershell
python train_modernbert.py
```

The trained model is saved to the output directory configured in the training script.

The final runtime guardrail should load the selected trained checkpoint rather than an untrained classification head.

The selected runtime model in the current implementation is:

```text
modernbert_guardrail_candidate_v2/
```

Model directories must be present at the paths expected by the guardrail implementation.

## 8. Running the Project

### 8.1 Run the Adaptive Dual-Agent Demonstration

The repository includes a controlled demonstration script:

```powershell
python run_adaptive_demo.py
```

The demonstration contains two predefined test cases.

**Case 1 – Malicious Input**

- A user requests a summary of a research report.
- The retrieved RAG content contains an explicit prompt-injection instruction.
- ModernBERT V2 identifies the malicious instruction.
- The request receives a BLOCK decision.
- Qwen execution is prevented.

**Case 2 – Benign Input**

- A normal user request is supplied.
- ModernBERT V2 evaluates the request.
- If the request receives an ALLOW decision, Qwen generates a response.
- The target LLM response and execution status are recorded.

The benign demonstration prompt is selected from candidate prompts after checking that the guardrail allows it.

This is a controlled demonstration using predefined test cases. It does not represent live Gemma-generated attack execution.

### 8.2 Run the M3–M4 Evaluation Pipeline

```powershell
python evaluation/run_m3_m4.py
```

The evaluation runner supports generated attack cases, optional Gemma generation, target application execution, feedback recording, and optional baseline evaluation.

### 8.3 Run the Model Comparison

```powershell
python evaluation/compare_guardrail_models.py
```

This evaluates ModernBERT V2 and the ProtectAI baseline on the configured test dataset.

### 8.4 Generate Comparison Graphs

```powershell
python evaluation/plot_model_comparison.py
```

The plotting script generates performance comparison visualizations using previously saved prediction results.

### 8.5 Run Tests

```powershell
python -m pytest tests -v
```

Live integration tests may require Ollama to be running and the configured models to be available.

## 9. Evaluation Metrics

The framework evaluates guardrail performance using standard classification metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall classification correctness. |
| Precision | Measures the proportion of predicted attacks that are actual attacks. |
| Recall | Measures the proportion of actual attacks detected. |
| F1-score | Provides a balance between precision and recall. |
| Specificity | Measures the correct identification of benign samples. |
| False Positive Rate | Measures the proportion of benign inputs incorrectly classified as attacks. |
| False Negative Rate | Measures the proportion of attacks incorrectly classified as benign. |
| Inference Latency | Measures the time taken by the guardrail to evaluate an input. |

The comparison and plotting utilities provide additional performance visualizations, including confusion matrices, metric comparisons, error rates, detection outcomes, and latency comparisons.

Latency results should be interpreted with consideration of hardware, model-loading overhead, warm-up, and execution device.

## 10. Model Comparison Results

The latest recorded comparison on the 942-sample test dataset produced the following results:

| Metric | ModernBERT V2 | ProtectAI |
|---|---:|---:|
| Accuracy | 96.07% | 91.40% |
| Precision | 96.70% | 97.75% |
| Recall | 96.53% | 87.20% |
| F1-score | 96.61% | 92.17% |
| Specificity | 95.44% | 97.22% |
| False Positive Rate | 4.56% | 2.78% |
| False Negative Rate | 3.47% | 12.80% |

### Observations

- ModernBERT V2 achieved higher accuracy, recall, and F1-score in this evaluation.
- ProtectAI achieved higher precision and specificity.
- ModernBERT V2 detected a larger proportion of the attack samples.
- ProtectAI produced a lower false-positive rate.

The recorded mean inference latencies were approximately 160.22 ms for ModernBERT V2 and 66.53 ms for ProtectAI. However, the models were not evaluated under confirmed identical hardware conditions, so these latency figures should not be used to claim a definitive speed advantage.

These results describe the guardrail detector comparison and should not be interpreted as a complete four-module end-to-end system evaluation.

## 11. Two-Case Demonstration Results

The controlled demonstration was executed using `run_adaptive_demo.py`.

| Evaluation | Case 1 | Case 2 |
|---|---|---|
| Input Type | Malicious RAG injection | Benign request |
| Expected Classification | Attack | Benign |
| Guardrail Decision | BLOCK | ALLOW |
| Qwen Execution | Not invoked | Completed |
| Execution Status | BLOCKED | COMPLETED |

### Observed Results

**Case 1 – Malicious**

- ModernBERT maliciousness score: 1.0000
- Decision: BLOCK
- Execution status: BLOCKED
- Target response: Request blocked by Guardrail.

**Case 2 – Benign**

- ModernBERT maliciousness score: 0.9491
- Decision: ALLOW
- Execution status: COMPLETED
- Qwen response: 2 + 2 equals 4.

### Demonstration Metrics

| Metric | Result |
|---|---:|
| True Positives | 1 |
| True Negatives | 1 |
| False Positives | 0 |
| False Negatives | 0 |
| Precision | 1.00 |
| Recall | 1.00 |
| F1-score | 1.00 |
| False Positive Rate | 0.00 |

**Note:** These metrics represent only the two controlled demonstration cases. They must not be interpreted as the overall performance of the framework or the 942-sample test dataset.

## 12. Feedback Memory and Adaptive Generation

The feedback memory component records attack-generation and execution outcomes.

The stored feedback includes information such as:

- Attack category.
- Expected attack label.
- Guardrail decision.
- Attack objective outcome.
- Classifier score.
- Target response.
- Execution status.
- Generation method.
- Evaluation run identifier.

The attack generator can retrieve relevant feedback to inform subsequent adversarial prompt generation.

The current implementation supports feedback-informed attack generation. It does not establish automatic retraining of ModernBERT or direct modification of its runtime classification threshold.

A controlled feedback-disabled versus feedback-enabled experiment is required to demonstrate measurable adaptation improvement.

## 13. Safety and Limitations

- The target application uses allowlisted mock tools.
- Real external actions are disabled.
- Context provenance labels are metadata and do not independently guarantee security.
- Guardrail thresholds should be calibrated using validation data.
- Feedback-informed generation does not automatically imply runtime classifier adaptation.
- The two-case demonstration is intended to illustrate execution flow, not establish statistical performance.
- Model comparison results apply to the evaluated detector configurations and dataset.
- Inference latency can vary depending on hardware, model loading, and execution conditions.
- Research evaluation results do not guarantee complete protection against all prompt injection attacks.

## 14. Current Verification Status

The following implementation and evaluation activities have been completed in the local development environment:

- ModernBERT V2 guardrail integration.
- Qwen 3.5 target LLM integration through Ollama.
- Gemma-based red-team generation support.
- Feedback memory and adaptive attack-generation support.
- Target application and safe tool execution integration.
- Evaluation pipeline and feedback recording.
- ModernBERT V2 versus ProtectAI baseline comparison.
- Performance graph generation.
- Controlled two-case BLOCK and ALLOW demonstration.
- Automated test suite previously reported as 85 tests passed.

The two-case demonstration produced one true positive and one true negative, with both requests receiving the expected decisions.

These verification results demonstrate the implemented execution paths and recorded experimental outcomes. They do not independently establish complete security effectiveness or runtime model adaptation.

## 15. Repository Structure

```text
LLM-Security-Guardrail/
│
├── app.py
├── guardrail.py
├── run_adaptive_demo.py
│
├── common/
│   └── schemas.py
│
├── target/
│   ├── app.py
│   ├── context_builder.py
│   ├── guardrail_adapter.py
│   ├── model_client.py
│   └── tool_executor.py
│
├── red_team/
│   ├── attack_generator.py
│   ├── gemma_client.py
│   └── feedback_memory.py
│
├── evaluation/
│   ├── run_m3_m4.py
│   ├── m3_m4_pipeline.py
│   ├── compare_guardrail_models.py
│   ├── plot_model_comparison.py
│   └── research_metrics.py
│
├── tests/
│
├── data/
│
├── modernbert_guardrail_candidate_v2/
│
├── train_modernbert.py
├── train_modernbert_audited.py
├── train_modernbert_original.py
└── evaluate_candidate.py
```

## 16. Conclusion

The Adaptive Dual-Agent Framework provides an integrated approach to testing and detecting prompt injection attacks in LLM-based applications. By combining a ModernBERT V2 runtime guardrail, Qwen target LLM, Gemma-based red-team generation, and feedback memory, the framework supports controlled adversarial testing and runtime request filtering.

The implementation demonstrates how malicious contextual instructions can be blocked before reaching the target LLM, while benign requests can proceed to response generation. The evaluation utilities further support quantitative comparison, performance analysis, and feedback-informed attack generation.

The framework establishes a foundation for further research into adaptive LLM security, prompt injection detection, and runtime guardrail mechanisms.
