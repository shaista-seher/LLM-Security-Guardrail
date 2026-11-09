
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths and initial decision thresholds
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modernbert_guardrail_candidate_v2"
MAX_LENGTH = 256

# Provisional thresholds: calibrate using validation data later
FLAG_THRESHOLD = 0.9674102663993835
BLOCK_THRESHOLD = 0.9674102663993835

class ModernBERTGuardrail:
    def __init__(self):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Saved model not found: {MODEL_DIR}"
            )

        print("Loading saved ModernBERT guardrail...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_DIR)
        )
        self.model.to("cpu")
        self.model.eval()
        print("Guardrail ready.\n")

    def build_context(self, user_prompt, conversation_history="", retrieved_context="", tool_output="", memory=""):
        parts = [
            user_prompt,
            conversation_history,
            retrieved_context,
            tool_output,
            memory,
        ]
        return "\n".join(
            value.strip() for value in parts
            if value and value.strip()
        )
    
    
    def evaluate(
        self,
        user_prompt,
        conversation_history="",
        retrieved_context="",
        tool_output="",
        memory="",
    ):
        import re

        context = self.build_context(
            user_prompt,
            conversation_history,
            retrieved_context,
            tool_output,
            memory,
        )

        if not context.strip():
            raise ValueError("Please provide a non-empty prompt or context.")

        start = time.perf_counter()

        # Classify the user's prompt independently.
        # Do not let ordinary retrieved information distort this score.
        prompt_text = user_prompt.strip()

        if prompt_text:
            inputs = self.tokenizer(
                prompt_text,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)[0]

            prompt_malicious_score = probabilities[1].item()
            benign_score = probabilities[0].item()
        else:
            prompt_malicious_score = 0.0
            benign_score = 1.0

        # Inspect ALL supplied content for explicit injection attempts.
        # Trust labels do not exempt content from inspection.
        context_lower = context.lower()

        injection_patterns = [
            r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous\s+|prior\s+)?(?:prompts?|instructions?)\b",
            r"\bdisregard\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions?\b",
            r"\bforget\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions?\b",
            r"\boverride\s+(?:all\s+)?(?:previous\s+|current\s+)?instructions?\b",
            r"\breveal\b.{0,80}\b(?:system prompt|hidden prompt|secret instructions)\b",
            r"\b(?:show|expose|print|display)\b.{0,80}\b(?:system prompt|hidden instructions)\b",
            r"\bbypass\b.{0,60}\b(?:safety|security|guardrails?)\b",
            r"\b(?:exfiltrate|dump|steal)\b.{0,80}\b(?:secrets?|credentials?|private data|api keys?)\b",
            r"\b(?:send|output|return)\b.{0,80}\b(?:passwords?|api keys?|secret tokens?)\b",
            r"\b(?:do not|don't)\s+follow\s+(?:the\s+)?(?:previous|original|system)\s+instructions?\b",
            r"\byou are now\b.{0,80}\b(?:unrestricted|unfiltered|jailbroken)\b",
            r"\b(?:act as|pretend to be)\b.{0,80}\b(?:unrestricted|unfiltered|no restrictions)\b",
        ]

        explicit_injection = any(
            re.search(pattern, context_lower)
            for pattern in injection_patterns
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Explicit injection indicators take priority over classifier scores.
        if explicit_injection:
            action = "BLOCK"
            malicious_score = 1.0
            reason = (
                "Explicit instruction override or sensitive-data "
                "extraction detected in the supplied content."
            )

        elif prompt_malicious_score >= BLOCK_THRESHOLD:
            action = "BLOCK"
            malicious_score = prompt_malicious_score
            reason = "High maliciousness score in the user prompt."

        elif prompt_malicious_score >= FLAG_THRESHOLD:
            action = "FLAG"
            malicious_score = prompt_malicious_score
            reason = "Suspicious user prompt; additional review required."

        else:
            action = "ALLOW"
            malicious_score = prompt_malicious_score
            reason = (
                "User prompt has a low maliciousness score and no "
                "explicit injection pattern was detected in the context."
            )

        return {
            "action": action,
            "malicious_score": round(malicious_score, 4),
            "benign_score": round(benign_score, 4),
            "reason": reason,
            "latency_ms": round(latency_ms, 2),
        }



def main():
    guardrail = ModernBERTGuardrail()

    print("ModernBERT Guardrail Demo")
    print("Enter 'exit' to quit.\n")

    while True:
        prompt = input("User prompt: ").strip()

        if prompt.lower() == "exit":
            print("Exiting guardrail.")
            break

        if not prompt:
            print("Please enter a prompt.\n")
            continue

        history = input(
            "Conversation history (Enter to skip): "
        ).strip()
        retrieved = input(
            "Retrieved/web content (Enter to skip): "
        ).strip()
        tool_output = input(
            "Tool output (Enter to skip): "
        ).strip()
        memory = input(
            "Agent memory (Enter to skip): "
        ).strip()

        try:
            result = guardrail.evaluate(
                user_prompt=prompt,
                conversation_history=history,
                retrieved_context=retrieved,
                tool_output=tool_output,
                memory=memory,
            )

            print("\n--- GUARDRAIL RESULT ---")
            print(f"Action:           {result['action']}")
            print(f"Malicious score:  {result['malicious_score']:.4f}")
            print(f"Benign score:     {result['benign_score']:.4f}")
            print(f"Reason:           {result['reason']}")
            print(f"Latency:          {result['latency_ms']:.2f} ms")
            print("------------------------\n")

        except Exception as error:
            print(f"Error evaluating prompt: {error}\n")


if __name__ == "__main__":
    main()
