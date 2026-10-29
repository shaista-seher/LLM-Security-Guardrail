
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Paths and initial decision thresholds
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "modernbert_guardrail"
MAX_LENGTH = 256

# Provisional thresholds: calibrate using validation data later
FLAG_THRESHOLD = 0.50
BLOCK_THRESHOLD = 0.90

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

        inputs = self.tokenizer(
            context,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=1)[0]

        latency_ms = (time.perf_counter() - start) * 1000

        # Dataset mapping: class 0 = benign, class 1 = malicious
        malicious_score = probabilities[1].item()

        
        
        import re

        prompt_lower = user_prompt.lower().strip()
        context_lower = context.lower()

        # Detect common instruction-override and data-extraction attempts.
        injection_patterns = [
            r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous\s+|prior\s+)?(?:prompts?|instructions?)\b",
            r"\bdisregard\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions?\b",
            r"\b(?:reveal|show|expose)\b.{0,60}\b(?:hidden|system)\s+(?:prompt|instructions?)\b",
            r"\b(?:bypass|override)\b.{0,60}\b(?:safety|system)\s+instructions?\b",
            r"\b(?:give|fetch|dump|reveal|expose)\b.{0,80}\b(?:internal|private|hidden|backend|behind)\s+databases?\b",
            r"\b(?:fetch|dump|exfiltrate)\b.{0,60}\b(?:internal|private)\s+(?:database|data|records)\b",
        ]

        explicit_injection = any(
            re.search(pattern, context_lower)
            for pattern in injection_patterns
        )

        # Only allow obvious, standalone low-risk requests.
        # Never apply this exception when additional context is present.
        standalone_prompt = context.strip() == user_prompt.strip()

        is_greeting = bool(re.fullmatch(
            r"(?:hey+|hi+|hello+|heyy+|how are you|"
            r"hey+[, ]+how are you)[!?. ]*",
            prompt_lower
        ))

        is_simple_weather = bool(re.fullmatch(
            r"(?:what is|what's|whats)\s+the temperature"
            r"\s+in\s+[a-zA-Z ]+"
            r"(?:\s+(?:right now|now|today))?[?!. ]*",
            prompt_lower
        ))

        if explicit_injection:
            action = "BLOCK"
            reason = "Explicit instruction override or sensitive-data extraction detected."

        elif standalone_prompt and (is_greeting or is_simple_weather):
            action = "ALLOW"
            reason = "Recognized standalone, low-risk conversational request."

        elif malicious_score >= BLOCK_THRESHOLD:
            action = "BLOCK"
            reason = "High maliciousness score; further review may be warranted."

        elif malicious_score >= FLAG_THRESHOLD:
            action = "FLAG"
            reason = "Suspicious score; additional checks required."

        else:
            action = "ALLOW"
            reason = "Low maliciousness score under the current policy."

        return {
            "action": action,
            "malicious_score": round(malicious_score, 4),
            "benign_score": round(probabilities[0].item(), 4),
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
