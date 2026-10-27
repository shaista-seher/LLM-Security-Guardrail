from typing import Any
from pydantic import ValidationError

from common.schemas import RequestSchema
from target.config import TargetConfig

class ContextBuilder:
    """Validates and prepares structured context for the target application."""

    ALLOWED_TRUST_VALUES = {"trusted", "untrusted", "unknown"}

    def __init__(
    self,
    max_prompt_chars: int = TargetConfig.MAX_PROMPT_CHARS,
    max_history_items: int = TargetConfig.MAX_HISTORY_ITEMS,
    max_context_items_per_source: int = TargetConfig.MAX_CONTEXT_ITEMS_PER_SOURCE,
    max_context_chars_per_item: int = TargetConfig.MAX_CONTEXT_CHARS_PER_ITEM,
    max_context_tokens: int = TargetConfig.MAX_CONTEXT_TOKENS,
):
        if max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive.")
        if max_history_items < 0:
            raise ValueError("max_history_items cannot be negative.")
        if max_context_items_per_source < 0:
            raise ValueError("max_context_items_per_source cannot be negative.")
        if max_context_chars_per_item <= 0:
            raise ValueError("max_context_chars_per_item must be positive.")
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive.")

        self.max_prompt_chars = max_prompt_chars
        self.max_history_items = max_history_items
        self.max_context_items_per_source = max_context_items_per_source
        self.max_context_chars_per_item = max_context_chars_per_item
        self.max_context_tokens = max_context_tokens

    def build(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Validate and prepare context within configured limits."""

        try:
            request = RequestSchema.model_validate(request_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid request data: {exc}") from exc

        if not request.case_id.strip():
            raise ValueError("case_id cannot be empty.")

        if not request.user_prompt.strip():
            raise ValueError("user_prompt cannot be empty.")

        if len(request.user_prompt) > self.max_prompt_chars:
            raise ValueError(
                f"user_prompt exceeds the {self.max_prompt_chars}-character limit."
            )

        prompt_tokens = self.estimate_tokens(request.user_prompt)
        if prompt_tokens > self.max_context_tokens:
            raise ValueError(
                "user_prompt exceeds the configured context token budget."
            )

        remaining_tokens = self.max_context_tokens - prompt_tokens

        context = {
            "case_id": request.case_id,
            "user_prompt": request.user_prompt,
            "conversation_history": [],
            "rag_content": [],
            "web_content": [],
            "tool_output": [],
            "memory": [],
            "requested_action": request.requested_action.model_dump(),
        }

        history = (
            request.conversation_history[-self.max_history_items:]
            if self.max_history_items
            else []
        )

        for message in history:
            limited_text = self._limit_text(
                message.content,
                self.max_context_chars_per_item,
            )
            limited_text, remaining_tokens = self._fit_text_to_budget(
                limited_text,
                remaining_tokens,
            )

            if limited_text:
                context["conversation_history"].append(
                    {"role": message.role, "content": limited_text}
                )

        source_fields = (
            "rag_content",
            "web_content",
            "tool_output",
            "memory",
        )

        for field_name in source_fields:
            items = getattr(request, field_name)
            selected_items = items[:self.max_context_items_per_source]

            for item in selected_items:
                prepared_item = self._prepare_context_item(
                    item.model_dump(),
                    field_name,
                )

                limited_text, new_remaining_tokens = self._fit_text_to_budget(
                    prepared_item["text"],
                    remaining_tokens,
                )

                if not limited_text:
                    continue

                prepared_item["text"] = limited_text
                context[field_name].append(prepared_item)
                remaining_tokens = new_remaining_tokens

        return context

    def _prepare_context_item(
        self,
        item: dict[str, Any],
        field_name: str,
    ) -> dict[str, str]:
        """Validate trust metadata and preserve provenance."""

        trust = item["trust"].strip().lower()

        if trust not in self.ALLOWED_TRUST_VALUES:
            raise ValueError(
                f"Invalid trust value '{item['trust']}' in {field_name}. "
                "Allowed values are: trusted, untrusted, unknown."
            )

        return {
            "source": item["source"],
            "trust": trust,
            "text": self._limit_text(
                item["text"],
                self.max_context_chars_per_item,
            ),
        }

    def _fit_text_to_budget(
        self,
        text: str,
        remaining_tokens: int,
    ) -> tuple[str, int]:
        """Fit text within the remaining approximate token budget."""

        if remaining_tokens <= 0 or not text:
            return "", remaining_tokens

        allowed_chars = remaining_tokens * 4
        limited_text = text[:allowed_chars]
        used_tokens = self.estimate_tokens(limited_text)

        return limited_text, remaining_tokens - used_tokens

    @staticmethod
    def _limit_text(text: str, limit: int) -> str:
        return text[:limit]

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count at approximately four characters per token."""

        if not text:
            return 0

        return (len(text) + 3) // 4