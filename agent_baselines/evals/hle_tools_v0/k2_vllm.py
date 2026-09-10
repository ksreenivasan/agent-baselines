"""K2 request replay compatibility for assistant turns with empty reasoning."""

from typing import Any

from inspect_ai.model import ChatMessage, modelapi
from inspect_ai.model._providers.vllm import VLLMAPI
from openai.types.chat import ChatCompletionMessageParam

K2_MODEL = "IFM/K2-Horizon-375B-A23B"
_THINKING_FIELDS = (
    "think",
    "reasoning",
    "reasoning_content",
    "think_fast",
    "think_faster",
)


def _ensure_thinking_fields(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    # Inspect omits empty reasoning on replay; this checkpoint's template requires
    # a thinking key on every assistant turn, including native tool-only turns.
    for message in messages:
        if message["role"] == "assistant" and not any(
            key in message for key in _THINKING_FIELDS
        ):
            message["reasoning_content"] = ""  # type: ignore[typeddict-unknown-key]
    return messages


@modelapi(name="k2-vllm")
class K2VLLMAPI(VLLMAPI):
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        if model_name != K2_MODEL:
            raise ValueError(f"k2-vllm only supports {K2_MODEL}")
        super().__init__(model_name=model_name, **kwargs)

    async def messages_to_openai(
        self, input: list[ChatMessage]
    ) -> list[ChatCompletionMessageParam]:
        return _ensure_thinking_fields(await super().messages_to_openai(input))
