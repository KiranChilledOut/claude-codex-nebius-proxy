import logging

from src.core.config import config
from src.models.claude import ClaudeMessage

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, config):
        self.config = config

    def contains_image_content(self, messages, *, latest_user_only: bool = False) -> bool:
        """Check if any (or just the latest user) message contains image content"""
        iterable = messages
        if latest_user_only:
            # Walk backward to the most recent user message only
            for message in reversed(messages):
                role = message.get("role") if isinstance(message, dict) else message.role
                if role == "user":
                    iterable = [message]
                    break

        for message in iterable:
            if isinstance(message, dict):
                content = message.get("content", [])
            else:
                content = message.content
            if isinstance(content, dict):
                content = [content]

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if (
                            block.get("type") in ("image", "input_image", "image_url")
                            or "image_url" in block
                        ):
                            return True
                    else:
                        # Check if it's a ClaudeContentBlockImage object
                        if hasattr(block, "type") and block.type in (
                            "image",
                            "input_image",
                            "image_url",
                        ):
                            return True
        return False

    def map_claude_model_to_openai(self, claude_model: str, messages=None) -> str:
        """Map Claude model names to OpenAI model names based on BIG/SMALL pattern"""

        # If messages contain images, route to vision model
        if messages and self.contains_image_content(messages, latest_user_only=True):
            return self.config.vision_model

        # If it's already an OpenAI model, return as-is
        if claude_model.startswith("gpt-") or claude_model.startswith("o1-"):
            return claude_model

        # If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if (
            claude_model.startswith("ep-")
            or claude_model.startswith("doubao-")
            or claude_model.startswith("deepseek-")
        ):
            return claude_model

        # Map based on model naming patterns
        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return self.config.small_model
        elif "sonnet" in model_lower:
            return self.config.middle_model
        elif "opus" in model_lower:
            return self.config.big_model
        else:
            # Default to big model for unknown models
            return self.config.big_model

    def map_codex_model(self, codex_model: str) -> str:
        """Map Codex (OpenAI) model names to backend OpenAI model names."""
        lower = codex_model.lower()
        if "mini" in lower:
            return self.config.small_model
        # "gpt-4", "gpt-5", "o1", etc. all map to big model
        if (
            codex_model.startswith("gpt-")
            or codex_model.startswith("o1-")
            or codex_model.startswith("o3-")
        ):
            return self.config.big_model
        # Default to big model for unknown models
        return self.config.big_model

    def long_context_enabled(self) -> bool:
        """True when LONG_CONTEXT_MODEL is set and a positive threshold exists.
        Lets callers skip prompt-token counting entirely when the feature is off."""
        return bool(
            getattr(self.config, "long_context_model", "")
            and getattr(self.config, "long_context_threshold", 0) > 0
        )

    def apply_long_context(self, base_model: str, prompt_tokens: int) -> str:
        """Escalate to the long-context model when the prompt exceeds the
        configured threshold. No-op when the feature is disabled or the base
        model is already the long-context model."""
        lc_model = getattr(self.config, "long_context_model", "")
        threshold = getattr(self.config, "long_context_threshold", 0)
        if lc_model and threshold > 0 and base_model != lc_model and prompt_tokens > threshold:
            logger.info(
                "Long-context routing: prompt ~%d tokens > %d; routing %s -> %s",
                prompt_tokens,
                threshold,
                base_model,
                lc_model,
            )
            return lc_model
        return base_model


model_manager = ModelManager(config)
