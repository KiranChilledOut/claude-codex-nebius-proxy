"""Voice module for Claude Code Proxy.

Provides TTS, STT, voice profiles, and provider discovery endpoints.
All configuration via environment variables from .env.
"""

from src.voice.config import voice_config
from src.voice.providers import provider_registry

__all__ = ["voice_config", "provider_registry"]
