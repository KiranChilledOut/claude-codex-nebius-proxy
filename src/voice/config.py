"""Voice configuration — centralized env-var loading.

All voice settings are read from the existing .env file, same pattern as
src.core.config. No separate config file needed.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    return val in ("1", "true", "yes") if val != "" else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


@dataclass
class VoiceConfig:
    """Voice configuration — all env vars in one place."""

    # Master switch
    enabled: bool = False

    # TTS settings
    tts_provider_url: str = "http://localhost:8880"
    tts_fallback_url: str = ""
    tts_voice: str = "af_river"
    tts_model: str = "kokoro"
    tts_response_format: str = "mp3"
    tts_speed: float = 1.0
    tts_api_key: str = ""

    # STT settings
    stt_provider_url: str = "http://localhost:2022"
    stt_fallback_url: str = ""
    stt_api_key: str = ""

    # Streaming audio playback
    streaming: bool = True
    stream_chunk_size: int = 4096
    stream_buffer_ms: int = 200
    stream_max_buffer: float = 2.0

    # Voice profiles (cloning)
    profiles_dir: str = ""
    remote_profiles_dir: str = ""  # For remote TTS servers

    # Client-side / fallback paths
    hear_binary_path: str = ""
    say_binary_path: str = "/usr/bin/say"

    # Turn controls (for claude-voice wrapper)
    voice_locale: str = "en-US"
    voice_silence_timeout: int = 2
    voice_max_chars: int = 600
    voice_rate: int = 190
    voice_name: str = ""

    # Provider discovery
    tts_base_urls: List[str] = field(default_factory=list)
    stt_base_urls: List[str] = field(default_factory=list)

    def __post_init__(self):
        # Resolve profiles_dir default
        if not self.profiles_dir:
            self.profiles_dir = os.path.expanduser("~/.claude/voice/profiles")

        if not self.hear_binary_path:
            self.hear_binary_path = os.path.expanduser("~/.claude/voice/bin/hear")

        # Build provider URL lists for discovery
        urls: List[str] = []
        if self.tts_provider_url:
            urls.append(self.tts_provider_url)
        if self.tts_fallback_url:
            urls.append(self.tts_fallback_url)
        self.tts_base_urls = urls

        stt_urls: List[str] = []
        if self.stt_provider_url:
            stt_urls.append(self.stt_provider_url)
        if self.stt_fallback_url:
            stt_urls.append(self.stt_fallback_url)
        self.stt_base_urls = stt_urls

        # TTS API key defaults to OPENAI_API_KEY if not set
        if not self.tts_api_key:
            self.tts_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.stt_api_key:
            self.stt_api_key = os.environ.get("OPENAI_API_KEY", "").strip()


def _load() -> VoiceConfig:
    """Load voice config from environment."""
    return VoiceConfig(
        enabled=_env_bool("VOICE_ENABLED", False),
        tts_provider_url=os.environ.get("TTS_PROVIDER_URL", "http://localhost:8880"),
        tts_fallback_url=os.environ.get("TTS_FALLBACK_URL", "").strip(),
        tts_voice=os.environ.get("TTS_VOICE", "af_river"),
        tts_model=os.environ.get("TTS_MODEL", "kokoro"),
        tts_response_format=os.environ.get("TTS_RESPONSE_FORMAT", "mp3"),
        tts_speed=_env_float("TTS_SPEED", 1.0),
        tts_api_key=os.environ.get("TTS_API_KEY", "").strip(),
        stt_provider_url=os.environ.get("STT_PROVIDER_URL", "http://localhost:2022"),
        stt_fallback_url=os.environ.get("STT_FALLBACK_URL", "").strip(),
        stt_api_key=os.environ.get("STT_API_KEY", "").strip(),
        streaming=_env_bool("VOICE_STREAMING", True),
        stream_chunk_size=_env_int("VOICE_STREAM_CHUNK_SIZE", 4096),
        stream_buffer_ms=_env_int("VOICE_STREAM_BUFFER_MS", 200),
        stream_max_buffer=_env_float("VOICE_STREAM_MAX_BUFFER", 2.0),
        profiles_dir=os.environ.get("VOICE_PROFILES_DIR", "").strip(),
        remote_profiles_dir=os.environ.get("VOICE_REMOTE_PROFILES_DIR", "").strip(),
        hear_binary_path=os.environ.get("HEAR_BINARY_PATH", "").strip(),
        say_binary_path=os.environ.get("SAY_BINARY_PATH", "/usr/bin/say"),
        voice_locale=os.environ.get("CLAUDE_VOICE_LOCALE", "en-US"),
        voice_silence_timeout=_env_int("CLAUDE_VOICE_SILENCE", 2),
        voice_max_chars=_env_int("CLAUDE_VOICE_MAX_CHARS", 600),
        voice_rate=_env_int("CLAUDE_VOICE_RATE", 190),
        voice_name=os.environ.get("CLAUDE_VOICE_NAME", "").strip(),
    )


voice_config: VoiceConfig = _load()
