"""Provider discovery and registry for voice endpoints.

Adapted from voicemode's provider_discovery.py for the proxy architecture.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import httpx

from src.voice.config import voice_config
from src.core.logging import logger


# Provider type detection
KOKORO_VOICES = [
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia",
    "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river",
    "af_sarah", "af_sky", "af_v0", "af_v0bella", "af_v0irulan",
    "af_v0nicole", "af_v0sarah", "af_v0sky", "am_adam", "am_echo",
    "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx",
    "am_puck", "am_santa", "am_v0adam", "am_v0gurney", "am_v0michael",
    "bf_alice", "bf_emma", "bf_lily", "bf_v0emma", "bf_v0isabella",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "bm_v0george",
    "bm_v0lewis", "ef_dora", "em_alex", "em_santa", "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi", "if_sara",
    "im_nicola", "jf_alpha", "jf_gongitsune", "jf_nezumi",
    "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
]

OPENAI_VOICES = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]


@dataclass
class EndpointInfo:
    base_url: str
    models: List[str] = field(default_factory=list)
    voices: List[str] = field(default_factory=list)
    provider_type: str = "unknown"
    last_check: Optional[str] = None
    last_error: Optional[str] = None


def _detect_provider_type(base_url: str) -> str:
    if not base_url:
        return "unknown"
    url_lower = base_url.lower()
    if "openai.com" in url_lower:
        return "openai"
    if ":8880" in url_lower:
        return "kokoro"
    if ":2022" in url_lower:
        return "whisper"
    if "localhost" in url_lower or "127.0.0.1" in url_lower:
        if base_url.rstrip("/").endswith("/v1"):
            port = base_url.rsplit(":", 1)[-1].split("/")[0]
            if port == "8880":
                return "kokoro"
            if port == "2022":
                return "whisper"
        return "local"
    return "unknown"


async def _discover_voices(base_url: str, api_key: str) -> List[str]:
    """Discover voices from a TTS endpoint."""
    provider_type = _detect_provider_type(base_url)

    if provider_type == "openai":
        return OPENAI_VOICES

    # Try /audio/voices endpoint (Kokoro/OpenAI-compatible)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = await client.get(
                f"{base_url.rstrip('/')}/audio/voices",
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "voices" in data:
                    return [v["id"] if isinstance(v, dict) else v for v in data["voices"]]
                elif isinstance(data, list):
                    return [v["id"] if isinstance(v, dict) else v for v in data]
    except Exception as e:
        logger.debug(f"Could not fetch voices from {base_url}: {e}")

    # Return default voices for known providers
    if provider_type == "kokoro":
        return KOKORO_VOICES

    return []


class ProviderRegistry:
    """Manages voice provider discovery and health."""

    def __init__(self):
        self._tts: Dict[str, EndpointInfo] = {}
        self._stt: Dict[str, EndpointInfo] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Probe all configured endpoints."""
        if self._initialized:
            return

        logger.info("Initializing voice provider registry...")

        # Discover TTS
        for url in voice_config.tts_base_urls:
            if not url:
                continue
            provider_type = _detect_provider_type(url)
            voices = await _discover_voices(url, voice_config.tts_api_key)
            self._tts[url] = EndpointInfo(
                base_url=url,
                models=["tts-1", "tts-1-hd"] if provider_type == "openai" else ["tts-1"],
                voices=voices,
                provider_type=provider_type,
                last_check=datetime.now(timezone.utc).isoformat(),
            )

        # Discover STT
        for url in voice_config.stt_base_urls:
            if not url:
                continue
            provider_type = _detect_provider_type(url)
            stt_models = ["whisper-1"]
            if provider_type == "openai":
                stt_models = ["whisper-1"]
            elif provider_type == "mlx-audio":
                stt_models = ["mlx-community/whisper-large-v3-turbo"]

            # Quick health check
            error = None
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    headers = {}
                    if voice_config.stt_api_key:
                        headers["Authorization"] = f"Bearer {voice_config.stt_api_key}"
                    resp = await client.get(
                        f"{url.rstrip('/')}/v1/models",
                        headers=headers
                    )
                    if resp.status_code != 200:
                        error = f"HTTP {resp.status_code}"
            except Exception as e:
                error = str(e)

            self._stt[url] = EndpointInfo(
                base_url=url,
                models=stt_models,
                provider_type=provider_type,
                last_check=datetime.now(timezone.utc).isoformat(),
                last_error=error,
            )

        self._initialized = True
        logger.info(
            f"Voice registry: {len(self._tts)} TTS, {len(self._stt)} STT endpoints"
        )

    def tts_endpoints(self) -> List[EndpointInfo]:
        """All TTS endpoints, in priority order."""
        return [self._tts[url] for url in voice_config.tts_base_urls if url in self._tts]

    def stt_endpoints(self) -> List[EndpointInfo]:
        """All STT endpoints, in priority order."""
        return [self._stt[url] for url in voice_config.stt_base_urls if url in self._stt]

    def get_first_tts(self) -> Optional[EndpointInfo]:
        """First healthy TTS endpoint."""
        for ep in self.tts_endpoints():
            if not ep.last_error:
                return ep
        return self.tts_endpoints()[0] if self.tts_endpoints() else None

    def get_first_stt(self) -> Optional[EndpointInfo]:
        """First healthy STT endpoint."""
        for ep in self.stt_endpoints():
            if not ep.last_error:
                return ep
        return self.stt_endpoints()[0] if self.stt_endpoints() else None

    def get_all_voices(self) -> List[str]:
        """All available voices from all TTS endpoints + profiles."""
        voices = set()
        for ep in self.tts_endpoints():
            voices.update(ep.voices)
        # Add OpenAI fallback voices
        voices.update(OPENAI_VOICES)
        return sorted(voices)

    def to_dict(self) -> Dict[str, Any]:
        """Registry state as JSON-friendly dict."""
        return {
            "tts": {
                url: {
                    "models": ep.models,
                    "voices": ep.voices,
                    "provider_type": ep.provider_type,
                    "last_check": ep.last_check,
                    "last_error": ep.last_error,
                }
                for url, ep in self._tts.items()
            },
            "stt": {
                url: {
                    "models": ep.models,
                    "provider_type": ep.provider_type,
                    "last_check": ep.last_check,
                    "last_error": ep.last_error,
                }
                for url, ep in self._stt.items()
            },
        }


# Global singleton
provider_registry = ProviderRegistry()
