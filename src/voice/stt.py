"""Speech-to-Text endpoint logic.

Wraps STT providers (Whisper.cpp, OpenAI) with failover.
"""

import io
import httpx

from src.voice.config import voice_config
from src.voice.providers import provider_registry
from src.core.logging import logger


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Transcribe audio bytes to text.

    Tries endpoints in priority order.
    Returns transcription text.
    """
    await provider_registry.initialize()

    for endpoint in provider_registry.stt_endpoints():
        try:
            text = await _call_stt_endpoint(
                endpoint.base_url,
                audio_bytes=audio_bytes,
                filename=filename,
                api_key=voice_config.stt_api_key,
            )
            return text
        except Exception as e:
            logger.warning(f"STT endpoint {endpoint.base_url} failed: {e}")
            continue

    # Fallback to OpenAI if API key present
    if voice_config.stt_api_key and "openai.com" not in voice_config.stt_provider_url:
        try:
            text = await _call_stt_endpoint(
                "https://api.openai.com/v1",
                audio_bytes=audio_bytes,
                filename=filename,
                api_key=voice_config.stt_api_key,
            )
            return text
        except Exception as e:
            logger.error(f"OpenAI STT fallback failed: {e}")

    raise RuntimeError("No STT provider available")


async def _call_stt_endpoint(
    base_url: str,
    audio_bytes: bytes,
    filename: str,
    api_key: str,
) -> str:
    """Call a single STT endpoint."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Build multipart form
        files = {
            "file": (filename, io.BytesIO(audio_bytes), f"audio/{filename.split('.')[-1]}"),
            "model": (None, "whisper-1"),
        }

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = await client.post(
            f"{base_url.rstrip('/')}/audio/transcriptions",
            files=files,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "")

