"""Text-to-Speech endpoint logic.

Wraps TTS providers (Kokoro, OpenAI) with transparent failover.
"""

import io
from typing import Optional, AsyncIterator

import httpx

from src.voice.config import voice_config
from src.voice.providers import provider_registry
from src.core.logging import logger


async def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
) -> tuple[bytes, str]:
    """Synthesize speech, returning (audio_bytes, content_type).

    Tries endpoints in priority order.
    """
    voice = voice or voice_config.tts_voice
    model = model or voice_config.tts_model
    response_format = response_format or voice_config.tts_response_format
    speed = speed or voice_config.tts_speed

    await provider_registry.initialize()

    audio_format = response_format.lower()
    content_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
    }
    content_type = content_type_map.get(audio_format, "audio/mpeg")

    for endpoint in provider_registry.tts_endpoints():
        try:
            audio = await _call_tts_endpoint(
                endpoint.base_url,
                text=text,
                voice=voice,
                model=model,
                response_format=audio_format,
                speed=speed,
                api_key=voice_config.tts_api_key,
            )
            return audio, content_type
        except Exception as e:
            logger.warning(f"TTS endpoint {endpoint.base_url} failed: {e}")
            continue

    # All endpoints failed — fall back to OpenAI if API key present
    if voice_config.tts_api_key and "openai.com" not in voice_config.tts_provider_url:
        try:
            audio = await _call_tts_endpoint(
                "https://api.openai.com/v1",
                text=text,
                voice=voice,
                model="tts-1",
                response_format=audio_format,
                speed=speed,
                api_key=voice_config.tts_api_key,
            )
            return audio, content_type
        except Exception as e:
            logger.error(f"OpenAI TTS fallback failed: {e}")

    raise RuntimeError("No TTS provider available")


async def synthesize_with_profile(
    text: str,
    profile_name: str,
    response_format: Optional[str] = None,
) -> tuple[bytes, str]:
    """Synthesize using a cloned voice profile (mlx-audio / Qwen3-TTS style)."""
    from src.voice.profiles import get_profile

    profile = get_profile(profile_name)
    if not profile:
        raise ValueError(f"Voice profile '{profile_name}' not found")

    response_format = response_format or voice_config.tts_response_format
    audio_format = response_format.lower()
    content_type_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "pcm": "audio/pcm",
    }
    content_type = content_type_map.get(audio_format, "audio/mpeg")

    # Call the clone endpoint (mlx-audio / Qwen3-TTS compatible)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": profile.model,
                "input": text,
                "voice": profile.ref_audio,
                "ref_text": profile.ref_text,
                "response_format": audio_format,
            }
            if voice_config.tts_speed != 1.0:
                payload["speed"] = voice_config.tts_speed

            headers = {}
            clone_key = voice_config.tts_api_key or "dummy-key"
            headers["Authorization"] = f"Bearer {clone_key}"
            headers["Content-Type"] = "application/json"

            resp = await client.post(
                f"{profile.base_url.rstrip('/')}/audio/speech",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.content, content_type
    except Exception as e:
        logger.error(f"Clone TTS failed for profile '{profile_name}': {e}")
        raise


async def stream_speech(
    text: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    response_format: Optional[str] = None,
    speed: Optional[float] = None,
) -> AsyncIterator[bytes]:
    """Stream TTS audio chunks (for chunked HTTP transfer)."""
    voice = voice or voice_config.tts_voice
    model = model or voice_config.tts_model
    response_format = response_format or voice_config.tts_response_format
    speed = speed or voice_config.tts_speed

    await provider_registry.initialize()

    for endpoint in provider_registry.tts_endpoints():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": model,
                    "input": text,
                    "voice": voice,
                    "response_format": response_format.lower(),
                }
                if speed != 1.0:
                    payload["speed"] = speed

                headers = {}
                if voice_config.tts_api_key:
                    headers["Authorization"] = f"Bearer {voice_config.tts_api_key}"

                # OpenAI-compatible streaming
                if response_format.lower() == "pcm":
                    payload["stream"] = True

                async with client.stream(
                    "POST",
                    f"{endpoint.base_url.rstrip('/')}/audio/speech",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    async for chunk in resp.aiter_bytes(
                        chunk_size=voice_config.stream_chunk_size
                    ):
                        yield chunk
                return
        except Exception as e:
            logger.warning(f"TTS stream endpoint {endpoint.base_url} failed: {e}")
            continue

    raise RuntimeError("No TTS provider available for streaming")


async def _call_tts_endpoint(
    base_url: str,
    text: str,
    voice: str,
    model: str,
    response_format: str,
    speed: float,
    api_key: str,
) -> bytes:
    """Call a single TTS endpoint."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }
        if speed != 1.0:
            payload["speed"] = speed

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        resp = await client.post(
            f"{base_url.rstrip('/')}/audio/speech",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        return resp.content
