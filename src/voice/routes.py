"""FastAPI router for voice endpoints.

All endpoints under /voice/* — conditionally registered when VOICE_ENABLED.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, Form, File
from fastapi.responses import Response, StreamingResponse

from src.voice.config import voice_config
from src.voice.providers import provider_registry
from src.voice import tts as tts_module
from src.voice import stt as stt_module
from src.voice.profiles import (
    load_profiles,
    list_profiles,
    get_profile,
    is_clone_voice,
)
from src.core.logging import logger

router = APIRouter()


@router.get("/health", response_model=dict)
async def voice_health():
    """Voice provider health status."""
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    await provider_registry.initialize()
    return provider_registry.to_dict()


@router.get("/voices", response_model=dict)
async def list_voices():
    """List available TTS voices from all providers."""
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    await provider_registry.initialize()
    voices = provider_registry.get_all_voices()
    profiles = list_profiles()
    return {
        "voices": voices,
        "clone_profiles": list(profiles.keys()),
        "default_voice": voice_config.tts_voice,
        "default_model": voice_config.tts_model,
    }


@router.post("/tts")
async def text_to_speech(
    input: str = Form(..., description="Text to synthesize"),
    voice: Optional[str] = Form(None, description="Voice identifier"),
    model: Optional[str] = Form(None, description="TTS model"),
    response_format: Optional[str] = Form(None, description="Audio format (mp3, wav, opus, etc)"),
    speed: Optional[float] = Form(None, description="Playback speed"),
    stream: bool = Form(False, description="Stream response"),
):
    """Convert text to speech.

    Returns audio bytes with appropriate Content-Type header.
    If stream=True, returns chunked streaming response.
    """
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    target_voice = voice or voice_config.tts_voice

    # Check if it's a clone voice profile
    if is_clone_voice(target_voice):
        try:
            audio, content_type = await tts_module.synthesize_with_profile(
                text=input,
                profile_name=target_voice,
                response_format=response_format or voice_config.tts_response_format,
            )
            return Response(
                content=audio,
                media_type=content_type,
                headers={"x-voice-provider": "clone"},
            )
        except Exception as e:
            logger.error(f"Clone TTS failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    if stream and voice_config.streaming:
        # Streaming response
        async def audio_stream():
            async for chunk in tts_module.stream_speech(
                text=input,
                voice=target_voice,
                model=model or voice_config.tts_model,
                response_format=response_format or voice_config.tts_response_format,
                speed=speed or voice_config.tts_speed,
            ):
                yield chunk

        return StreamingResponse(
            audio_stream(),
            media_type="audio/pcm" if (response_format or voice_config.tts_response_format) == "pcm" else "audio/mpeg",
            headers={"x-voice-provider": "streaming"},
        )

    # Buffer-then-return
    try:
        audio, content_type = await tts_module.synthesize_speech(
            text=input,
            voice=target_voice,
            model=model or voice_config.tts_model,
            response_format=response_format or voice_config.tts_response_format,
            speed=speed or voice_config.tts_speed,
        )
        return Response(
            content=audio,
            media_type=content_type,
            headers={"x-voice-provider": "sync"},
        )
    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stt", response_model=dict)
async def speech_to_text(
    file: UploadFile = File(..., description="Audio file to transcribe"),
):
    """Transcribe uploaded audio to text.

    Supports WAV, MP3, FLAC, OGG, M4A formats.
    """
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    try:
        audio_bytes = await file.read()
        text = await stt_module.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
        )
        return {"text": text, "language": None}
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles", response_model=dict)
async def list_voice_profiles():
    """List all voice cloning profiles."""
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    profiles = load_profiles()
    return {
        "profiles": {
            name: {
                "name": p.name,
                "description": p.description,
                "ref_audio": p.ref_audio,
                "model": p.model,
                "base_url": p.base_url,
            }
            for name, p in profiles.items()
        }
    }


@router.get("/profiles/{name}", response_model=dict)
async def get_voice_profile(name: str):
    """Get details of a specific voice profile."""
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    profile = get_profile(name)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Voice profile '{name}' not found")

    return {
        "name": profile.name,
        "description": profile.description,
        "ref_audio": profile.ref_audio,
        "ref_text": profile.ref_text,
        "model": profile.model,
        "base_url": profile.base_url,
    }


@router.post("/profiles/{name}/tts")
async def profile_text_to_speech(
    name: str,
    input: str = Form(..., description="Text to synthesize"),
    response_format: Optional[str] = Form(None),
):
    """Synthesize speech using a cloned voice profile."""
    if not voice_config.enabled:
        raise HTTPException(status_code=404, detail="Voice features not enabled")

    try:
        audio, content_type = await tts_module.synthesize_with_profile(
            text=input,
            profile_name=name,
            response_format=response_format or voice_config.tts_response_format,
        )
        return Response(
            content=audio,
            media_type=content_type,
            headers={"x-voice-name": name, "x-voice-provider": "clone"},
        )
    except Exception as e:
        logger.error(f"Profile TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
