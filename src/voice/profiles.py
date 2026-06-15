"""Voice cloning profiles — filesystem-based.

Adapted from voicemode's voice_profiles.py.
Voices live in ``~/.claude/voice/profiles`` (or ``VOICE_PROFILES_DIR``).
Each subdirectory is a voice profile with ``default.wav`` + ``default.txt``.
"""

import logging
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.voice.config import voice_config
from src.core.logging import logger


_VOICE_PROFILES_DIR = Path(
    os.path.expanduser(
        voice_config.profiles_dir or "~/.claude/voice/profiles"
    )
)

# Default clone endpoint (mlx-audio Qwen3-TTS style)
DEFAULT_CLONE_BASE_URL = os.environ.get(
    "VOICE_CLONE_BASE_URL", "http://127.0.0.1:8890/v1"
)
DEFAULT_CLONE_MODEL = os.environ.get(
    "VOICE_CLONE_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit"
)

_REMOTE_PROFILES_DIR = voice_config.remote_profiles_dir

_INDEX_RE = re.compile(r"^([^/\[\]]+)\[(\d+)\]$")


@dataclass
class VoiceProfile:
    name: str
    ref_audio: str
    ref_text: str
    model: str
    base_url: str
    description: str = ""
    voice_dir: str = ""


_profiles: Dict[str, VoiceProfile] = {}
_loaded = False


def _resolve_default_wav(voice_dir: Path) -> Optional[Path]:
    default = voice_dir / "default.wav"
    if default.exists():
        return default
    wavs = sorted(voice_dir.glob("*.wav"))
    if not wavs:
        return None
    if len(wavs) == 1:
        return wavs[0]
    return None


def _resolve_transcript(wav_path: Path) -> str:
    same_name = wav_path.with_suffix(".txt")
    if same_name.exists():
        return same_name.read_text().strip()
    fallback = wav_path.parent / "default.txt"
    if fallback.exists():
        return fallback.read_text().strip()
    return ""


def _read_description(voice_dir: Path) -> str:
    desc_path = voice_dir / "description.txt"
    if desc_path.exists():
        return desc_path.read_text().strip()
    return ""


def _translate_path(local_path: Path) -> str:
    """Translate local path to remote path if REMOTE_PROFILES_DIR is set."""
    if not _REMOTE_PROFILES_DIR:
        return str(local_path.resolve() if local_path.exists() else local_path)
    try:
        rel = local_path.resolve().relative_to(_VOICE_PROFILES_DIR.resolve())
    except ValueError:
        return str(local_path.resolve())
    return str(Path(_REMOTE_PROFILES_DIR) / rel)


def _build_profile(voice_dir: Path, wav: Path) -> VoiceProfile:
    transcript = _resolve_transcript(wav)
    if not transcript:
        logger.warning(
            f"Voice profile '{voice_dir.name}' has no transcript. "
            f"TTS quality may be reduced."
        )
    return VoiceProfile(
        name=voice_dir.name,
        ref_audio=_translate_path(wav),
        ref_text=transcript,
        model=DEFAULT_CLONE_MODEL,
        base_url=DEFAULT_CLONE_BASE_URL,
        description=_read_description(voice_dir),
        voice_dir=str(voice_dir),
    )


def _load_dir_profiles() -> Dict[str, VoiceProfile]:
    profiles: Dict[str, VoiceProfile] = {}
    seen: Dict[str, Path] = {}
    conflicts: Dict[str, List[Path]] = {}

    if not _VOICE_PROFILES_DIR.exists():
        return profiles

    def walk(dir_path: Path) -> None:
        wav = _resolve_default_wav(dir_path)
        if wav is not None:
            leaf = dir_path.name
            if leaf in seen:
                conflicts.setdefault(leaf, [seen[leaf]]).append(dir_path)
                return
            seen[leaf] = dir_path
            profiles[leaf] = _build_profile(dir_path, wav)
            return
        for child in sorted(p for p in dir_path.iterdir() if p.is_dir()):
            walk(child)

    for top in sorted(p for p in _VOICE_PROFILES_DIR.iterdir() if p.is_dir()):
        walk(top)

    for leaf, paths in conflicts.items():
        rel_paths = [str(p.relative_to(_VOICE_PROFILES_DIR)) for p in paths]
        logger.error(
            f"Voice name collision: '{leaf}' at {rel_paths}. "
            f"All conflicting profiles dropped."
        )
        profiles.pop(leaf, None)

    return profiles


def load_profiles() -> Dict[str, VoiceProfile]:
    global _profiles, _loaded
    _profiles = _load_dir_profiles()
    _loaded = True
    if _profiles:
        logger.info(f"Loaded {len(_profiles)} voice profiles: {list(_profiles.keys())}")
    return _profiles


def parse_voice_expr(expr: str) -> Tuple[Optional[str], Optional[str]]:
    if not expr:
        return None, None
    if expr.startswith("/"):
        return None, expr
    m = _INDEX_RE.match(expr)
    if m:
        return m.group(1), f"[{m.group(2)}]"
    if "/" in expr:
        head, _, tail = expr.partition("/")
        return head, tail
    return expr, None


def _list_samples(voice_dir: Path) -> List[Path]:
    return sorted(voice_dir.glob("*.wav"))


def resolve_voice_expr(expr: str) -> Optional[VoiceProfile]:
    if not _loaded:
        load_profiles()

    name, selector = parse_voice_expr(expr)

    # Absolute path
    if name is None and selector and selector.startswith("/"):
        clip = Path(selector)
        sidecar = _resolve_transcript(clip) if clip.exists() else ""
        return VoiceProfile(
            name=expr,
            ref_audio=selector,
            ref_text=sidecar,
            model=DEFAULT_CLONE_MODEL,
            base_url=DEFAULT_CLONE_BASE_URL,
            description="(absolute path)",
        )

    if not name:
        return None

    profile = _profiles.get(name)
    if profile is None:
        return None

    if selector is None:
        return profile

    voice_dir = Path(profile.voice_dir) if profile.voice_dir else _VOICE_PROFILES_DIR / name

    if selector.startswith("[") and selector.endswith("]"):
        try:
            idx = int(selector[1:-1])
        except ValueError:
            return profile
        samples = _list_samples(voice_dir)
        if not samples:
            return profile
        if 0 <= idx < len(samples):
            wav = samples[idx]
            return replace(
                profile,
                ref_audio=_translate_path(wav),
                ref_text=_resolve_transcript(wav),
            )
        return profile

    # Explicit file
    wav = voice_dir / selector
    return replace(
        profile,
        ref_audio=_translate_path(wav),
        ref_text=_resolve_transcript(wav) if wav.exists() else "",
    )


def get_profile(name: str) -> Optional[VoiceProfile]:
    return resolve_voice_expr(name)


def list_profiles() -> Dict[str, VoiceProfile]:
    if not _loaded:
        load_profiles()
    return _profiles


def is_clone_voice(expr: str) -> bool:
    if not _loaded:
        load_profiles()
    if not expr:
        return False
    name, selector = parse_voice_expr(expr)
    if name is None and selector and selector.startswith("/"):
        return True
    return name in _profiles


def reload_profiles() -> Dict[str, VoiceProfile]:
    global _loaded
    _loaded = False
    return load_profiles()
