# Voice Features

The proxy exposes voice endpoints (`/voice/*`) and a `claude --voice` shell
shortcut for hands-free conversations. All configuration is via environment
variables.

## Quick Start

```bash
# 1. Enable voice in .env
VOICE_ENABLED=true
TTS_PROVIDER_URL=http://localhost:8880   # Kokoro (default)
STT_PROVIDER_URL=http://localhost:2022   # Whisper.cpp (default)

# 2. Start the proxy
python start_proxy.py

# 3. Use voice
claude --voice            # Direct Anthropic, hands-free
claude --voice --proxy    # Via Nebius proxy, hands-free
claudius --voice          # Alias
```

## Architecture

| Layer | Component | Notes |
|-------|-----------|-------|
| **Client** | `scripts/claude-voice.py` | tmux wrapper, uses proxy TTS/STT or `hear`/`say` |
| **Proxy** | `src/voice/` | FastAPI endpoints: `/voice/tts`, `/voice/stt`, `/voice/profiles` |
| **Providers** | Kokoro, Whisper, OpenAI | OpenAI-compatible local or cloud endpoints |

## Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/voice/health` | GET | Provider status JSON |
| `/voice/voices` | GET | Available voices + clone profiles |
| `/voice/tts` | POST | Text→speech (returns audio bytes) |
| `/voice/stt` | POST | Speech→text (multipart upload) |
| `/voice/profiles` | GET | List voice cloning profiles |
| `/voice/profiles/{name}` | GET | Profile details |
| `/voice/profiles/{name}/tts` | POST | TTS with cloned voice |

### POST /voice/tts

Parameters (form data):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize |
| `voice` | string | `TTS_VOICE` | Voice identifier |
| `model` | string | `TTS_MODEL` | Provider model |
| `response_format` | string | `mp3` | `mp3`, `wav`, `opus`, `aac`, `flac`, `pcm` |
| `speed` | float | 1.0 | Playback speed |
| `stream` | bool | false | Chunked HTTP streaming |

### POST /voice/stt

Upload audio file via multipart form-data. Field name: `file`.

Returns: `{"text": "transcribed text", "language": null}`

## Configuration

All settings live in `.env` (same file as proxy core).

```
# Master switch
VOICE_ENABLED=false

# TTS — local Kokoro or OpenAI fallback
TTS_PROVIDER_URL=http://localhost:8880
TTS_FALLBACK_URL=https://api.openai.com/v1
TTS_VOICE=af_river
TTS_MODEL=kokoro
TTS_RESPONSE_FORMAT=mp3
TTS_SPEED=1.0

# STT — local Whisper.cpp or OpenAI fallback
STT_PROVIDER_URL=http://localhost:2022
STT_FALLBACK_URL=

# Voice cloning profiles (filesystem-based)
VOICE_PROFILES_DIR=~/.claude/voice/profiles

# Client fallback paths (if proxy TTS/STT unavailable)
HEAR_BINARY_PATH=~/.claude/voice/bin/hear
SAY_BINARY_PATH=/usr/bin/say

# Conversation turn controls
CLAUDE_VOICE_LOCALE=en-US
CLAUDE_VOICE_SILENCE=2
CLAUDE_VOICE_MAX_CHARS=600
CLAUDE_VOICE_RATE=190
CLAUDE_VOICE_NAME=
```

## Voice Cloning Profiles

Filesystem-based cloning (adapted from voicemode):

```
~/.claude/voice/profiles/
  samantha/
    default.wav          # reference audio
    default.txt          # transcript of the audio
    description.txt      # optional description
```

Access profile via: `voice="samantha"` in TTS requests or
`POST /voice/profiles/samantha/tts`.

## Shell Shortcuts

After running the installer (`./install.sh`), your shell supports:

| Command | Description |
|---------|-------------|
| `claude` | Direct login |
| `claude --proxy` | Via proxy |
| `claude --voice` | Hands-free, direct |
| `claude --voice --proxy` | Hands-free, via proxy |
| `claudius` | Alias for `--proxy` |
| `claudio` | Alias for `--voice` |
| `claudio-proxy` | Alias for `--voice --proxy` |

## Client-Side Fallback

When proxy voice endpoints are unavailable (e.g. proxy not running), the
`claude-voice.py` wrapper falls back to:

| Direction | Tool | Requirement |
|-----------|------|-------------|
| Dictation in | `hear` CLI | macOS, from github.com/sveinbjornt/hear |
| Speech out | `say` | Built-in macOS |

Install `hear`:
```bash
mkdir -p ~/.claude/voice/bin
curl -sL -o /tmp/hear.zip https://github.com/sveinbjornt/hear/releases/download/0.8/hear-0.8.zip
unzip -j /tmp/hear.zip -d ~/.claude/voice/bin '*/hear'
chmod 755 ~/.claude/voice/bin/hear
```

## Hooks

Install for spoken replies:
```bash
cp contrib/voice/speak-reply.py contrib/voice/speak-notification.py ~/.claude/voice/
chmod +x ~/.claude/voice/*.py
```

Add to `~/.claude/settings.json` (hooks must be **sync**):
```json
{
  "hooks": {
    "Stop": [{"hooks": [{"type": "command", "command": "~/.claude/voice/speak-reply.py", "timeout": 10}]}],
    "Notification": [{"hooks": [{"type": "command", "command": "~/.claude/voice/speak-notification.py", "timeout": 10}]}]
  }
}
```

The hook scripts automatically detect and use proxy TTS when available, or
fall back to `say`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Voice endpoints 404 | `VOICE_ENABLED=true` must be set in `.env` |
| Proxy TTS fails | Check `TTS_PROVIDER_URL` is reachable; set `TTS_FALLBACK_URL` |
| `hear` not found | Install per instructions above, or use proxy STT endpoint |
| Nothing spoken | Check `"hooks"` in settings.json are **sync** (not `"async": true`) |
| Mute switch | `touch ~/.claude/voice/off` disables speech; remove to re-enable |
