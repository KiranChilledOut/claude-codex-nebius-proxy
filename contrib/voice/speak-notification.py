#!/usr/bin/env python3
"""Claude Code Notification hook: speak blocked-state notifications.

Called when Claude is waiting for user input (e.g. permission prompt).
Uses proxy TTS when available, falls back to macOS say.

Mute:  touch ~/.claude/voice/off      (disable)
        rm ~/.claude/voice/off         (re-enable)
"""

import json
import os
import subprocess
import sys
import urllib.request


VOICE_DIR = os.path.expanduser("~/.claude/voice")
SAY_PID_FILE = "/tmp/claude-voice-say.pid"
VOICE_RATE = int(os.environ.get("CLAUDE_VOICE_RATE", "190"))
VOICE_NAME = os.environ.get("CLAUDE_VOICE_NAME", "")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_river")


def _proxy_url(path: str) -> str:
    proxy = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    if not proxy:
        return ""
    return f"{proxy}/voice{path}"


def _proxy_available() -> bool:
    url = _proxy_url("/health")
    if not url:
        return False
    try:
        req = urllib.request.Request(url, method="GET")
        token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "claude-local")
        req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _speak(text: str) -> None:
    if _proxy_available():
        try:
            url = _proxy_url("/tts")
            data = json.dumps({"input": text, "voice": TTS_VOICE}).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('ANTHROPIC_AUTH_TOKEN', 'claude-local')}",
                },
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                audio = resp.read()
                proc = subprocess.Popen(
                    ["/usr/bin/afplay", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                proc.stdin.write(audio)
                proc.stdin.close()
                return
        except Exception:
            pass  # Fallback to say

    cmd = ["/usr/bin/say", "-r", str(VOICE_RATE)]
    if VOICE_NAME:
        cmd += ["-v", VOICE_NAME]
    cmd.append(text)
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def main() -> int:
    if os.path.exists(os.path.join(VOICE_DIR, "off")):
        return 0

    raw = sys.stdin.read()
    if not raw.strip():
        return 0

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    # Notification payloads have a message field
    msg = payload.get("message")
    if msg:
        _speak(f"Notification: {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
