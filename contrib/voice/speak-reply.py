#!/usr/bin/env python3
"""Claude Code Stop hook: speak the assistant's final reply.

Reads the hook payload from stdin, pulls the last assistant text from the
transcript, cleans it (strips markdown, code blocks, URLs), and speaks it.

When the proxy's voice endpoints are available (/voice/tts), uses them
for higher-quality TTS (Kokoro, voice profiles). Falls back to macOS `say`.

Mute:  touch ~/.claude/voice/off      (disable speech)
        rm ~/.claude/voice/off         (re-enable)
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request


VOICE_DIR = os.path.expanduser("~/.claude/voice")
SAY_PID_FILE = "/tmp/claude-voice-say.pid"
MAX_CHARS = int(os.environ.get("CLAUDE_VOICE_MAX_CHARS", "600"))
VOICE_RATE = int(os.environ.get("CLAUDE_VOICE_RATE", "190"))
VOICE_NAME = os.environ.get("CLAUDE_VOICE_NAME", "")
TTS_VOICE = os.environ.get("TTS_VOICE", "af_river")


def _proxy_url(path: str) -> str:
    """Build proxy voice endpoint URL from environment."""
    proxy = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    if not proxy:
        return ""
    return f"{proxy}/voice{path}"


def _proxy_available() -> bool:
    """Check if proxy voice TTS is available."""
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


def last_assistant_text(transcript_path: str) -> str:
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    for raw in reversed(lines):
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if entry.get("type") != "assistant":
            continue
        content = (entry.get("message") or {}).get("content") or []
        texts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(t for t in texts if t.strip())
        if text.strip():
            return text
    return ""


def clean_for_speech(text: str, max_chars: int) -> str:
    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "link", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_>|#]+", " ", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        cut = text[:max_chars]
        m = re.search(r"^.*[.!?]", cut, flags=re.DOTALL)
        text = (m.group(0) if m and len(m.group(0)) > max_chars * 0.4 else cut) + " ... more in the terminal."
    return text


def stop_previous_say() -> None:
    try:
        with open(SAY_PID_FILE) as f:
            os.kill(int(f.read().strip()), signal.SIGTERM)
    except (OSError, ValueError):
        pass


def speak_via_proxy(text: str) -> bool:
    """Try to speak via proxy TTS endpoint. Returns True on success."""
    url = _proxy_url("/tts")
    if not url:
        return False
    try:
        data = json.dumps({
            "input": text,
            "voice": TTS_VOICE,
        }).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.environ.get('ANTHROPIC_AUTH_TOKEN', 'claude-local')}",
            },
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            audio = resp.read()
            # Play via afplay
            proc = subprocess.Popen(
                ["/usr/bin/afplay", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            proc.stdin.write(audio)
            proc.stdin.close()
            try:
                with open(SAY_PID_FILE, "w") as f:
                    f.write(str(proc.pid))
            except OSError:
                pass
            return True
    except Exception as e:
        # Silently fall back — proxy TTS failure shouldn't break the hook
        return False


def speak_via_say(text: str) -> None:
    """Speak via macOS say command."""
    cmd = ["/usr/bin/say", "-r", str(VOICE_RATE)]
    if VOICE_NAME:
        cmd += ["-v", VOICE_NAME]
    cmd.append(text)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        with open(SAY_PID_FILE, "w") as f:
            f.write(str(proc.pid))
    except OSError:
        pass


def speak(text: str) -> None:
    """Speak text. Tries proxy TTS first, then say fallback."""
    if _proxy_available():
        if speak_via_proxy(text):
            return
    speak_via_say(text)


def _trace(msg: str) -> None:
    try:
        import datetime
        with open("/tmp/claude-voice-hook.log", "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} speak-reply {msg}\n")
    except OSError:
        pass


def main() -> int:
    _trace("invoked")
    if os.path.exists(os.path.join(VOICE_DIR, "off")):
        _trace("muted — exit")
        return 0

    raw = sys.stdin.read()
    if not raw.strip():
        _trace("EMPTY stdin — no payload delivered")
        return 0

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        _trace(f"bad JSON on stdin: {raw[:80]!r}")
        return 0

    transcript_path = payload.get("transcript_path", "")
    text = ""
    for attempt in range(5):
        text = last_assistant_text(transcript_path)
        if text:
            break
        time.sleep(0.3)

    if not text:
        _trace(f"no assistant text in {transcript_path!r} after retries")
        return 0

    text = clean_for_speech(text, MAX_CHARS)
    if not text:
        return 0

    stop_previous_say()
    speak(text)
    _trace(f"speaking {len(text)} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
