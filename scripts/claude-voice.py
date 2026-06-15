#!/usr/bin/env python3
"""claude-voice — hands-free voice conversation with Claude Code (macOS).

A Python wrapper that orchestrates voice interaction with Claude Code.
Uses the proxy's /voice/* endpoints (TTS/STT) when available,
falling back to macOS `say` and `hear` when not.

Modes:
  --handsfree [--proxy]   Opens tmux: Claude Code on top, voice HUD below.
  --loop -t pane          Runs inside the HUD pane (auto-listen, auto-send).
  ptt -t pane             Push-to-talk into existing Claude tmux pane.

Environment:
  CLAUDE_VOICE_LOCALE     Speech recognition locale (default: en-US)
  CLAUDE_VOICE_SILENCE    Seconds of silence to end utterance (default: 2)
  CLAUDE_VOICE_NAME      `say` voice name (default: system)
  CLAUDE_VOICE_RATE      `say` words per minute (default: 190)
  CLAUDE_VOICE_MAX_CHARS  Max chars to speak (default: 600)
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import tempfile
import urllib.request
import urllib.error
from pathlib import Path


# ── Config ──
VOICE_DIR = Path.home() / ".claude" / "voice"
HEAR = os.environ.get("HEAR_BINARY_PATH") or str(VOICE_DIR / "bin" / "hear")
SAY = os.environ.get("SAY_BINARY_PATH", "/usr/bin/say")
LOCALE = os.environ.get("CLAUDE_VOICE_LOCALE", "en-US")
SILENCE_TIMEOUT = int(os.environ.get("CLAUDE_VOICE_SILENCE", "2"))
TURN_TIMEOUT = int(os.environ.get("CLAUDE_VOICE_TURN_TIMEOUT", "900"))
VOICE_NAME = os.environ.get("CLAUDE_VOICE_NAME", "")
VOICE_RATE = int(os.environ.get("CLAUDE_VOICE_RATE", "190"))
MAX_CHARS = int(os.environ.get("CLAUDE_VOICE_MAX_CHARS", "600"))
SAY_PID_FILE = Path("/tmp/claude-voice-say.pid")
HOOK_LOG = Path("/tmp/claude-voice-hook.log")

# Proxy endpoint (for when running through proxy)
def _proxy_voice_url(path: str) -> str:
    proxy = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    return f"{proxy}/voice{path}"


def _voice_enabled() -> bool:
    """Check if proxy voice endpoints are available."""
    proxy = os.environ.get("ANTHROPIC_BASE_URL", "")
    if not proxy:
        return False
    try:
        req = urllib.request.Request(f"{proxy.rstrip('/')}/voice/health", method="GET")
        req.add_header("Authorization", f"Bearer {os.environ.get('ANTHROPIC_AUTH_TOKEN', 'claude-local')}")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def _say(text: str) -> None:
    """Speak text. Uses proxy TTS if available, else macOS say."""
    # Kill previous say
    try:
        with open(SAY_PID_FILE) as f:
            os.kill(int(f.read().strip()), signal.SIGTERM)
    except (OSError, ValueError):
        pass

    if _voice_enabled():
        try:
            data = json.dumps({
                "input": text,
                "voice": VOICE_NAME or "af_river",
                "response_format": "mp3",
            }).encode()
            req = urllib.request.Request(
                _proxy_voice_url("/tts"),
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.environ.get('ANTHROPIC_AUTH_TOKEN', 'claude-local')}",
                },
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                audio = resp.read()
                # Play audio via afplay
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
                return
        except Exception as e:
            print(f"⚠ proxy TTS failed ({e}), falling back to say", flush=True)

    # Fallback: macOS say
    cmd = [SAY, "-r", str(VOICE_RATE)]
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


def _say_running() -> bool:
    try:
        with open(SAY_PID_FILE) as f:
            pid = int(f.read().strip())
        return os.kill(pid, 0) is None
    except (OSError, ValueError):
        return False


def _wait_say_done() -> None:
    while _say_running():
        time.sleep(0.4)


def _listen_once() -> str:
    """Record and transcribe one utterance.

    Returns transcribed text string. Empty string if nothing heard.
    """
    # Check if proxy STT is available
    if _voice_enabled():
        # Record via sox (if available) or sox
        try:
            # Use sox to record temporary WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            # Record with sox -d (default device)
            record_cmd = ["sox", "-d", "-r", "16000", "-c", "1", "-b", "16", tmp_path]
            proc = subprocess.Popen(record_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            # Wait for silence or max duration
            time.sleep(1.5 + SILENCE_TIMEOUT)
            proc.terminate()
            proc.wait()

            # Upload to proxy STT
            with open(tmp_path, "rb") as f:
                audio = f.read()
            boundary = "----form-boundary-claude-voice"
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\";"
                f' filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
            ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()

            req = urllib.request.Request(
                _proxy_voice_url("/stt"),
                data=body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "Authorization": f"Bearer {os.environ.get('ANTHROPIC_AUTH_TOKEN', 'claude-local')}",
                },
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode())
                return data.get("text", "").strip()
        except FileNotFoundError:
            print("⚠ sox not found for recording, falling back to hear", flush=True)
        except Exception as e:
            print(f"⚠ proxy STT failed ({e}), falling back to hear", flush=True)
        finally:
            try:
                os.unlink(tmp_name)
            except (NameError, OSError):
                pass

    # Fallback: hear binary (Apple on-device STT)
    hear_path = HEAR if os.path.isfile(HEAR) else None
    if hear_path is None:
        hear_path = subprocess.run(
            ["sh", "-c", "command -v hear"],
            capture_output=True, text=True
        ).stdout.strip()

    if not hear_path or not os.path.isfile(hear_path):
        print("error: No STT provider available (hear binary or sox needed)", file=sys.stderr)
        return ""

    result = subprocess.run(
        [hear_path, "-d", "-p", "-t", str(SILENCE_TIMEOUT), "-l", LOCALE],
        capture_output=True, text=True,
    )
    lines = result.stdout.strip().split("\n")
    return lines[-1].strip() if lines else ""


def _pane_busy(target: str) -> bool:
    result = subprocess.run(
        ["tmux", "capture-pane", "-t", target, "-p"],
        capture_output=True, text=True,
    )
    return "esc to interrupt" in result.stdout.lower()


def clean_for_speech(text: str) -> str:
    """Strip markdown/code/URLs for spoken output."""
    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"https?://\S+", "link", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_>|#]+", " ", text)
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_CHARS:
        cut = text[:MAX_CHARS]
        m = re.search(r"^.*[.!?]", cut, flags=re.DOTALL)
        text = (m.group(0) if m and len(m.group(0)) > MAX_CHARS * 0.4 else cut) + " ... more in the terminal."
    return text


# ── Push-to-talk ──
def run_ptt(target: str) -> None:
    if not target:
        result = subprocess.run(
            ["tmux", "list-panes", "-a", "-F", "#{pane_id} #{pane_current_command}"],
            capture_output=True, text=True,
        )
        for line in result.stdout.split("\n"):
            parts = line.strip().split(" ")
            if len(parts) >= 2 and parts[1] in ("claude", "node"):
                target = parts[0]
                break
        if not target:
            print("no claude pane found; use -t session:win.pane or --handsfree", file=sys.stderr)
            sys.exit(1)

    print(f"push-to-talk → {target} · Enter to talk · q to quit")
    while True:
        line = input("\n🎤 Enter to talk: ")
        if line == "q":
            return
        print("listening...")
        text = _listen_once()
        if not text:
            print("(heard nothing)")
            continue
        print(f"→ {text}")
        subprocess.run(["tmux", "send-keys", "-t", target, "-l", text])
        subprocess.run(["tmux", "send-keys", "-t", target, "Enter"])


# ── Hands-free loop ──
def run_handsfree(proxy: bool) -> None:
    import socket
    session = f"voicechat-{os.getpid()}"
    claude_cmd = "claude"
    if proxy:
        claude_cmd = "claude --proxy"

    # Start session with Claude Code
    subprocess.run([
        "tmux", "new-session", "-d", "-s", session,
        "-x", "220", "-y", "55", "-c", os.getcwd(),
        f"zsh -ic '{claude_cmd}'",
    ])
    # Split window for HUD
    script_path = Path(__file__).resolve()
    subprocess.run([
        "tmux", "split-window", "-v", "-l", "9", "-t", session,
        f"CLAUDE_VOICE_LOCALE='{LOCALE}' python3 '{script_path}' --loop -t '{session}:0.0'",
    ])
    subprocess.run(["tmux", "select-pane", "-t", f"{session}:0.0"])
    subprocess.run(["tmux", "attach", "-t", session])


def run_loop(target: str) -> None:
    """Runs inside the tmux HUD pane — auto-listen client."""
    if not target:
        print("--loop requires -t", file=sys.stderr)
        sys.exit(2)

    print(f"🟢 hands-free voice HUD → {target}")
    print('   speak after the 🎤 · say "goodbye claude" to stop · Ctrl-C to abort')

    # Unmute if muted
    mute_file = VOICE_DIR / "off"
    if mute_file.exists():
        mute_file.unlink()
        print("   (unmuted spoken replies)")

    # Wait for Claude Code to boot
    print("   waiting for Claude Code to boot", end="", flush=True)
    for _ in range(60):
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", target, "-p"],
            capture_output=True, text=True,
        )
        if "❯" in result.stdout:
            break
        print(".", end="", flush=True)
        time.sleep(1)
    print()

    with open(HOOK_LOG, "a") as log:
        while True:
            _wait_say_done()
            time.sleep(0.5)
            print("🎤 listening...", flush=True)
            text = _listen_once()
            if not text:
                continue

            lower_text = text.lower()
            if any(x in lower_text for x in ["goodbye claude", "exit voice mode", "stop listening"]):
                subprocess.run(["/usr/bin/say", "-r", "200", "goodbye"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("👋 voice loop ended — claude keeps running (Ctrl-b d to detach)")
                sys.exit(0)

            print(f"→ {text}", flush=True)
            mark = sum(1 for _ in open(HOOK_LOG)) if HOOK_LOG.exists() else 0
            subprocess.run(["tmux", "send-keys", "-t", target, "-l", text])
            subprocess.run(["tmux", "send-keys", "-t", target, "Enter"])

            print("⏳ Claude is working", end="", flush=True)
            waited = 0
            while waited < TURN_TIMEOUT:
                now_lines = sum(1 for _ in open(HOOK_LOG)) if HOOK_LOG.exists() else 0
                if now_lines > mark:
                    break
                if waited >= 5 and not _pane_busy(target):
                    break
                print(".", end="", flush=True)
                time.sleep(1)
                waited += 1
            print()
            print("🔊 speaking reply...", flush=True)
            time.sleep(1.5)
            _wait_say_done()


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice conversation with Claude Code")
    parser.add_argument("--handsfree", "-H", action="store_true", help="Hands-free mode")
    parser.add_argument("--loop", action="store_true", help="HUD loop (internal)")
    parser.add_argument("--proxy", action="store_true", help="Use proxy mode")
    parser.add_argument("-t", dest="target", help="Target tmux pane")
    args = parser.parse_args()

    if args.handsfree:
        run_handsfree(args.proxy)
        return 0
    elif args.loop:
        run_loop(args.target or "")
        return 0
    else:
        run_ptt(args.target or "")
        return 0


if __name__ == "__main__":
    sys.exit(main())
