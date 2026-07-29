# Shell Function Reference

The TUI installer (`./install.sh`) can automatically append the most recent version of these functions to your shell profile (`~/.zshrc`, `~/.bashrc`, or PowerShell `$PROFILE`).

---

## Claude Code

| Command                | Description                                  |
| ---------------------- | -------------------------------------------- |
| `claude`               | Direct Claude Code (subscription login)      |
| `claude --proxy`       | Proxy mode via Nebius with session forwarder |
| `claude --proxy --bypass` | Proxy mode, non-interactive (defaults) for agent-to-agent orchestration |
| `claude --proxy <dir>` | Proxy mode starting in a specific directory  |
| `claudius`             | Alias for `claude --proxy`                   |

### Bypass mode (agent-to-agent)

`claude --proxy --bypass [extra args]` skips the interactive session/model/
ensemble prompts and launches immediately with defaults: session name
`Agent2Agent`, model = the `.env` `MODEL` default (no per-session override), and
ensemble **off**. `--bypass` must come **second**, right after `--proxy`; any
remaining args (e.g. `--dangerously-skip-permissions`, `-p "…"`) pass straight
through to Claude Code. Intended for orchestration where prompts would block.

### Session Forwarder

The installed bash/zsh function uses `scripts/session_forwarder.py` to spin up a temporary forwarder on a random free port for each proxy session. This gives the statusline independent per-session metrics. When Claude Code exits, the forwarder is cleaned up automatically.

The PowerShell function uses `Start-Job` for equivalent behaviour.

Forwarder output (including network errors) is written to `logs/session-forwarder.log` so it does not appear in the Claude Code TUI.

### Visual Feedback

- **Green** (`▐▛▜▌ Claude Direct`) = Subscription login mode
- **Purple** (`▐▛▜▌ Claude via Proxy`) = Proxy mode via Nebius

---

## Codex CLI

| Command         | Description                                             |
| --------------- | ------------------------------------------------------- |
| `codex`         | Direct Codex CLI (standard OpenAI auth)                 |
| `codex --proxy` | Proxy mode via Nebius (sets OPENAI_API_KEY from `.env`) |
| `codexius`      | Alias for `codex --proxy`                               |

### Environment Variables

The `codex --proxy` function reads your `OPENAI_API_KEY` from the environment or from `.env` in the proxy repo, then exports it before launching Codex. No additional manual steps are needed.

### Visual Feedback

- **Green** (`▐▛▜▌ Codex Direct`) = Standard OpenAI auth
- **Purple** (`▐▛▜▌ Codex via Proxy`) = Proxy mode via Nebius

---

## Troubleshooting

### Proxy not running?

```bash
cd /path/to/claude-codex-nebius-proxy
.venv/bin/python start_proxy.py
```

### Session-forwarder errors in the TUI?

If you see errors like `[forwarder] request forwarding failed` inside Claude Code, they are expected during normal operation (brief upstream disconnects). They are redirected to `logs/session-forwarder.log` by the installed shell function. If they appear in the TUI, re-run `./install.sh` to get the latest shell function.

### Port different from 8083?

Re-run `./install.sh` and enter your custom port on the API Key & Port step.

### Model name in the status line isn't clickable

The model name in the proxy status line is wrapped in an OSC 8 terminal hyperlink
that opens the per-session model picker (`/dashboard/pick?session=<name>`) in your
browser. Clicking requires a hyperlink-capable terminal (iTerm2, Kitty, WezTerm,
Ghostty). **Terminal.app does not support OSC 8**, and tmux/SSH may strip the
sequences — there the model still displays, it just isn't clickable.

If your terminal supports hyperlinks but Claude Code doesn't render them, set
`FORCE_HYPERLINK=1` before launching Claude Code:

```bash
export FORCE_HYPERLINK=1
```

You can also open the picker directly: `http://localhost:<port>/dashboard/pick?session=<your-session-name>`.

To enable clicking, you must re-run `./install.sh` after upgrading — it refreshes
the shell function (`NEBIUS_SESSION_NAME`) and the statusline command.

### Upgrading an existing install (shell functions)

When the installer sees a profile that already has the `claude`/`claudius`/`codex`/`codexius`
shortcuts, it marks that profile `(already configured)` and **skips** it by default — so a
re-run won't double-append. That also means an older function version won't refresh on its own.

To refresh an already-configured machine, on the final step of the wizard check
**"Reinstall (overwrite) existing shell functions"**. Each already-configured profile then
becomes a selectable checkbox (unchecked by default); check the ones you want to refresh,
and on **Apply & Finish** the installer backs up the rc file (to `<rc>.bak.<mtime>`) and
replaces the old Claude+Codex function block with the current version. Re-running with the
box checked and a profile selected is idempotent — it swaps the block in place, no duplication.

You can also do it by hand: remove the `# Claude Shell Function …` block through
`alias codexius='codex --proxy'` from each rc file, then run `./install.sh`.
