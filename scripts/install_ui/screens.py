"""Textual screens for the claude-codex-nebius-proxy installer wizard."""

from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import socket
import subprocess
import sys
import time

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Input,
    Markdown,
    Select,
    ProgressBar,
    RichLog,
    Static,
)

from .utils import (
    InstallState,
    ShellType,
    append_shell_function,
    detect_shell,
    fetch_nebius_models,
    get_all_shell_profiles,
    get_claude_settings_path,
    get_codex_config_path,
    get_repo_root,
    pick_default_models,
    safe_merge_settings,
    shell_function_is_present,
    write_codex_config,
    sync_pip_install,
    sync_venv_create,
    write_env,
)

# ─── helpers ─────────────────────────────────────────────────

def _check_prereqs() -> dict[str, tuple[bool, str]]:
    """Returns{"python3": (ok, version), "pip": (ok, msg), "curl": (ok, ""}"""
    results: dict[str, tuple[bool, str]] = {}

    try:
        version = subprocess.run(
            ["python3", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()
        ok = subprocess.run(
            ["python3", "-c", "import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)"],
            capture_output=True, check=False,
        ).returncode == 0
        results["python3"] = (ok, version)
    except Exception:
        results["python3"] = (False, "")

    try:
        p = subprocess.run(["python3", "-m", "pip", "--version"], capture_output=True, check=False)
        results["pip"] = (p.returncode == 0, "")
    except Exception:
        results["pip"] = (False, "")

    try:
        p = subprocess.run(["curl", "--version"], capture_output=True, check=False)
        results["curl"] = (p.returncode == 0, "")
    except Exception:
        results["curl"] = (False, "")

    return results


# ─── modal: overwrite confirmation ─────────────────────────────

class OverwriteStatuslineModal(ModalScreen[bool]):
    """Ask user whether to overwrite an existing statusLine config."""

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("A different statusLine is already configured.", classes="modal_title")
            yield Static("Overwrite it with the proxy statusline?", classes="modal_text")
            with Horizontal(classes="button_bar"):
                yield Button("Overwrite", variant="error", id="overwrite")
                yield Button("Keep existing", variant="primary", id="keep")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "overwrite")


# ─── shared base ─────────────────────────────────────────────

class WizardStep(Container):
    """Base layout: header + content + nav bar."""

    step_num: int
    step_title: str
    nav_back_target: str | None = None
    nav_next_target: str | None = None
    nav_next_label: str = "Continue  →"
    content_id: str = "content"

    def compose(self) -> ComposeResult:
        yield Static(f"[{self.step_num} / 8]  {self.step_title}", classes="step_header")
        with Container(classes="content_panel", id=self.content_id):
            yield None        # subclasses override compose_content
        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back" if self.nav_back_target else "back", disabled=not self.nav_back_target)
            yield Button(self.nav_next_label, variant="success", id="next" if self.nav_next_target else "next", disabled=not self.nav_next_target)

    def build_nav(self) -> ComposeResult:
        with Horizontal(classes="button_bar"):
            if self.nav_back_target:
                yield Button("←  Back", id="back")
            else:
                yield Button("←  Back", id="back", disabled=True)
            if self.nav_next_target:
                next_label = getattr(self, 'nav_next_label', "Continue  →")
                yield Button(next_label, variant="success", id="next")
            else:
                yield Button("Continue  →", variant="success", id="next", disabled=True)


# ─── 1. Welcome ────────────────────────────────────────────────

class WelcomeScreen(Screen):

    def compose(self) -> ComposeResult:
        with Container(classes="content_panel", id="welcome_content"):
            yield Static(
                "\n"
                "    ▐▛▜▌  Claude & Codex Proxy\n"
                "    ───────────────────────────\n",
                classes="banner",
            )
            yield Static("powered by Nebius Token Factory", classes="subtitle")
            yield Static("", classes="info_label")
            yield Markdown(
                ""
                "This wizard will:\n"
                "-  Check your system\n"
                "-  Set up a Python environment\n"
                "-  Connect to your Nebius API\n"
                "-  Configure your clients in one step\n",
            )
            yield Static("\nReady to set up your proxy?", classes="info_label")
        with Horizontal(classes="button_bar"):
            yield Button("Start Setup  ▶", variant="success", id="start")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.push_screen(PrerequisiteScreen())


# ─── 2. Prerequisites ──────────────────────────────────────────

class PrerequisiteScreen(Screen):

    results: reactive[dict[str, tuple[bool, str]]] = reactive({}, always_update=True)

    def compose(self) -> ComposeResult:
        yield Static("[1 / 8]  Checking Prerequisites", classes="step_header")
        with Container(classes="content_panel", id="prereq_content"):
            yield Static("Validating your system …", id="status")
            yield Static("", id="output")
            yield Static("", id="hint")
        with Horizontal(classes="button_bar"):
            yield Button("←  Back", disabled=True, id="back")
            yield Button("Continue  →", variant="success", id="next", disabled=True)
            yield Button("Recheck", variant="primary", id="recheck")

    def on_mount(self) -> None:
        self._run_check()

    def _run_check(self) -> None:
        self.results = _check_prereqs()

    def watch_results(self) -> None:
        out = self.query_one("#output", Static)
        status = self.query_one("#status", Static)
        hint = self.query_one("#hint", Static)
        lines: list[str] = []
        all_ok = True
        for name, (ok, info) in self.results.items():
            icon = "[green]✔[/]" if ok else "[red]✘[/]"
            extra = f"  ({info})" if info and ok else ""
            lines.append(f"  {icon}  {name}{extra}")
            if not ok:
                all_ok = False
        out.update("\n".join(lines))
        if all_ok:
            status.update("[green]✔  All set! You're good to go.[/]")
            hint.update("[dim]Press Continue to proceed.[/]")
            self.query_one("#next", Button).disabled = False
        else:
            status.update("[red]✘  Some checks failed.[/]")
            hint.update("[dim]Fix the issues above, then click Recheck.[/]")
            self.query_one("#next", Button).disabled = True

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            self.app.state.python_version = self.results.get("python3", (True, ""))[1]
            self.app.state.has_pip = self.results.get("pip", (False, ""))[0]
            self.app.state.has_curl = self.results.get("curl", (False, ""))[0]
            self.app.push_screen(VenvScreen())
        elif event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "recheck":
            self._run_check()


# ─── 3. Venv ───────────────────────────────────────────────────

class VenvScreen(Screen):

    _done: bool = False

    def compose(self) -> ComposeResult:
        yield Static("[2 / 8]  Virtual Environment", classes="step_header")
        with Container(classes="content_panel", id="venv_content"):
            yield Static("Setting up Python environment …", id="status")
            yield ProgressBar(total=100, id="progress")
            yield Static("", id="hint")
        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back")
            yield Button("Continue  →", variant="success", id="next", disabled=True)

    def on_mount(self) -> None:
        self.set_timer(0.3, self._create_venv)

    def _create_venv(self) -> None:
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)
        progress.update(progress=10)
        ok, msg = sync_venv_create()
        if ok:
            status.update(f"[green]✔  {msg}[/]")
            self._done = True
            self.query_one("#next", Button).disabled = False
        else:
            status.update(f"[red]✘  {msg}[/]")
            self.query_one("#hint", Static).update(
                "[dim]Fix: python3 -m ensurepip --upgrade[/dim]"
            )
        progress.update(progress=100 if ok else 80)

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next" and self._done:
            self.app.push_screen(DepsScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


# ─── 4. Dependencies ─────────────────────────────────────────

class DepsScreen(Screen):

    _done: bool = False

    def compose(self) -> ComposeResult:
        yield Static("[3 / 8]  Installing Dependencies", classes="step_header")
        with Container(classes="content_panel", id="deps_content"):
            yield Static("Installing packages …", id="status")
            yield ProgressBar(total=100, id="progress")
            yield RichLog(id="log", wrap=True)
        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back")
            yield Button("Continue  →", variant="success", id="next", disabled=True)

    def on_mount(self) -> None:
        self.set_timer(0.3, self._install)

    def _install(self) -> None:
        progress = self.query_one("#progress", ProgressBar)
        log = self.query_one("#log", RichLog)
        status = self.query_one("#status", Static)
        progress.update(progress=20)
        log.write("Upgrading pip …")
        ok, msg = sync_pip_install()
        progress.update(progress=100)
        if ok:
            self._done = True
            status.update("[green]✔  Dependencies installed[/]")
            log.write("[green]All set![/green]")
            self.query_one("#next", Button).disabled = False
        else:
            status.update(f"[red]✘  {msg}[/]")
            log.write(f"[red]{msg}[/red]")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next" and self._done:
            self.app.push_screen(ApiKeyScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


# ─── 5. API Key & Port ───────────────────────────────────────

class ApiKeyScreen(Screen):
    """Step 4: collect API key + port, auto-load from .env, test key live."""

    def _load_existing_env(self) -> tuple[str, str]:
        """Read .env (if present) and return (api_key, port)."""
        env_path = get_repo_root() / ".env"
        key = ""
        port = "8083"
        if not env_path.is_file():
            return key, port
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY="):
                val = line.split("=", 1)[1]
                key = val.strip("'\"")
            elif line.startswith("PORT="):
                port = line.split("=", 1)[1].strip()
        return key, port

    def on_mount(self) -> None:
        env_key, env_port = self._load_existing_env()
        if env_key:
            self.query_one("#api_key", Input).value = env_key
            self.query_one("#status", Static).update(
                "[rgb(0,188,212)]Found existing key in .env[/]"
            )
        if env_port:
            self.query_one("#port", Input).value = env_port

    def compose(self) -> ComposeResult:
        yield Static("[4 / 8]  API Key & Port", classes="step_header")
        with Container(classes="content_panel", id="api_content"):
            yield Static("Nebius API Key:", classes="form_label")
            yield Input(
                placeholder="Paste your key here — it will be masked",
                password=True,
                id="api_key",
            )
            yield Static("", id="status")   # green/red below key field
            yield Static("Proxy port (default 8083):", classes="form_label")
            yield Input(value="8083", id="port")
            yield Static(
                "[dim]Get your API key from https://nebius.com  "
                "The proxy will listen on this port on your machine.[/]",
                classes="hint_label",
            )
            # Test-result line (separate from error so both can coexist)
            yield Static("", id="test_result")
            yield Static("", id="error")
        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back")
            yield Button("Test Key", variant="primary", id="test_key")
            yield Button("Continue  →", variant="success", id="next")

    # ── helper to set disabled on a button (called from main thread) ──
    def _set_btn_disabled(self, btn_id: str, disabled: bool) -> None:
        self.query_one(f"#{btn_id}", Button).disabled = disabled

    def _test_key(self) -> None:
        """Callback run by a background thread."""
        key = self.query_one("#api_key", Input).value.strip()
        if not key:
            self.app.call_from_thread(
                self.query_one("#test_result", Static).update,
                "[red]⚠  Cannot test – key field is empty[/]",
            )
            self.app.call_from_thread(self._set_btn_disabled, "test_key", False)
            return

        self.app.call_from_thread(
            self.query_one("#test_result", Static).update,
            "[rgb(136,136,136)]Testing …  (this may take a few seconds)[/]",
        )
        result = fetch_nebius_models(key, self.app.state.base_url)
        if result["ok"]:
            msg = f"[rgb(76,175,80)]✔  Key accepted – {len(result.get('models', []))} models reachable[/]"
        else:
            msg = f"[red]✘  Key failed – {result.get('error', 'unknown error')}[/]"
        self.app.call_from_thread(self.query_one("#test_result", Static).update, msg)
        self.app.call_from_thread(self._set_btn_disabled, "test_key", False)

    def _try_continue(self) -> None:
        """Validate inputs and advance to model selection."""
        key = self.query_one("#api_key", Input).value.strip()
        port_str = self.query_one("#port", Input).value.strip()
        error = self.query_one("#error", Static)
        if not key:
            error.update("[red]Please enter your API key.[/]")
            return
        try:
            port = int(port_str)
            if not (1024 <= port <= 65535):
                raise ValueError
        except ValueError:
            error.update("[red]Port must be a number from 1024 to 65535.[/]")
            return
        error.update("")
        self.app.state.api_key = key
        self.app.state.port = port
        self.app.push_screen(ModelScreen())

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            self._try_continue()
        elif event.button.id == "test_key":
            self.query_one("#test_key", Button).disabled = True
            self.query_one("#test_result", Static).update(
                "[rgb(136,136,136)]Testing …  (this may take a few seconds)[/]",
            )
            import threading
            threading.Thread(target=self._test_key, daemon=True).start()
        elif event.button.id == "back":
            self.app.pop_screen()

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("api_key", "port"):
            self._try_continue()


# ─── 6. Model Selection ────────────────────────────────────────

class ModelScreen(Screen):

    _defaults: dict[str, str] = {}
    _env_defaults: dict[str, str] = {}

    def _load_env_models(self) -> dict[str, str]:
        """Read existing BIG_MODEL etc. from .env if present."""
        env_path = get_repo_root() / ".env"
        d: dict[str, str] = {}
        if not env_path.is_file():
            return d
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            for key in ("BIG_MODEL", "MIDDLE_MODEL", "SMALL_MODEL", "VISION_MODEL"):
                if line.startswith(f"{key}="):
                    val = line.split("=", 1)[1].strip("'\"")
                    if val:
                        d[key] = val
        return d

    def _pick_from_env(self, env_models: dict[str, str], available: list[str]) -> dict[str, str]:
        """Prefer .env values when they exist and are in available list."""
        defaults = pick_default_models(available)
        for key in ("BIG_MODEL", "MIDDLE_MODEL", "SMALL_MODEL", "VISION_MODEL"):
            env_val = env_models.get(key)
            if env_val and env_val in available:
                defaults[key] = env_val
        return defaults

    def compose(self) -> ComposeResult:
        yield Static("[5 / 8]  Model Selection", classes="step_header")
        with Container(classes="content_panel", id="model_content"):
            yield Static("Fetching available models from Nebius …", id="status")
            yield ProgressBar(total=100, id="progress")
            yield Static("", id="model_info")
            yield Static("", id="error")
        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back")
            yield Button("Auto-Fill from .env", variant="primary", id="defaults", disabled=True)
            yield Button("Continue  →", variant="success", id="next", disabled=True)

    def on_mount(self) -> None:
        self._env_defaults = self._load_env_models()
        self.set_timer(0.3, self._fetch_models)

    def _fetch_models(self) -> None:
        progress = self.query_one("#progress", ProgressBar)
        status = self.query_one("#status", Static)
        info = self.query_one("#model_info", Static)
        progress.update(progress=30)
        result = fetch_nebius_models(self.app.state.api_key, self.app.state.base_url)
        progress.update(progress=100)

        if not result["ok"]:
            status.update("[red]✘  Could not fetch models[/]")
            self.query_one("#error", Static).update(f"[red]{result['error']}[/]")
            self.query_one("#next", Button).disabled = False
            return

        models = result["models"]
        self.app.state.available_models = models
        self.app.state.models_fetched = True
        self._defaults = self._pick_from_env(self._env_defaults, models)
        status.update(f"[green]✔  Found {len(models)} models[/]")
        if self._env_defaults:
            used = [v for k, v in self._env_defaults.items() if v in models]
            info.update(f"[rgb(0,188,212)]Loaded {len(used)} model(s) from existing .env[/]")
        else:
            info.update(f"[dim]Using smart defaults: {self._defaults['BIG_MODEL']}[/dim]")
        self.query_one("#defaults", Button).disabled = False
        self.query_one("#next", Button).disabled = False

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in ("next", "defaults"):
            self.app.state.big_model = self._defaults["BIG_MODEL"]
            self.app.state.middle_model = self._defaults["MIDDLE_MODEL"]
            self.app.state.small_model = self._defaults["SMALL_MODEL"]
            self.app.state.vision_model = self._defaults["VISION_MODEL"]
            self.app.push_screen(ReviewScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


# ─── 7. Review ─────────────────────────────────────────────────

class ReviewScreen(Screen):

    _initial_loaded: bool = False

    _LABELS: dict[str, str] = {
        "big_model": "Big Model",
        "middle_model": "Middle Model",
        "small_model": "Small Model",
        "vision_model": "Vision Model",
    }

    def _populate_selects(self) -> None:
        """Fill dropdowns after mount (set_options needs DOM)."""
        models = self.app.state.available_models
        if not models:
            return
        defaults: dict[str, str] = {
            "big_model": self.app.state.big_model,
            "middle_model": self.app.state.middle_model,
            "small_model": self.app.state.small_model,
            "vision_model": self.app.state.vision_model,
        }
        opts = [(m, m) for m in models]
        for var, default in defaults.items():
            sel = self.query_one(f"#{var}", Select)
            sel.set_options(opts)
            sel.value = default if default in models else models[0]

    def compose(self) -> ComposeResult:
        yield Static("[6 / 8]  Review Your Choices", classes="step_header")
        with Container(classes="content_panel", id="review_content"):
            yield Static(
                "Pick your models from the dropdowns below. "
                "When everything looks right, hit Continue.",
                classes="hint_label",
            )
            yield Static("", id="model_info")

            for var, label in self._LABELS.items():
                yield Static(f"{label}:", classes="form_label")
                yield Select([("Loading models …", "")], id=var, allow_blank=False)

        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back")
            yield Button("Continue  →", variant="success", id="confirm")

    def on_mount(self) -> None:
        if not self._initial_loaded:
            if self.app.state.available_models:
                self._populate_selects()
                self._initial_loaded = True
            else:
                self.query_one("#model_info", Static).update(
                    "[yellow]⚠  No models loaded — go back and test your API key.[/]"
                )

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.app.state.big_model = self.query_one("#big_model", Select).value
            self.app.state.middle_model = self.query_one("#middle_model", Select).value
            self.app.state.small_model = self.query_one("#small_model", Select).value
            self.app.state.vision_model = self.query_one("#vision_model", Select).value
            write_env(self.app.state)
            self.app.push_screen(SmokeScreen())
        elif event.button.id == "back":
            self.app.pop_screen()


# ─── 8. Smoke Test ────────────────────────────────────────────

class SmokeScreen(Screen):

    _passed: bool = False

    def compose(self) -> ComposeResult:
        yield Static("[7 / 8]  Smoke Test", classes="step_header")
        with Container(classes="content_panel", id="smoke_content"):
            yield Static(
                "Start the proxy in another terminal, then press Test.",
                id="status",
            )
            yield RichLog(id="log", wrap=True, markup=True)
        with Horizontal(classes="button_bar", id="choice_bar"):
            yield Button("←  Back", id="back")
        with Horizontal(classes="button_bar", id="action_bar"):
            yield Button("🧪  Test: curl /health", id="btn_test")
            yield Button("Continue  →", variant="success", id="next", disabled=True)

    def on_mount(self) -> None:
        self._show_commands()

    def _show_commands(self) -> None:
        log = self.query_one("#log", RichLog)
        s = self.app.state
        log.write("[bold]Start the proxy in another terminal:[/bold]")
        log.write("")
        log.write("Run the command appropriate for your setup.")
        log.write(f"See:  [rgb(0,188,212)]docs/start-proxy.md[/rgb(0,188,212)]  for full instructions.")
        log.write("")
        log.write(f"Dashboard: http://localhost:{s.port}/dashboard")

    def _test_proxy(self) -> None:
        import urllib.request

        status = self.query_one("#status", Static)
        log = self.query_one("#log", RichLog)
        s = self.app.state

        log.write("")
        log.write(f"Checking http://127.0.0.1:{s.port}/health …")

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{s.port}/health")
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            status.update(f"[red]✘  Proxy not responding on port {s.port}[/]")
            log.write("[red]Proxy not running.[/red]")
            log.write("Steps:")
            log.write("  1. Start the proxy (see docs/START-PROXY.md)")
            log.write("  2. Wait for 'Uvicorn running' message")
            log.write("  3. Press 'Test: curl /health' again")
            return

        log.write("[green]Health check passed[/green]")

        # Test connection
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{s.port}/test-connection")
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read().decode())
            status_val = data.get("status", "unknown")
        except Exception as e:
            status_val = f"error: {e}"

        if status_val == "success":
            self._passed = True
            s.smoke_test_passed = True
            status.update("[green]✔  All tests passed[/]")
            log.write("[green]Test connection succeeded.[/green]")
            log.write(f"Open http://localhost:{s.port}/dashboard")
            self.query_one("#next", Button).disabled = False
        else:
            status.update(f"[red]✘  Test failed: {status_val}[/]")
            log.write(f"[red]Test connection failed: {status_val}[/red]")
            log.write("Check your API key and Nebius connectivity.")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        eid = event.button.id or ""
        if eid == "next" and self._passed:
            self.app.push_screen(ConfigurationScreen())
        elif eid == "back":
            self.app.pop_screen()
        elif eid == "btn_test":
            log = self.query_one("#log", RichLog)
            log.clear()
            self._show_commands()
            self._test_proxy()


# ─── 8. Configuration (Consolidated) ───────────────────────────

class ConfigurationScreen(Screen):
    """Single-screen config: Claude ✓ Codex ✓ Shell shortcuts ✓"""

    STATUSLINE_CMD = (
        '[ -z "$ANTHROPIC_BASE_URL" ] && exit 0; '
        'base="${ANTHROPIC_BASE_URL%/}"; '
        'cfg=$(curl -fsS --max-time 1 "$base/api/observability/config" 2>/dev/null || true); '
        'model=$(printf \'%s\' "$cfg" | python3 -c \'import json,sys; d=json.load(sys.stdin); '
        'print((d.get("configured_models") or {}).get("big") or "")\' 2>/dev/null || true); '
        'ctx=$(curl -fsS --max-time 1 "$base/api/observability/context-usage" 2>/dev/null || true); '
        'free=$(printf \'%s\' "$ctx" | python3 -c \'import json,sys; d=json.load(sys.stdin); '
        'r=d.get("remaining_tokens",1048576) or 1048576; '
        't=d.get("context_limit",1048576) or 1048576; '
        'print(f"{round((r/t)*100)}")\' 2>/dev/null || true); '
        'if [ -n "$model" ]; then '
        'if [ -n "$free" ] && [[ "$free" =~ ^[0-9]+$ ]]; then '
        'if [ "$free" -le 20 ]; then c="\\\\033[31m"; '
        'elif [ "$free" -le 40 ]; then c="\\\\033[38;5;208m"; '
        'elif [ "$free" -le 50 ]; then c="\\\\033[33m"; '
        'else c="\\\\033[32m"; fi; '
        'e="\\\\033[0m"; '
        'echo "[nebius://$model $c${free}% free$e] $base/dashboard"; '
        'else echo "[nebius://$model] $base/dashboard"; fi; '
        'else echo "[proxy://$base]"; fi'
    )

    def compose(self) -> ComposeResult:
        yield Static("[8 / 8]  Configure Your Setup", classes="step_header")
        with Container(classes="content_panel", id="config_content"):
            yield Static(
                "Choose what to configure.  All options checked by default.",
                classes="hint_label",
            )
            yield Static("", id="config_status")

            # ── Claude ──
            yield Static("[bold rgb(0,188,212)]▐▛▜▌ Claude Code[/]", classes="section_header")
            yield Checkbox(
                "Configure Claude Code statusline (shows model + token usage in status bar)",
                value=True,
                id="do_claude",
            )
            yield Static("  [dim]Writes ~/.claude/settings.json[/]", classes="hint_label", id="claude_hint")

            # ── Codex ──
            yield Static("[bold rgb(0,188,212)]▐▛▜▌ Codex CLI[/]", classes="section_header")
            yield Checkbox(
                "Configure Codex CLI proxy routing (~/.codex/config.toml)",
                value=True,
                id="do_codex",
            )
            yield Static("", classes="hint_label", id="codex_hint")

            # ── Shell Shortcuts / Profiles ──
            yield Static("[bold rgb(0,188,212)]▐▛▜▌ Shell Shortcuts[/]", classes="section_header")
            yield Static("Select profiles to add shortcuts:", classes="hint_label", id="profile_label")
            yield Static("", id="profile_status")

        with Horizontal(classes="button_bar"):
            yield Button("←  Back", id="back")
            yield Button("Apply & Finish  →", variant="success", id="next")

    def on_mount(self) -> None:
        s = self.app.state

        # ── Claude hint ──
        claude_hint = self.query_one("#claude_hint", Static)
        settings_path = get_claude_settings_path()
        if settings_path.exists():
            try:
                existing = json.loads(settings_path.read_text(encoding="utf-8"))
                if "statusLine" in existing:
                    self.app.state.statusline_exists = True
                    claude_hint.update(f"  [dim]Note: ~/.claude/settings.json exists[/] — will prompt if different")
            except Exception:
                pass

        # ── Codex hint ──
        codex_hint = self.query_one("#codex_hint", Static)
        codex_path = get_codex_config_path()
        if codex_path and codex_path.exists():
            model = s.big_model
            if model:
                provider_model = f"nebius/{model}" if not model.startswith("nebius/") else model
                codex_hint.update(
                    f"  [dim]Found existing config — will update model={provider_model}[/]"
                )
        else:
            codex_hint.update("  [dim]No existing config — will create ~/.codex/config.toml[/]")

        # ── Shell profiles ──
        from .utils import get_all_shell_profiles

        profile_container = self.query_one("#config_content", Container)
        profile_label = self.query_one("#profile_label", Static)
        profile_status = self.query_one("#profile_status", Static)

        s.available_profiles = get_all_shell_profiles()

        if not s.available_profiles:
            profile_label.update("[yellow]⚠ No shell profiles detected — install bash, zsh, or PowerShell[/]")
            profile_status.update("[dim]Skipping profile configuration.[/dim]")
        else:
            # Clear the old hint and add checkboxes for each profile
            profile_label.update("Select profiles to add shortcuts:")
            selected: list[tuple[ShellType, str]] = []

            for idx, (stype, path) in enumerate(s.available_profiles):
                # Check if functions already exist in this profile
                already_has = shell_function_is_present(stype, path)
                checkbox_id = f"profile_{idx}"

                if already_has:
                    # Show as already configured, unchecked and disabled
                    profile_container.mount(Checkbox(
                        f"[dim]{path} (already configured)[/dim]",
                        value=False,
                        disabled=True,
                        id=checkbox_id,
                    ))
                else:
                    # Checked by default, user can uncheck
                    selected.append((stype, path))
                    profile_container.mount(Checkbox(
                        path,
                        value=True,
                        id=checkbox_id,
                    ))

            s.selected_profiles = selected
            profile_status.update(f"[dim]Will add to {len(selected)} profile(s)[/dim]")

    @on(Checkbox.Changed)
    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Update state when checkboxes toggle."""
        cb = event.checkbox
        if cb.id == "do_claude":
            self.app.state.configure_claude = cb.value
        elif cb.id == "do_codex":
            self.app.state.configure_codex = cb.value
        elif cb.id and cb.id.startswith("profile_"):
            # Recompute selected profiles from all profile_* checkboxes
            s = self.app.state
            selected: list[tuple[ShellType, str]] = []
            for idx, (stype, path) in enumerate(s.available_profiles):
                try:
                    cb_widget = self.query_one(f"#profile_{idx}", Checkbox)
                    if cb_widget and cb_widget.value:
                        selected.append((stype, path))
                except Exception:
                    continue
            s.selected_profiles = selected
            profile_status = self.query_one("#profile_status", Static)
            count = len(selected)
            total = len(s.available_profiles)
            if count == 0:
                profile_status.update("[dim]No profiles selected[/dim]")
            else:
                profile_status.update(f"[dim]Will add to {count}/{total} profile(s)[/dim]")

    def _write_codex(self) -> str:
        result = write_codex_config(
            self.app.state.big_model,
            self.app.state.port,
            get_repo_root(),
        )
        if result.get("action") == "written":
            return f"[green]✔  {result.get('message', 'Done')}[/]"
        else:
            return f"[red]✘  {result.get('message', 'Unknown error')}[/]"

    def _write_statusline(self) -> tuple[str, bool]:
        """Return (msg, blocked_on_modal)."""
        settings_path = get_claude_settings_path()
        if settings_path.exists():
            try:
                existing = json.loads(settings_path.read_text(encoding="utf-8"))
                sl = existing.get("statusLine")
                if isinstance(sl, dict) and sl.get("command", "").strip() == self.STATUSLINE_CMD.strip():
                    return ("[green]✔  statusLine already configured correctly[/]", False)
                if "statusLine" in existing:
                    def handle(overwrite: bool) -> None:
                        status = self.query_one("#config_status", Static)
                        if overwrite:
                            result = safe_merge_settings(self.STATUSLINE_CMD, get_repo_root())
                            status.update(f"[green]✔  {result.get('message', '')}[/]")
                            self.app.state.configure_statusline = True
                        else:
                            status.update("[dim]Kept existing statusLine — skipped.[/]")
                            self.app.state.configure_statusline = False
                        self._finish()
                    self.app.push_screen(OverwriteStatuslineModal(), callback=handle)
                    return ("", True)
            except Exception:
                pass
        result = safe_merge_settings(self.STATUSLINE_CMD, get_repo_root())
        action = result.get("action", "")
        if action in ("created", "added"):
            return (f"[green]✔  {result.get('message', '')}[/]", False)
        elif action == "exists":
            return (f"[green]✔  {result.get('message', '')}[/]", False)
        return (f"[dim]{result.get('message', '')}[/]", False)

    def _finish(self) -> None:
        self.app.push_screen(DoneScreen())

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "next":
            s = self.app.state
            status = self.query_one("#config_status", Static)
            lines: list[str] = []

            # ── Shell profiles ──
            for stype, path in s.selected_profiles:
                append_shell_function(stype, path, s.port, get_repo_root())
                lines.append(f"[green]✔  Added to {path}[/green]")
            skipped = sum(1 for _, path in s.available_profiles
                         if shell_function_is_present(ShellType.UNKNOWN, path)) if s.available_profiles else 0
            if skipped > 0:
                lines.append(f"[dim]  Skipped {skipped} already-configured profile(s)[/dim]")
            if not s.selected_profiles and not skipped:
                lines.append("[dim]  No shell profiles selected[/dim]")

            # ── Codex ──
            if s.configure_codex:
                lines.append(self._write_codex())

            # ── Claude / statusline ──
            if s.configure_claude:
                msg, blocked = self._write_statusline()
                if blocked:
                    # Modal will call _finish() when done
                    return
                if msg:
                    lines.append(msg)

            if lines:
                status.update("\n".join(lines))
                self.set_timer(0.5, self._finish)
            else:
                self._finish()

        elif event.button.id == "back":
            self.app.pop_screen()


# ─── 9. Done ──────────────────────────────────────────────────

class DoneScreen(Screen):

    def compose(self) -> ComposeResult:
        yield Static("[8 / 8]  Done!", classes="step_header")
        with Container(classes="content_panel", id="done_content"):
            yield Static("Your proxy is ready!", classes="success_label")
            s = self.app.state
            lines: list[str] = []
            lines.append("**Start the proxy:**")
            lines.append("```")
            lines.append(".venv/bin/python start_proxy.py")
            lines.append("```")
            lines.append("")

            if s.configure_claude:
                lines.append("**Use Claude Code (via proxy):**")
                lines.append("```")
                lines.append("claude --proxy")
                lines.append("# or: claudius")
                lines.append("```")
                lines.append("")
            if s.configure_codex:
                lines.append("**Use Codex CLI (via proxy):**")
                lines.append("```")
                lines.append("codex --proxy")
                lines.append("# or: codexius")
                lines.append("```")
                lines.append("*(Codex Desktop app will use proxy automatically)*")
                lines.append("")
            if s.selected_profiles:
                profiles_str = ", ".join(f"`{path}`" for _, path in s.selected_profiles)
                lines.append(f"**Restart shells** to activate shortcuts: {profiles_str}")
                lines.append("")
            lines.append("**Alternative: Docker** (if available)")
            lines.append("```")
            lines.append("docker compose up --build -d")
            lines.append("# stop: docker compose down")
            lines.append("```")
            lines.append("")

            lines.append(f"**Dashboard:**  http://localhost:{s.port}/dashboard")
            lines.append("")
            lines.append("[dim]Hit Finish to close this wizard.[/dim]")

            yield Markdown("\n".join(lines))
        with Horizontal(classes="button_bar"):
            yield Button("Finish", variant="success", id="finish")

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.app.exit()
