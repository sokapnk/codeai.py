#!/usr/bin/env python3
"""
Ollama TUI — A terminal UI for chatting with local LLMs via Ollama's HTTP API.

Features:
  - Vim-style navigation (j/k/g/G/h/l/i/Esc/Tab)
  - Model selector on startup
  - Real-time streaming responses
  - Dual-pane layout (Chat + Code)
  - Code block extraction, editing, and saving
  - Rich color theming with mode-aware status bar
  - 30 FPS redraw cap, graceful resize handling

Usage:
    python3 ollama_tui.py

Requires: Python 3.7+, Ollama running at localhost:11434
"""

import curses
import curses.textpad
import json
import os
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"
ENDPOINT_TAGS = f"{OLLAMA_BASE}/api/tags"
ENDPOINT_CHAT = f"{OLLAMA_BASE}/api/chat"
FPS = 30
FRAME_SEC = 1.0 / FPS

# Colour-pair IDs (must match init_colors)
C_CYAN = 1
C_GREEN = 2
C_YELLOW = 3
C_MAGENTA = 4
C_RED = 5
C_BLUE = 6
C_WHITE = 7

# Mode names
MODE_MODEL_SELECT = "model_select"
MODE_NORMAL = "normal"
MODE_INSERT = "insert"
MODE_COMMAND = "command"

# Default system prompt
SYSTEM_PROMPT = (
    "You are a helpful coding assistant. "
    "When writing code, always use fenced code blocks with a language tag."
)

# Language → default filename mapping (≥ 15 entries)
LANG_FILENAME = {
    "python": "main.py",
    "py": "main.py",
    "python3": "main.py",
    "javascript": "index.js",
    "js": "index.js",
    "typescript": "index.ts",
    "ts": "index.ts",
    "rust": "main.rs",
    "rs": "main.rs",
    "bash": "script.sh",
    "sh": "script.sh",
    "shell": "script.sh",
    "zsh": "script.sh",
    "c": "main.c",
    "cpp": "main.cpp",
    "c++": "main.cpp",
    "cc": "main.cpp",
    "java": "Main.java",
    "go": "main.go",
    "ruby": "main.rb",
    "rb": "main.rb",
    "perl": "script.pl",
    "pl": "script.pl",
    "php": "index.php",
    "lua": "main.lua",
    "sql": "query.sql",
    "html": "index.html",
    "css": "style.css",
    "scss": "style.scss",
    "json": "data.json",
    "yaml": "config.yaml",
    "yml": "config.yml",
    "toml": "config.toml",
    "xml": "data.xml",
    "markdown": "README.md",
    "md": "README.md",
    "dockerfile": "Dockerfile",
    "docker": "Dockerfile",
    "makefile": "Makefile",
    "vim": ".vimrc",
    "elixir": "main.ex",
    "ex": "main.ex",
    "erlang": "main.erl",
    "erl": "main.erl",
    "haskell": "Main.hs",
    "hs": "Main.hs",
    "scala": "Main.scala",
    "kotlin": "Main.kt",
    "kt": "Main.kt",
    "swift": "main.swift",
    "dart": "main.dart",
    "r": "script.R",
    "julia": "main.jl",
    "zig": "main.zig",
    "nix": "default.nix",
}

# Box-drawing characters
TL = "╭"; TR = "╮"; BL = "╰"; BR = "╯"
HZ = "─"; VT = "│"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def suppress_stderr():
    """Redirect stderr to /dev/null to prevent thread tracebacks from
    destroying the TUI. Returns the original stderr fd so it can be restored."""
    orig = sys.stderr
    try:
        devnull = open(os.devnull, "w")
        sys.stderr = devnull
    except Exception:
        devnull = None
    return orig, devnull


def restore_stderr(orig, devnull):
    """Restore stderr to its original destination."""
    sys.stderr = orig
    if devnull:
        try:
            devnull.close()
        except Exception:
            pass


def fetch_models() -> List[str]:
    """Fetch model names from Ollama /api/tags. Returns [] on failure."""
    try:
        req = urllib.request.Request(ENDPOINT_TAGS, method="GET")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def stream_chat(model: str, messages: List[Dict], on_chunk, on_done, on_error):
    """Start a streaming chat request in a daemon thread.

    *on_chunk(content_part)* is called for each token.
    *on_done()* is called when the stream completes.
    *on_error(msg)* is called on any failure.
    """

    def _worker():
        try:
            payload = json.dumps({
                "model": model,
                "messages": messages,
                "stream": True,
            }).encode()
            req = urllib.request.Request(ENDPOINT_CHAT, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    part = chunk.get("message", {}).get("content", "")
                    if part:
                        on_chunk(part)
                    if chunk.get("done", False):
                        break
            on_done()
        except Exception as exc:
            on_error(str(exc))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def parse_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract fenced code blocks from markdown text.

    Returns list of (language, code) tuples.
    """
    pattern = re.compile(r"```(\w*)\s*\n(.*?)```", re.DOTALL)
    return [(m.group(1) or "text", m.group(2).rstrip("\n")) for m in pattern.finditer(text)]


# ---------------------------------------------------------------------------
# Application State
# ---------------------------------------------------------------------------


class AppState:
    """All mutable UI state — protected by a lock for thread safety."""

    def __init__(self):
        self.lock = threading.Lock()

        # Modes & focus
        self.mode = MODE_MODEL_SELECT
        self.focus = "chat"  # "chat" or "code"

        # Model selector
        self.models: List[str] = []
        self.model_idx = 0
        self.model_fetch_ok = False

        # Chat
        self.history: List[Dict[str, str]] = []
        self.chat_lines: List[Tuple[str, int]] = []  # (text, color_pair)
        self.chat_scroll = 0

        # Code blocks
        self.blocks: List[Tuple[str, str]] = []  # (lang, code)
        self.block_idx = 0
        self.code_scroll = 0

        # Input buffers
        self.input_buf = ""
        self.cmd_buf = ""

        # Generation state
        self.is_generating = False
        self.partial_response = ""
        self.spinner_frame = 0

        # Status / toast
        self.toast = "Select a model"
        self.status_color = C_CYAN

        # Timing
        self.last_draw = 0.0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def _safe_addstr(win, y, x, s, attr=0):
    """Add a string to a curses window, clamping to its dimensions."""
    my, mx = win.getmaxyx()
    if y < 0 or y >= my or x < 0 or x >= mx:
        return
    # Truncate to fit
    max_len = mx - x
    if max_len <= 0:
        return
    s = s[:max_len]
    try:
        win.addstr(y, x, s, attr)
    except curses.error:
        pass  # bottom-right corner write


def _draw_box(win, y, x, h, w, color_pair):
    """Draw a box using box-drawing characters with the given color."""
    attr = curses.color_pair(color_pair)
    _safe_addstr(win, y, x, TL + HZ * (w - 2) + TR, attr)
    for row in range(1, h - 1):
        _safe_addstr(win, y + row, x, VT, attr)
        _safe_addstr(win, y + row, x + w - 1, VT, attr)
    _safe_addstr(win, y + h - 1, x, BL + HZ * (w - 2) + BR, attr)


def _draw_box_with_header(win, y, x, h, w, header, color_pair):
    """Draw a bordered box with a centered header label on the top edge."""
    _draw_box(win, y, x, h, w, color_pair)
    if header:
        # Overwrite the top border with the header text
        attr = curses.color_pair(color_pair) | curses.A_BOLD
        hx = x + max(1, (w - len(header) - 2) // 2)
        # Build the header segment: "─ Header ─"
        seg = HZ + " " + header + " " + HZ
        _safe_addstr(win, y, hx, seg, attr)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


def main(stdscr):
    """Entry point wrapped by curses.wrapper()."""

    # ── Curses setup ──────────────────────────────────────────────────
    curses.curs_set(0)
    curses.noecho()
    curses.raw()
    stdscr.timeout(int(FRAME_SEC * 1000))  # non-blocking getch at ~30 FPS

    # Colours
    try:
        curses.use_default_colors()
        bg = -1
    except Exception:
        bg = 0

    curses.init_pair(C_CYAN, curses.COLOR_CYAN, bg)
    curses.init_pair(C_GREEN, curses.COLOR_GREEN, bg)
    curses.init_pair(C_YELLOW, curses.COLOR_YELLOW, bg)
    curses.init_pair(C_MAGENTA, curses.COLOR_MAGENTA, bg)
    curses.init_pair(C_RED, curses.COLOR_RED, bg)
    curses.init_pair(C_BLUE, curses.COLOR_BLUE, bg)
    curses.init_pair(C_WHITE, curses.COLOR_WHITE, bg)

    # ── State ─────────────────────────────────────────────────────────
    S = AppState()

    # ── Fetch models (background) ─────────────────────────────────────
    def _on_models_fetched(model_list):
        with S.lock:
            S.models = model_list
            S.model_fetch_ok = len(model_list) > 0
            if S.model_fetch_ok:
                S.toast = f"{len(model_list)} model(s) available"
            else:
                S.toast = "Error: Is Ollama running?"

    def _fetch_models_bg():
        models = fetch_models()
        _on_models_fetched(models)

    threading.Thread(target=_fetch_models_bg, daemon=True).start()

    # ── Chat streaming callbacks ──────────────────────────────────────
    def _on_chunk(part):
        with S.lock:
            S.partial_response += part
            # Update the last chat line (assistant partial)
            if S.chat_lines and S.chat_lines[-1][1] == C_WHITE:
                # Replace last assistant line
                S.chat_lines[-1] = (S.partial_response, C_WHITE)
            # If no assistant line yet, append
            # (handled in _start_generation)

    def _on_stream_done():
        with S.lock:
            full = S.partial_response
            S.is_generating = False
            S.toast = "Ready"
            S.status_color = C_CYAN
            # Finalize the assistant message
            S.history.append({"role": "assistant", "content": full})
            # Ensure the chat line is the final version
            if S.chat_lines and S.chat_lines[-1][1] == C_WHITE:
                S.chat_lines[-1] = (full, C_WHITE)
            # Parse code blocks
            new_blocks = parse_code_blocks(full)
            if new_blocks:
                prev_count = len(S.blocks)
                S.blocks.extend(new_blocks)
                added = len(S.blocks) - prev_count
                # Auto-switch focus to code pane
                S.focus = "code"
                S.block_idx = len(S.blocks) - 1
                S.code_scroll = 0
                S.toast = f"Found {added} new code block(s)"
                S.status_color = C_GREEN

    def _on_stream_error(msg):
        with S.lock:
            S.is_generating = False
            S.toast = "Error"
            S.status_color = C_RED
            S.chat_lines.append((f"[Error] {msg}", C_RED))

    def _start_generation():
        """Initiate a streaming chat request with the current history."""
        with S.lock:
            S.is_generating = True
            S.partial_response = ""
            S.toast = "Thinking..."
            S.status_color = C_CYAN
            # Add a placeholder line for the assistant's response
            S.chat_lines.append(("", C_WHITE))
            # Copy history for the request (avoid mutation during stream)
            msgs = list(S.history)
            model = S.models[S.model_idx] if S.models else "unknown"

        # Submit in a background thread
        def _go():
            stream_chat(model, msgs, _on_chunk, _on_stream_done, _on_stream_error)

        threading.Thread(target=_go, daemon=True).start()

    # ── Save command ──────────────────────────────────────────────────
    def _do_save(path: str):
        with S.lock:
            if not S.blocks or S.block_idx >= len(S.blocks):
                S.toast = "No block to save"
                S.status_color = C_RED
                return
            lang, code = S.blocks[S.block_idx]
        try:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            # Ensure trailing newline
            if code and not code.endswith("\n"):
                code += "\n"
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            with S.lock:
                S.toast = f"saved -> {path}"
                S.status_color = C_GREEN
                S.chat_lines.append((f"[Saved] {path}", C_GREEN))
        except Exception as exc:
            with S.lock:
                S.toast = str(exc)
                S.status_color = C_RED
                S.chat_lines.append((f"[Save Error] {exc}", C_RED))

    # ── Edit block ────────────────────────────────────────────────────
    def _edit_block():
        """Open a fullscreen textpad for editing the current code block."""
        with S.lock:
            if not S.blocks or S.block_idx >= len(S.blocks):
                S.toast = "No block to edit"
                return
            lang, code = S.blocks[S.block_idx]
            block_num = S.block_idx + 1

        # Save current screen state
        stdscr.clear()
        maxy, maxx = stdscr.getmaxyx()

        # Draw header
        header = f" EDITING BLOCK {block_num} ({lang}) | Ctrl-G to save "
        _safe_addstr(stdscr, 0, 0, header, curses.A_REVERSE | curses.color_pair(C_YELLOW))

        # Create a sub-window for the textpad (rows 1..maxy-2)
        edit_h = maxy - 2
        edit_w = maxx
        edit_win = curses.newwin(edit_h, edit_w, 1, 0)
        edit_win.keypad(True)

        # Pre-fill with the code content
        for i, line in enumerate(code.split("\n")):
            if i < edit_h:
                _safe_addstr(edit_win, i, 0, line)

        # Create the textbox
        curses.curs_set(1)
        textbox = curses.textpad.Textbox(edit_win)
        try:
            # Let the user edit — Ctrl-G (or ESC mapped below) confirms
            result = textbox.edit()
        except Exception:
            result = edit_win.instr(0, 0).decode("utf-8", errors="replace")

        curses.curs_set(0)

        # Trim trailing whitespace per line
        new_code = "\n".join(line.rstrip() for line in result.split("\n"))
        # Also strip leading empty lines the textbox might add
        new_code = new_code.strip("\n")

        with S.lock:
            S.blocks[S.block_idx] = (lang, new_code)
            S.toast = f"Block {block_num} updated"

        stdscr.clear()

    # ── Word-wrap helper ──────────────────────────────────────────────
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text to the given width, preserving explicit newlines."""
        if width <= 0:
            return [text]
        lines = []
        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            while len(paragraph) > width:
                # Find a good break point
                break_at = width
                # Try to break at a space
                space = paragraph.rfind(" ", 0, width + 1)
                if space > 0:
                    break_at = space
                lines.append(paragraph[:break_at])
                paragraph = paragraph[break_at:].lstrip(" ")
            if paragraph:
                lines.append(paragraph)
        return lines

    # ── Render model selector ─────────────────────────────────────────
    def _render_model_select():
        stdscr.clear()
        maxy, maxx = stdscr.getmaxyx()
        with S.lock:
            models = list(S.models)
            idx = S.model_idx
            fetch_ok = S.model_fetch_ok

        # Title
        title = " Ollama TUI — Select a Model "
        tx = max(0, (maxx - len(title)) // 2)
        _safe_addstr(stdscr, 1, tx, title, curses.A_BOLD | curses.color_pair(C_CYAN))

        if not models and not fetch_ok:
            # Still loading or error
            err = "Error: Is Ollama running?"
            ex = max(0, (maxx - len(err)) // 2)
            _safe_addstr(stdscr, 4, ex, err, curses.A_BOLD | curses.color_pair(C_RED))
            hint = "Press q to quit"
            hx = max(0, (maxx - len(hint)) // 2)
            _safe_addstr(stdscr, 6, hx, hint, curses.color_pair(C_WHITE))
            return

        # List models
        list_h = min(len(models), maxy - 6)
        start_y = 3
        for i in range(list_h):
            label = f"  {models[i]}  "
            is_active = (i == idx)
            attr = (curses.A_REVERSE | curses.A_BOLD | curses.color_pair(C_GREEN)) if is_active else curses.color_pair(C_WHITE)
            ly = start_y + i
            lx = max(0, (maxx - len(label)) // 2)
            _safe_addstr(stdscr, ly, lx, label, attr)

        # Navigation hint
        hint = "j/k: navigate  Enter/l: select  q: quit"
        hx = max(0, (maxx - len(hint)) // 2)
        _safe_addstr(stdscr, maxy - 2, hx, hint, curses.A_DIM | curses.color_pair(C_WHITE))

    # ── Render main UI ────────────────────────────────────────────────
    def _render_main():
        stdscr.clear()
        maxy, maxx = stdscr.getmaxyx()

        with S.lock:
            mode = S.mode
            focus = S.focus
            chat_lines = list(S.chat_lines)
            chat_scroll = S.chat_scroll
            blocks = list(S.blocks)
            block_idx = S.block_idx
            code_scroll = S.code_scroll
            is_generating = S.is_generating
            partial = S.partial_response
            toast = S.toast
            status_color = S.status_color
            model_name = S.models[S.model_idx] if S.models else "?"
            input_buf = S.input_buf
            cmd_buf = S.cmd_buf
            spinner = S.spinner_frame

        # Layout calculations
        status_h = 1
        input_h = 1
        top_margin = 0
        avail_h = maxy - status_h - input_h - top_margin
        chat_h = max(3, int(avail_h * 0.6))
        code_h = max(3, avail_h - chat_h)

        chat_y = top_margin
        code_y = chat_y + chat_h

        # Border colours based on focus
        chat_border_c = C_CYAN if focus == "chat" else C_WHITE
        code_border_c = C_CYAN if focus == "code" else C_WHITE
        chat_border_attr = curses.color_pair(chat_border_c)
        code_border_attr = curses.color_pair(code_border_c)
        if focus != "chat":
            chat_border_attr |= curses.A_DIM
        if focus != "code":
            code_border_attr |= curses.A_DIM

        # ── Chat pane ─────────────────────────────────────────────────
        _draw_box(stdscr, chat_y, 0, chat_h, maxx, chat_border_c)
        # Header for chat pane
        chat_header = " Chat "
        _safe_addstr(stdscr, chat_y, 2, chat_header, chat_border_attr | curses.A_BOLD)

        # Render chat content inside the box
        inner_x = 2
        inner_w = maxx - 4
        inner_y_start = chat_y + 1
        inner_h = chat_h - 2  # usable rows

        # Word-wrap all chat lines
        wrapped = []
        for text, color in chat_lines:
            for wl in _wrap_text(text, inner_w):
                wrapped.append((wl, color))

        # While generating, also include the partial response if it's the last line
        if is_generating and partial:
            # The partial is already in chat_lines via _on_chunk; just ensure scroll
            pass

        total_chat = len(wrapped)
        # Clamp scroll
        chat_scroll = max(0, min(chat_scroll, total_chat - inner_h))
        with S.lock:
            S.chat_scroll = chat_scroll

        # Draw visible lines
        for i in range(inner_h):
            line_idx = chat_scroll + i
            if line_idx < total_chat:
                text, color = wrapped[line_idx]
                attr = curses.color_pair(color)
                _safe_addstr(stdscr, inner_y_start + i, inner_x, text, attr)

        # ── Code pane ─────────────────────────────────────────────────
        code_header_text = ""
        if blocks:
            lang, code = blocks[block_idx] if block_idx < len(blocks) else ("", "")
            code_header_text = f" Code Block {block_idx + 1}/{len(blocks)} ({lang}) "
        else:
            code_header_text = " Code (no blocks) "

        _draw_box_with_header(stdscr, code_y, 0, code_h, maxx, code_header_text, code_border_c)

        # Render code content
        code_inner_x = 2
        code_inner_w = maxx - 4
        code_inner_y_start = code_y + 1
        code_inner_h = code_h - 2

        if blocks and block_idx < len(blocks):
            _, code = blocks[block_idx]
            code_lines_raw = code.split("\n")
            # Add line numbers
            code_lines = []
            for i, cl in enumerate(code_lines_raw, 1):
                num = f"{i:>4}{VT}"
                code_lines.append((num, cl))

            total_code = len(code_lines)
            code_scroll = max(0, min(code_scroll, total_code - code_inner_h))
            with S.lock:
                S.code_scroll = code_scroll

            for i in range(code_inner_h):
                line_idx = code_scroll + i
                if line_idx < total_code:
                    num_str, code_str = code_lines[line_idx]
                    # Line number in dim blue
                    _safe_addstr(stdscr, code_inner_y_start + i, code_inner_x,
                                 num_str, curses.A_DIM | curses.color_pair(C_BLUE))
                    # Code text in white
                    display_code = code_str[:code_inner_w - 6] if code_inner_w > 6 else code_str
                    _safe_addstr(stdscr, code_inner_y_start + i, code_inner_x + 5,
                                 display_code, curses.color_pair(C_WHITE))

        # ── Input line ────────────────────────────────────────────────
        input_y = maxy - 2  # one row above status bar
        if mode == MODE_INSERT:
            prompt_str = "> " + input_buf
            _safe_addstr(stdscr, input_y, 0, prompt_str, curses.color_pair(C_GREEN))
            # Position cursor
            curses.curs_set(1)
            stdscr.move(input_y, 2 + len(input_buf))
        elif mode == MODE_COMMAND:
            prompt_str = ": " + cmd_buf
            _safe_addstr(stdscr, input_y, 0, prompt_str, curses.color_pair(C_YELLOW))
            curses.curs_set(1)
            stdscr.move(input_y, 2 + len(cmd_buf))
        else:
            curses.curs_set(0)
            if mode == MODE_NORMAL:
                hint = "i:insert  :cmd  Tab:focus  j/k:scroll  e:edit  s:save  q:quit"
                _safe_addstr(stdscr, input_y, 0, hint, curses.A_DIM | curses.color_pair(C_WHITE))

        # ── Status bar ────────────────────────────────────────────────
        status_y = maxy - 1
        # Mode badge
        mode_label = f" {mode.upper()} "
        # Determine status bar color
        if mode == MODE_INSERT:
            sb_color = C_GREEN
        elif mode == MODE_COMMAND:
            sb_color = C_YELLOW
        elif status_color == C_RED:
            sb_color = C_RED
        elif status_color == C_GREEN:
            sb_color = C_GREEN
        else:
            sb_color = C_CYAN
        sb_attr = curses.A_REVERSE | curses.color_pair(sb_color)

        # Spinner
        spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        sp = spinner_chars[spinner % len(spinner_chars)] if is_generating else ""

        left = mode_label
        center = f" {model_name} "
        right = f" {toast} {sp} "

        _safe_addstr(stdscr, status_y, 0, " " * maxx, sb_attr)
        _safe_addstr(stdscr, status_y, 0, left, sb_attr | curses.A_BOLD)
        cx = max(0, (maxx - len(center)) // 2)
        _safe_addstr(stdscr, status_y, cx, center, sb_attr)
        rx = max(0, maxx - len(right) - 1)
        _safe_addstr(stdscr, status_y, rx, right, sb_attr)

    # ── Main loop ─────────────────────────────────────────────────────
    running = True
    while running:
        now = time.time()

        # Render
        with S.lock:
            current_mode = S.mode
            S.spinner_frame += 1

        if current_mode == MODE_MODEL_SELECT:
            _render_model_select()
        else:
            _render_main()

        stdscr.refresh()

        # Input
        try:
            ch = stdscr.getch()
        except Exception:
            ch = -1

        if ch == curses.KEY_RESIZE:
            curses.update_lines_columns()
            stdscr.clear()
            continue

        if ch == -1:
            # No input this frame — throttle
            elapsed = time.time() - now
            if elapsed < FRAME_SEC:
                time.sleep(FRAME_SEC - elapsed)
            continue

        # ── Dispatch input by mode ────────────────────────────────────

        with S.lock:
            mode = S.mode

        # ---- MODEL SELECT ----
        if mode == MODE_MODEL_SELECT:
            with S.lock:
                n = len(S.models)

            if ch == ord("q"):
                running = False
            elif ch in (ord("j"), curses.KEY_DOWN):
                with S.lock:
                    if n:
                        S.model_idx = (S.model_idx + 1) % n
            elif ch in (ord("k"), curses.KEY_UP):
                with S.lock:
                    if n:
                        S.model_idx = (S.model_idx - 1) % n
            elif ch in (ord("\n"), curses.KEY_ENTER, ord("l")):
                with S.lock:
                    if S.models:
                        S.mode = MODE_NORMAL
                        S.focus = "chat"
                        S.toast = "Ready"
                        S.status_color = C_CYAN
                        S.chat_lines.append(
                            (f"Connected to {S.models[S.model_idx]}", C_CYAN)
                        )

        # ---- NORMAL ----
        elif mode == MODE_NORMAL:
            with S.lock:
                focus = S.focus
                has_blocks = len(S.blocks) > 0

            if ch == ord("q"):
                running = False
            elif ch == ord("i") or ch == ord("\n") or ch == curses.KEY_ENTER:
                with S.lock:
                    S.mode = MODE_INSERT
                    S.input_buf = ""
                    S.toast = "INSERT"
            elif ch == ord(":"):
                with S.lock:
                    S.mode = MODE_COMMAND
                    S.cmd_buf = ""
                    S.toast = "COMMAND"
            elif ch == ord("\t"):
                with S.lock:
                    S.focus = "code" if S.focus == "chat" else "chat"
            elif ch in (ord("j"), curses.KEY_DOWN):
                with S.lock:
                    if S.focus == "chat":
                        S.chat_scroll += 1
                    else:
                        S.code_scroll += 1
            elif ch in (ord("k"), curses.KEY_UP):
                with S.lock:
                    if S.focus == "chat":
                        S.chat_scroll = max(0, S.chat_scroll - 1)
                    else:
                        S.code_scroll = max(0, S.code_scroll - 1)
            elif ch == ord("g"):
                with S.lock:
                    if S.focus == "chat":
                        S.chat_scroll = 0
                    else:
                        S.code_scroll = 0
            elif ch == ord("G"):
                with S.lock:
                    # Jump to bottom — scroll as far as possible
                    # Actual max will be clamped during render
                    S.chat_scroll = 99999 if S.focus == "chat" else S.chat_scroll
                    S.code_scroll = 99999 if S.focus == "code" else S.code_scroll
            elif ch == ord("h"):
                # Previous code block (wraps)
                with S.lock:
                    if S.focus == "code" and S.blocks:
                        S.block_idx = (S.block_idx - 1) % len(S.blocks)
                        S.code_scroll = 0
            elif ch == ord("l"):
                # Next code block (wraps)
                with S.lock:
                    if S.focus == "code" and S.blocks:
                        S.block_idx = (S.block_idx + 1) % len(S.blocks)
                        S.code_scroll = 0
            elif ch == ord("e"):
                # Edit current block
                with S.lock:
                    if S.focus == "code" and S.blocks:
                        should_edit = True
                    else:
                        should_edit = False
                if should_edit:
                    _edit_block()
            elif ch == ord("s"):
                # Save current block — pre-fill command
                with S.lock:
                    if S.focus == "code" and S.blocks and S.block_idx < len(S.blocks):
                        lang = S.blocks[S.block_idx][0]
                        default_name = LANG_FILENAME.get(lang.lower(), f"code.{lang}")
                        S.mode = MODE_COMMAND
                        S.cmd_buf = f"save {default_name}"
                        S.toast = "COMMAND"

        # ---- INSERT ----
        elif mode == MODE_INSERT:
            if ch == 27:  # ESC
                with S.lock:
                    S.mode = MODE_NORMAL
                    S.toast = "Ready"
                    S.status_color = C_CYAN
            elif ch in (ord("\n"), curses.KEY_ENTER):
                with S.lock:
                    user_text = S.input_buf.strip()
                    S.input_buf = ""

                if user_text:
                    with S.lock:
                        # First message — inject system prompt
                        if not S.history:
                            S.history.append({"role": "system", "content": SYSTEM_PROMPT})
                        S.history.append({"role": "user", "content": user_text})
                        S.chat_lines.append((f"You: {user_text}", C_GREEN | curses.A_BOLD))
                        S.mode = MODE_NORMAL
                        S.toast = "Generating..."
                        S.status_color = C_CYAN

                    _start_generation()
                else:
                    with S.lock:
                        S.mode = MODE_NORMAL
                        S.toast = "Ready"
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                with S.lock:
                    S.input_buf = S.input_buf[:-1]
            elif ch == curses.KEY_RESIZE:
                pass  # handled above
            elif 32 <= ch <= 126:  # printable ASCII
                with S.lock:
                    S.input_buf += chr(ch)

        # ---- COMMAND ----
        elif mode == MODE_COMMAND:
            if ch == 27:  # ESC
                with S.lock:
                    S.mode = MODE_NORMAL
                    S.cmd_buf = ""
                    S.toast = "Ready"
                    S.status_color = C_CYAN
            elif ch in (ord("\n"), curses.KEY_ENTER):
                with S.lock:
                    cmd = S.cmd_buf.strip()
                    S.cmd_buf = ""
                    S.mode = MODE_NORMAL

                if cmd.startswith("save "):
                    path = cmd[5:].strip()
                    if path:
                        _do_save(path)
                    else:
                        with S.lock:
                            S.toast = "Usage: save <path>"
                            S.status_color = C_RED
                elif cmd == "q" or cmd == "quit":
                    running = False
                else:
                    with S.lock:
                        S.toast = f"Unknown command: {cmd}"
                        S.status_color = C_RED
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                with S.lock:
                    S.cmd_buf = S.cmd_buf[:-1]
            elif 32 <= ch <= 126:
                with S.lock:
                    S.cmd_buf += chr(ch)

        # Throttle
        elapsed = time.time() - now
        if elapsed < FRAME_SEC:
            time.sleep(FRAME_SEC - elapsed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    """Suppress stderr, launch curses app, restore stderr on exit."""
    orig_err, devnull = suppress_stderr()
    try:
        curses.wrapper(main)
    finally:
        restore_stderr(orig_err, devnull)


if __name__ == "__main__":
    run()
