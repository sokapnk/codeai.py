#!/usr/bin/env python3
"""
qwen-coder — chat with a local LLM, grab code blocks, save them.

Keys:
  i       type a prompt          Enter   send
  j / k   scroll chat            h / l   prev/next code block
  g / G   scroll top / bottom    s       save code block
  :       command mode            q       quit

Commands:
  :save <file>   save current code block
  :help          show help
"""

import curses
import json
import os
import urllib.error
import urllib.request

# ── config ───────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:3b"
SYSTEM_PROMPT = (
    "You are a concise coding assistant. "
    "Always use fenced code blocks with a language tag. "
    "Keep explanations brief."
)

LANG_FILE = {
    "python": "main.py", "py": "main.py",
    "javascript": "index.js", "js": "index.js",
    "typescript": "index.ts", "ts": "index.ts",
    "rust": "main.rs", "go": "main.go",
    "c": "main.c", "cpp": "main.cpp",
    "java": "Main.java",
    "ruby": "main.rb", "rb": "main.rb",
    "bash": "script.sh", "sh": "script.sh", "shell": "script.sh",
    "html": "index.html", "css": "style.css",
    "sql": "query.sql", "lua": "main.lua",
    "json": "data.json",
    "yaml": "config.yaml", "yml": "config.yaml",
    "dockerfile": "Dockerfile", "makefile": "Makefile",
}

# ── ollama ───────────────────────────────────────────────────────────


def ask_ollama(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "stream": False}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"]
    except urllib.error.URLError as e:
        return f"[connection error - is ollama running? {e.reason}]"
    except Exception as e:
        return f"[error: {e}]"

# ── code block parser ────────────────────────────────────────────────


def parse_code_blocks(text):
    blocks, buf, lang, inside = [], [], "", False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("```") and not inside:
            inside, lang, buf = True, s[3:].strip(), []
        elif s == "```" and inside:
            inside = False
            blocks.append((lang, "\n".join(buf)))
        elif inside:
            buf.append(line)
    if inside and buf:
        blocks.append((lang, "\n".join(buf)))
    return blocks

# ── app ──────────────────────────────────────────────────────────────


class App:
    # color pair ids
    _CY, _GR, _YE, _MA, _RE, _BL, _WH = range(1, 8)

    def __init__(self, scr):
        self.scr = scr
        curses.curs_set(0)
        scr.keypad(True)
        scr.timeout(100)
        self._init_colors()

        self.alive = True
        self.mode = "normal"
        self.inbuf = ""
        self.cmdbuf = ""
        self.toast = "ready"

        self.history = []
        self.chat = []
        self.cscroll = 0

        self.blocks = []
        self.bi = 0
        self.bscroll = 0

        self._log("  qwen-coder", self.BOLD | self.C["ma"])
        self._log("  i:chat  s:save  j/k:scroll  h/l:block  q:quit", self.DIM | self.C["wh"])
        self._log("")

    def _init_colors(self):
        self.BOLD = curses.A_BOLD
        self.DIM = curses.A_DIM
        self.REV = curses.A_REVERSE
        self.C = {}
        if not curses.has_colors():
            for k in ("cy", "gr", "ye", "ma", "re", "bl", "wh"):
                self.C[k] = curses.A_NORMAL
            return
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except Exception:
            bg = 0
        pairs = [
            (self._CY, curses.COLOR_CYAN, "cy"),
            (self._GR, curses.COLOR_GREEN, "gr"),
            (self._YE, curses.COLOR_YELLOW, "ye"),
            (self._MA, curses.COLOR_MAGENTA, "ma"),
            (self._RE, curses.COLOR_RED, "re"),
            (self._BL, curses.COLOR_BLUE, "bl"),
            (self._WH, curses.COLOR_WHITE, "wh"),
        ]
        for pid, fg, key in pairs:
            curses.init_pair(pid, fg, bg)
            self.C[key] = curses.color_pair(pid)

    def _log(self, text, attr=curses.A_NORMAL):
        self.chat.append((text, attr))

    # ── main loop ──

    def run(self):
        while self.alive:
            self._draw()
            self._input()

    # ── layout ──

    def _layout(self):
        """Return (chat_top, chat_h, code_top, code_h, input_top, status_y)."""
        H, W = self.scr.getmaxyx()
        # fixed zones from bottom up
        status_y = H - 1
        input_top = H - 4          # 3 rows: border, content, border
        sep_y = input_top - 1      # separator line
        # split the space above separator into chat + code
        above_sep = sep_y          # rows 0..sep_y-1 are available
        code_h = max(above_sep // 3, 2)
        chat_h = above_sep - code_h
        # chat: rows 0..chat_h-1
        # code header: row chat_h (= sep_y - code_h), but we use sep_y for separator
        # Actually let me be precise:
        #   chat occupies rows [0, chat_h)
        #   separator at row chat_h
        #   code occupies rows [chat_h+1, chat_h+1+code_h)
        #   input at row H-4
        #   status at row H-1
        chat_top = 0
        sep_row = chat_h
        code_top = chat_h + 1
        # code_h = input_top - code_top  (fill remaining space)
        code_h = input_top - code_top
        return chat_top, chat_h, code_top, code_h, input_top, status_y, sep_row, W

    # ── drawing ──

    def _draw(self):
        s = self.scr
        H, W = s.getmaxyx()
        if H < 8 or W < 10:
            return
        s.erase()

        chat_top, chat_h, code_top, code_h, input_top, status_y, sep_row, W = self._layout()

        # ── chat ──
        n = len(self.chat)
        max_scroll = max(0, n - chat_h)
        self.cscroll = max(0, min(self.cscroll, max_scroll))
        for i in range(chat_h):
            idx = self.cscroll + i
            if idx < n:
                _s(s, chat_top + i, 0, self.chat[idx][0], self.chat[idx][1])
        if n > chat_h:
            vis = min(self.cscroll + chat_h, n)
            _s(s, chat_top + chat_h - 1, W - 14, f"[{self.cscroll+1}-{vis}/{n}]", self.DIM)

        # ── separator ──
        if self.blocks:
            tag = f" {self.bi+1}/{len(self.blocks)} code "
        else:
            tag = " no code "
        sep = "\u2500" * 3 + tag + "\u2500" * max(W - 3 - len(tag), 0)
        _s(s, sep_row, 0, sep, self.BOLD | self.C["cy"])

        # ── code ──
        if not self.blocks:
            _s(s, code_top, 2, "code blocks appear here after AI responses", self.DIM | self.C["wh"])
        else:
            lang, code = self.blocks[self.bi]
            fname = LANG_FILE.get(lang.lower(), "output.txt")
            _s(s, code_top, 1, f"[{self.bi+1}/{len(self.blocks)}] {lang or 'code'} -> {fname}", self.BOLD | self.C["gr"])
            clines = code.split("\n")
            nc = len(clines)
            # code lines go from code_top+1 to code_top+code_h-1
            line_rows = code_h - 1
            self.bscroll = max(0, min(self.bscroll, max(0, nc - line_rows)))
            for i in range(line_rows):
                ci = self.bscroll + i
                if ci < nc:
                    _s(s, code_top + 1 + i, 1, f"{ci+1:>3} ", self.DIM | self.C["bl"])
                    _s(s, code_top + 1 + i, 5, clines[ci], self.C["wh"])

        # ── input box ──
        _s(s, input_top, 0, "+" + "-" * max(W - 2, 0) + "+")
        _s(s, input_top + 2, 0, "+" + "-" * max(W - 2, 0) + "+")
        if self.mode == "insert":
            _s(s, input_top + 1, 1, "> " + self.inbuf, self.C["gr"])
            curses.curs_set(1)
            s.move(input_top + 1, min(3 + len(self.inbuf), W - 3))
        elif self.mode == "command":
            _s(s, input_top + 1, 1, ":" + self.cmdbuf, self.C["ye"])
            curses.curs_set(1)
            s.move(input_top + 1, min(2 + len(self.cmdbuf), W - 3))
        else:
            _s(s, input_top + 1, 2, "i chat | s save | j/k scroll | h/l block | q quit", self.DIM | self.C["wh"])
            curses.curs_set(0)

        # ── status bar ──
        attr = self.REV | self.C["cy"]
        if "error" in self.toast.lower():
            attr = self.REV | self.C["re"]
        elif "saved" in self.toast.lower():
            attr = self.REV | self.C["gr"]
        _s(s, status_y, 0, f" {self.mode.upper()}  {self.toast}" + " " * W, attr)

        s.refresh()

    # ── input ──

    def _input(self):
        try:
            ch = self.scr.getch()
        except Exception:
            return
        if ch == -1 or ch == curses.KEY_RESIZE:
            return
        {"normal": self._kn, "insert": self._ki, "command": self._kc}[self.mode](ch)

    def _kn(self, ch):
        if ch == ord("q"):
            self.alive = False
        elif ch in (ord("j"), curses.KEY_DOWN):
            self.cscroll += 1
        elif ch in (ord("k"), curses.KEY_UP):
            self.cscroll = max(0, self.cscroll - 1)
        elif ch == ord("g"):
            self.cscroll = 0
        elif ch == ord("G"):
            self.cscroll = len(self.chat)
        elif ch == ord("h") and self.blocks:
            self.bi = max(0, self.bi - 1)
            self.bscroll = 0
        elif ch == ord("l") and self.blocks:
            self.bi = min(len(self.blocks) - 1, self.bi + 1)
            self.bscroll = 0
        elif ch in (ord("i"), ord("\n"), 10):
            self.mode = "insert"
            self.inbuf = ""
        elif ch == ord("s"):
            self._start_save()
        elif ch == ord(":"):
            self.mode = "command"
            self.cmdbuf = ""

    def _ki(self, ch):
        if ch == 27:
            self.mode = "normal"
        elif ch in (curses.KEY_ENTER, 10, 13):
            self._send()
            self.mode = "normal"
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            self.inbuf = self.inbuf[:-1]
        elif 32 <= ch < 127:
            self.inbuf += chr(ch)

    def _kc(self, ch):
        if ch == 27:
            self.mode = "normal"
        elif ch in (curses.KEY_ENTER, 10, 13):
            self._run_cmd()
            self.mode = "normal"
        elif ch in (curses.KEY_BACKSPACE, 127, 8):
            if self.cmdbuf:
                self.cmdbuf = self.cmdbuf[:-1]
            else:
                self.mode = "normal"
        elif 32 <= ch < 127:
            self.cmdbuf += chr(ch)

    # ── save ──

    def _start_save(self):
        if not self.blocks:
            self.toast = "no code blocks to save"
            return
        lang = self.blocks[self.bi][0]
        fname = LANG_FILE.get(lang.lower(), "output.txt")
        self.cmdbuf = f"save {fname}"
        self.mode = "command"

    def _do_save(self, path):
        if not self.blocks:
            self.toast = "nothing to save"
            return
        try:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            code = self.blocks[self.bi][1]
            with open(path, "w") as f:
                f.write(code if code.endswith("\n") else code + "\n")
            self._log(f"  saved -> {path}", self.BOLD | self.C["ye"])
            self.cscroll = len(self.chat)
            self.toast = f"saved -> {path}"
        except Exception as e:
            self.toast = f"save error: {e}"

    # ── commands ──

    def _run_cmd(self):
        cmd = self.cmdbuf.strip()
        if not cmd:
            return
        parts = cmd.split(None, 1)
        name = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        if name == "save" and arg:
            self._do_save(arg)
        elif name == "save":
            self._start_save()
        elif name == "help":
            self._log("  :save <file>  :help", self.DIM | self.C["wh"])
            self.cscroll = len(self.chat)
            self.toast = "save <file> | help"
        else:
            self.toast = f"unknown: {name}"

    # ── chat ──

    def _send(self):
        text = self.inbuf.strip()
        if not text:
            return

        self._log(f"  you: {text}", self.BOLD | self.C["ma"])

        if not self.history:
            self.history.append({"role": "system", "content": SYSTEM_PROMPT})
        self.history.append({"role": "user", "content": text})

        self.toast = "thinking..."
        self._draw()

        reply = ask_ollama(self.history)
        self.history.append({"role": "assistant", "content": reply})

        self._log("  qwen:", self.BOLD | self.C["cy"])
        for line in reply.split("\n"):
            s = line.strip()
            if s.startswith("```"):
                a = self.DIM | self.C["cy"]
            elif s.startswith(("#", "//", "-- ", "/*")):
                a = self.DIM | self.C["wh"]
            elif line and line[0] in (" ", "\t"):
                a = self.C["wh"]
            else:
                a = curses.A_NORMAL
            self._log(f"  {line}", a)
        self._log("")

        self.blocks = parse_code_blocks(reply)
        self.bi = 0
        self.bscroll = 0
        self.cscroll = len(self.chat)
        self.toast = f"{len(self.blocks)} block(s)" if self.blocks else "ready"


def _s(scr, y, x, text, attr=curses.A_NORMAL):
    """Safe addstr that never crashes."""
    H, W = scr.getmaxyx()
    if y < 0 or y >= H or x < 0 or x >= W:
        return
    try:
        scr.addstr(y, x, text[:W - x - 1], attr)
    except curses.error:
        pass


def main():
    try:
        curses.wrapper(lambda s: App(s).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
