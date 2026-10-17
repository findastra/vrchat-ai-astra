"""AI Astra Voice: AI Astra's cognitive core + Claude for language + VRChat chatbox over OSC.

How a turn works
  1. Someone says something (you type it for now; mic input comes later).
  2. AI Astra recalls related past turns from her own memory file.
  3. AI Astra's Norn cognitive core (cognitive_core.py, unchanged) appraises the
     message and picks a goal, a strategy and a directive.
  4. Claude writes AI Astra's reply, following her persona and that directive.
  5. A safety filter checks the reply, then it is sent to the VRChat chatbox.
  6. AI Astra learns which strategies work from your /good and /bad feedback.

Zero third-party dependencies: the Claude API and OSC are spoken with the
Python standard library.  State lives in ai_astra_voice/state/, never in
ai_astra_phoenix_brain.db.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sqlite3
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
AI_ASTRA_ROOT = HERE.parent  # the repo root: AI Astra's backend modules live there
sys.path.insert(0, str(AI_ASTRA_ROOT))
from cognitive_core import NornCognitiveCore  # noqa: E402  (AI Astra's own module, untouched)

STATE_DIR = Path(os.environ.get("AI_ASTRA_VOICE_STATE_DIR", HERE / "state"))
PERSONA_FILE = HERE / "persona.md"
DEFAULT_MODEL = os.environ.get("AI_ASTRA_VOICE_MODEL", "claude-haiku-4-5-20251001")
API_URL = "https://api.anthropic.com/v1/messages"
CHATBOX_LIMIT = 144          # VRChat chatbox max characters per message
CHATBOX_MIN_SECONDS = 3.0    # keep each chunk on screen at least this long
MAX_REPLY_TOKENS = 220

# --------------------------------------------------------------------------
# Safety filter: last line of defense before anything reaches a public world.
# The persona already asks for PG-13; this catches anything that slips.
# --------------------------------------------------------------------------
BLOCKED_PATTERNS = [
    r"\b(kill|murder|stab|impale|behead|dismember|torture)\w*\b",
    r"\b(suicide|self[- ]harm|cutting myself)\b",
    r"\b(nsfw|porn\w*|nude\w*|sex\w*)\b",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",          # phone numbers
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",                 # email addresses
    r"\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|drive|dr)\b",  # street addresses
]
FALLBACK_LINE = "Hmm, let me not say that one out loud. Ask me something else?"


def safety_check(text: str) -> tuple[bool, str]:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return False, pattern
    return True, ""


# --------------------------------------------------------------------------
# Minimal OSC over UDP (VRChat listens on 127.0.0.1:9000)
# --------------------------------------------------------------------------
def _osc_pad(data: bytes) -> bytes:
    return data + b"\x00" * (4 - len(data) % 4)


def osc_message(address: str, *args) -> bytes:
    tags, payload = ",", b""
    for arg in args:
        if isinstance(arg, bool):
            tags += "T" if arg else "F"
        elif isinstance(arg, int):
            tags += "i"
            payload += struct.pack(">i", arg)
        elif isinstance(arg, float):
            tags += "f"
            payload += struct.pack(">f", arg)
        else:
            tags += "s"
            payload += _osc_pad(str(arg).encode("utf-8"))
    return _osc_pad(address.encode("utf-8")) + _osc_pad(tags.encode("utf-8")) + payload


class VRChatChatbox:
    def __init__(self, host="127.0.0.1", port=9000, enabled=True):
        self.addr = (host, port)
        self.enabled = enabled
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def typing(self, on: bool):
        if self.enabled:
            self.sock.sendto(osc_message("/chatbox/typing", bool(on)), self.addr)

    def say(self, text: str):
        """Split into chatbox-sized chunks and show them one after another."""
        for chunk in split_for_chatbox(text):
            print(f"  [chatbox] {chunk}")
            if self.enabled:
                # args: text, send immediately (skip keyboard), play notification sound
                self.sock.sendto(osc_message("/chatbox/input", chunk, True, False), self.addr)
            time.sleep(max(CHATBOX_MIN_SECONDS, len(chunk) / 15.0))


def split_for_chatbox(text: str, limit: int = CHATBOX_LIMIT) -> list[str]:
    words, chunks, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = word[:limit]
    if current:
        chunks.append(current)
    return chunks or [""]


# --------------------------------------------------------------------------
# AI Astra's conversational memory (separate from her statistical brain)
# --------------------------------------------------------------------------
class EpisodeMemory:
    TOKEN = re.compile(r"[a-z0-9']+")
    STOP = NornCognitiveCore.STOPWORDS

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        con.execute(
            """CREATE TABLE IF NOT EXISTS voice_turns (
                   turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                   created_at REAL NOT NULL,
                   speaker TEXT NOT NULL,
                   user_text TEXT NOT NULL,
                   ai_astra_text TEXT NOT NULL,
                   goal TEXT, action TEXT, feedback INTEGER)"""
        )
        con.commit()

    def _terms(self, text):
        return {t for t in self.TOKEN.findall(text.lower()) if len(t) > 2 and t not in self.STOP}

    def recall(self, text: str, limit: int = 3, scan: int = 400):
        terms = self._terms(text)
        rows = self.con.execute(
            "SELECT turn_id, created_at, speaker, user_text, ai_astra_text, feedback "
            "FROM voice_turns ORDER BY turn_id DESC LIMIT ?", (scan,)
        ).fetchall()
        scored = []
        now = time.time()
        for turn_id, created, speaker, user_text, ai_astra_text, feedback in rows:
            overlap = terms & self._terms(user_text + " " + ai_astra_text)
            if not overlap or not terms:
                continue
            lexical = len(overlap) / len(terms)
            recency = 1.0 / (1.0 + (now - created) / 86400.0)      # decays over days
            quality = 0.0 if feedback == -1 else (0.15 if feedback == 1 else 0.05)
            scored.append((lexical * 0.7 + recency * 0.2 + quality, speaker, user_text, ai_astra_text))
        scored.sort(reverse=True)
        return scored[:limit]

    def recent(self, limit: int = 6):
        rows = self.con.execute(
            "SELECT speaker, user_text, ai_astra_text FROM voice_turns ORDER BY turn_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return list(reversed(rows))

    def store(self, speaker, user_text, ai_astra_text, goal, action) -> int:
        cur = self.con.execute(
            "INSERT INTO voice_turns(created_at, speaker, user_text, ai_astra_text, goal, action) VALUES (?,?,?,?,?,?)",
            (time.time(), speaker, user_text, ai_astra_text, goal, action),
        )
        self.con.commit()
        return cur.lastrowid

    def mark(self, turn_id: int, value: int):
        self.con.execute("UPDATE voice_turns SET feedback=? WHERE turn_id=?", (value, turn_id))
        self.con.commit()


# --------------------------------------------------------------------------
# Claude as AI Astra's voice
# --------------------------------------------------------------------------
def call_claude(api_key: str, model: str, system: str, messages: list[dict]) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": MAX_REPLY_TOKENS,
        "system": system,
        "messages": messages,
    }).encode("utf-8")
    request = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(request, timeout=45) as response:
        data = json.loads(response.read().decode("utf-8"))
    return "".join(block.get("text", "") for block in data.get("content", [])).strip()


def build_system_prompt(persona: str, trace: dict, recalled: list, state: dict) -> str:
    mood = ", ".join(f"{k} {v:.2f}" for k, v in state.items())
    lines = [
        persona.strip(),
        "",
        "## AI Astra's mind this turn (from her cognitive core — follow it)",
        f"- Goal: {trace['goal'].replace('_', ' ')}",
        f"- Strategy: {trace['action'].replace('_', ' ')} — {trace['directive']}",
        f"- Why: {'; '.join(trace.get('reasons', []))}",
        f"- Inner state (0-1): {mood}",
        "  Let the state color your tone subtly (high curiosity = more questions,",
        "  high frustration = shorter and more careful, high affiliation = warmer).",
    ]
    if recalled:
        lines += ["", "## Memories that might be relevant (these really happened; don't invent others)"]
        for _, speaker, user_text, ai_astra_text in recalled:
            lines.append(f'- {speaker} said: "{user_text[:160]}" — you replied: "{ai_astra_text[:160]}"')
    else:
        lines += ["", "## Memories: none related. If asked to remember something, say you don't have it yet."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
HELP = """Type what someone in the world said, e.g.
    Astra: hey astra, how's your night going?
    (no name prefix = 'someone')
Commands: /good  /bad  (teach AI Astra whether that reply worked)
          /state (show AI Astra's inner state)   /help   /quit"""


def parse_line(line: str) -> tuple[str, str]:
    match = re.match(r"^\s*([^:]{1,32}):\s*(.+)$", line)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "someone", line.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Astra Voice — AI Astra in the VRChat chatbox, powered by Claude.")
    parser.add_argument("--no-osc", action="store_true", help="print replies only; don't send to VRChat")
    parser.add_argument("--offline", action="store_true", help="no Claude calls; show AI Astra's goal/strategy only")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--osc-port", type=int, default=9000)
    args = parser.parse_args()
    try:  # Windows consoles can choke on emoji in replies
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not args.offline and not api_key:
        print("ANTHROPIC_API_KEY is not set. Set it (see README) or run with --offline.")
        return 2

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(STATE_DIR / "ai_astra_voice.db", check_same_thread=False)
    mind = NornCognitiveCore(con)          # AI Astra's own cognitive core, own state file
    memory = EpisodeMemory(con)
    chatbox = VRChatChatbox(port=args.osc_port, enabled=not args.no_osc)
    persona = PERSONA_FILE.read_text(encoding="utf-8")
    last_turn_id = None

    print("AI Astra Voice is ready." + (" (offline)" if args.offline else f" Model: {args.model}"))
    print("Make sure OSC is enabled in VRChat: Action Menu > Options > OSC > Enabled.\n")
    print(HELP)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line == "/help":
            print(HELP)
            continue
        if line == "/state":
            print(json.dumps(mind.get_snapshot(), indent=2))
            continue
        if line in ("/good", "/bad"):
            if last_turn_id is None:
                print("Nothing to rate yet.")
                continue
            positive = line == "/good"
            mind.apply_feedback(positive)
            memory.mark(last_turn_id, 1 if positive else -1)
            print("AI Astra will lean into that strategy." if positive else "AI Astra will try a different strategy next time.")
            continue

        speaker, text = parse_line(line)
        recalled = memory.recall(text)
        salience = recalled[0][0] if recalled else 0.0
        trace = mind.begin_turn(f"{text}", memory_salience=min(1.0, salience))
        print(f"  [mind] goal={trace['goal']}  strategy={trace['action']}")

        if args.offline:
            mind.observe_response(quality=0.5, coherence=0.5, response_text="")
            continue

        system = build_system_prompt(persona, trace, recalled, mind.state)
        messages = []
        for past_speaker, past_user, past_ai_astra in memory.recent():
            messages.append({"role": "user", "content": f"{past_speaker}: {past_user}"})
            messages.append({"role": "assistant", "content": past_ai_astra})
        messages.append({"role": "user", "content": f"{speaker}: {text}"})

        chatbox.typing(True)
        quality = 0.7
        try:
            reply = call_claude(api_key, args.model, system, messages)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            print(f"  [error] Claude API {error.code}: {detail}")
            reply, quality = "Sorry, my thoughts got scrambled for a second.", 0.1
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  [error] Couldn't reach Claude: {error}")
            reply, quality = "Give me a moment, I lost my train of thought.", 0.1
        chatbox.typing(False)

        ok, reason = safety_check(reply)
        if not ok:
            print(f"  [filter] blocked a reply (matched {reason}): {reply!r}")
            reply, quality = FALLBACK_LINE, 0.2

        chatbox.say(reply)
        mind.observe_response(quality=quality, coherence=0.7, response_text=reply)
        last_turn_id = memory.store(speaker, text, reply, trace["goal"], trace["action"])

    con.close()
    print("Bye from AI Astra.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
