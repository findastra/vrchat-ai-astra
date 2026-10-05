# astra-vrchat

Local preparation for an Astra character in VRChat: a small text bridge between Mai's
HTTP backend and the VRChat chatbox, plus a read-only inventory tool for the Meep avatar
package. Nothing here is live yet.

## Status (2026-09-30, Claude)

| Piece | State |
|---|---|
| `bridge.py` | Reviewed and finished. Standard library only. 40 offline tests pass. Never run against real Mai or real VRChat. |
| `tools/inspect_unitypackage.py` | Written and tested on a synthetic package (9 tests pass). Reports a lock assessment. **Not yet run on Meep or Poppy.** |
| `docs/meep-inventory.md` | Observed facts from Meep's UPLOAD INFORMATION.txt, plus a proposed parameter map (assumptions marked). |
| `docs/poppy-inventory.md` | Fallback base if Meep's license key doesn't work (owner rule): what's in the Poppy folder, cautions, scan commands, parameter map. |
| `docs/next-steps.md` | Open runtime, voice and avatar decisions, and the next live test. |

Tests ran in Claude's Linux workspace (Python 3.11.15), not yet on Windows.

## Commands

From this folder in PowerShell (the bundled Python is enough; nothing to install):

```powershell
$py = "C:\Users\audra\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

# Tests: loopback stubs on random ports only, never port 9000 or Mai
& $py -m unittest discover -s tests -v

# Offline: build and show a chatbox packet, no network
& $py bridge.py preview "Hi, I'm Astra ✨"

# Mai health (GET /health on 127.0.0.1:8765 unless --endpoint is given)
& $py bridge.py status

# Ask Mai. This calls generate_response, which can change Mai's conversation/learning state.
# The answer is printed only. It is never forwarded to VRChat.
& $py bridge.py ask "hello"

# Send ONE reviewed line to VRChat as a chatbox draft (immediate=false, no notification sound).
# The owner still presses send in VRChat.
& $py bridge.py queue "Hi, I'm Astra ✨"
```

If Mai needs a token, set it for the session only: `$env:MAI_BACKEND_AUTH_TOKEN = "..."`. Don't write it to a file.

## What the bridge does and doesn't do

- `status`: GET `/health`. Prints only `ok`, `transport`, `api_name`, `api_version`.
- `ask`: POST `/api` `{"id","method":"generate_response","params":{"user_input"}}`. Requires the same `id` back and `ok: true`, and `result.response` must be a string. Terminal control characters in the reply are escaped before printing.
- `preview`: validates text and prints the OSC packet as hex. It opens no sockets.
- `queue`: sends one UDP packet `/chatbox/input ,sFF <text>` to `127.0.0.1:9000` (or `--port`). UDP gives no delivery receipt, so check VRChat.
- Loopback only: `localhost`/`127.0.0.1`, plain HTTP, no path, query or credentials. Proxies are ignored. Redirects are refused, so the token can't leave the machine.
- Chat limits: 144 UTF-16 units (an emoji counts as 2), at most 9 lines, and no control characters except newline. Wrapped lines can still exceed 9 on screen, so check visually.
- No auto-forwarding, avatar movement, microphone, voice, Discord, background service, or other Mai methods.

## Changes Claude made to the original bridge

- Malformed HTTP responses (`http.client.HTTPException`) now give a clean error, not a traceback.
- `ask` input is capped at 4,000 characters.
- Chat text containing control characters other than newline (ESC, tab, C1) is rejected.
- Mai's reply is printed with terminal escape sequences neutralized, plus a note that it wasn't sent.
- `queue --port` was added so the CLI can be tested against a stub. The default is still 9000.
- The `queue` message now says what actually happened (one UDP packet, no receipt).
- stdout/stderr use UTF-8, so emoji don't crash when output is redirected on Windows.

## Limitations

- Mai's live endpoint, port and token are unknown. The contract comes from `maimain/headless_api.py` and `backend_api.py` as recorded in HANDOFF-CLAUDE.md. Claude did not re-read the archive.
- The OSC chatbox format follows VRChat's docs. Only the byte layout is tested, not how VRChat displays it.
