# AI Astra Voice

AI Astra in the VRChat chatbox. Her **cognitive core** decides what to do (answer, ask,
recall a memory, repair a mistake), and **Claude** writes the words in her voice.

This is a tool of AI Astra's repo. It imports her `cognitive_core.py` from the repo root,
read-only, and never touches her brain. Its own conversation state lives in `voice/state/`
(git-ignored). Delete that folder any time to give AI Astra Voice a fresh start.

No packages to install: it uses the repo's `.venv` when it exists, or any Python 3.11+.

## Setup (one time)

1. **Get an API key.** Go to console.anthropic.com → API Keys → Create Key. Add a
   small amount of credit and set a monthly spend limit (Settings → Limits).
   Then in a normal Command Prompt:

   ```
   setx ANTHROPIC_API_KEY "paste-your-key-here"
   ```

   Close and reopen the Command Prompt afterwards. Keep the key private and don't
   put it in any file inside `C:\ai_astra`.

2. **Turn on OSC in VRChat.** Action Menu → Options → OSC → Enabled.

## Run it

Double-click `Run_AIAstra_Voice.bat`. Then type what people say, with their name first:

```
> Astra: hey astra, how's your night?
  [mind] goal=maintain_connection  strategy=acknowledge_socially
  [chatbox] Hey Astra! Pretty good — what are you working on tonight?
```

| Command  | What it does |
|----------|--------------|
| `/good`  | That reply worked. AI Astra favors that strategy in similar moments. |
| `/bad`   | That reply missed. AI Astra tries a different strategy next time. |
| `/state` | Shows AI Astra's inner state, goals and what she's learned. |
| `/quit`  | Exit. |

Options: `--no-osc` (print only, don't send to VRChat), `--offline` (no Claude calls),
`--model claude-sonnet-5` (smarter but slower and pricier than the default Haiku).

## Shaping AI Astra

Edit `persona.md`. That's her personality, voice and boundaries. Changes apply the
next time you start her.

## Safety

- The persona keeps her PG-13, honest about being an AI, and away from personal info.
- A filter in `ai_astra_voice.py` (`BLOCKED_PATTERNS`) blocks violent, sexual or personal-info
  replies before they reach the chatbox. It's deliberately strict, so it will sometimes
  catch harmless slang like "you killed it". Loosen it only after watching her in a
  private world first.
- Test in a private or friends-only instance before bringing her to Mommy's.

## Cost

With the default Haiku model a chatbox reply costs a small fraction of a cent. The
spend limit from step 1 is your safety net.

## Next steps

- **Voice input:** Whisper speech-to-text so she hears the world instead of you typing.
- **Voice output:** text-to-speech into Voicemeeter so she talks.
- **Avatar expressions:** drive avatar parameters over OSC from her inner state
  (e.g. curiosity → head tilt, satisfaction → smile).
