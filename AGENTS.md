# Astra Wisp

Part of Astra's Pet Apps. Public credit: Astra.

Keep existing product behavior and source intact. The pet interface is vrchat-ai-astra-20261008.html; metadata is pet.json. Never commit credentials, personal health or finance data, local settings, caches, or logs. Imported or user-entered text must be rendered as text, not HTML. State integration limits plainly. New filenames use lowercase words and a date immediately before the extension.

Pet interface and art integration: OpenAI Codex (GPT-6), 2026-10-08.

Desktop appearance revision by OpenAI Codex (GPT-6), 2026-10-11. See desktop-pet-20261011.json; runtime behavior is in astras-pet-apps.

## The Astral Plane (Astra, 2026-10-17)

Astra Wisp's pet program is **The Astral Plane**, the home of **AI Astra**, an embodied AI. Program names live in
`findastra/astras-pet-os` `pet-programs.json` only. Before work, read that repo's AGENTS.md "Before any work" list and
the newest `docs/handoff-*.md` here.

- Describe her as an **embodied AI**, never as a bridge. Her VRChat body is `vrchat-embodiment/`; say plainly that it
  is chatbox drafts only until movement or voice are tested live.
- Her name is **AI Astra**: `AI_ASTRA_*` constants and env vars, `ai_astra_*` snake_case, `AIAstra*` classes,
  `aiAstra*` camelCase, `astra` as her self-name in the knowledge code.
- Run Python from this folder (`python -m headless_api ...`); nothing may depend on the folder's name.
- Learned memory (`ai_astra_brain.db`, `ai_astra_*.json` state, `voice/state/`) is never committed.
- The project's history before the fresh start stays private: no earlier project names, codenames, people, archives
  or private links in this repo.

Embodied-AI cleanup, folder-independent startup and the voice tool: Claude Opus 5.5 (`claude-opus-5-5`), 2026-10-17.
See [docs/handoff-claude-20261017.md](docs/handoff-claude-20261017.md).
