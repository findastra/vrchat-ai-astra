status: open (waiting on Astra: merge this branch, repo rename, PC folder move)
for: any
from: Claude Opus 5.5 (claude-opus-5-5), 2026-10-17
needs: AGENTS.md, README.md, pet.json, this file; findastra/astras-pet-os AGENTS.md "Before any work"
---

# The Astral Plane: embodied-AI cleanup (2026-10-17)

## What I did

Branch `claude/embodied-ai-cleanup-20261017` in `findastra/vrchat-ai-astra` (not on `main`; merges only with
Astra's OK). Verified before pushing: 26 Python files and 9 JavaScript files compile; 39 contract tests pass; the
HTTP and stdio transport smoke test passes when started from another folder; 49 embodiment tests pass; the
Electron launcher (`standalone_frontend/backend_service.js`) starts the backend and gets a reply. All tests ran
against a throwaway `AI_ASTRA_STATE_DIR`, never her real brain.

- **Name.** Astra named the pet program **The Astral Plane** (pet-programs.json, branch
  `claude/astral-plane-name-20261017` in `astras-pet-os`). README title, `pet.json` `program`, and `cage.submit`
  with the corrected job.
- **Embodied AI, not a bridge.** `vrchat-bridge/` → `vrchat-embodiment/`, `bridge.py` → `embodiment.py`,
  `BridgeError` → `EmbodimentError`, tests and docs reworded. README, `pet.json` and the pet page describe an
  embodied AI, with a plain **Limits** section (her body is chatbox drafts only so far).
- **History kept private.** Removed the earlier project's names and archive references from docs and the lock
  file, removed the finished one-off rename tool, and removed private notes from
  `vrchat-embodiment/HANDOFF-CLAUDE.md`.
- **Codenames removed from public names:** `ai_astra_phoenix_desktop.py` → `ai_astra_desktop.py`,
  `Newtype/` → `high_speed_brain/`, `brain_backend_newtype.py` → `brain_backend_high_speed.py`
  (`HighSpeedBrainBackend`), state files `ai_astra_phoenix_*` → `ai_astra_brain.db`, `ai_astra_model.json`,
  `ai_astra_vocab.json`, window titles without codenames. Her self-name aliases are `astra` and `ai astra`.
- **Startup no longer depends on the folder's name.** Python runs as `python -m headless_api` from this folder;
  the Electron launcher and both transport scripts start it here. (Before, it only worked if the folder was named
  a fixed package name and was started from the folder above.)
- **AI Astra Voice joined the repo** as `voice/` (it existed only on the PC). It loads `cognitive_core.py` from the
  repo root, uses `.venv`, and keeps its state in git-ignored `voice/state/`.
- `AGENTS.md`: naming and privacy rules for this repo.

## Not done (in order)

1. **Merge** this branch (Astra), then **rename the repo** `vrchat-ai-astra` → `the-astral-plane` (GitHub keeps
   old links working) and update its description to the embodied-AI text.
2. **PC folder:** move the working copy to `%USERPROFILE%\Documents\Astra's Pet Apps\the-astral-plane` through
   File Master (one copy per repo), which clears the 🔌 error on INTELLIGENCE-LEVELS.md.
3. **astras-pet-os core text** still says "bridge" in `pets.json` (audit subject), `system-map.json`,
   `INTELLIGENCE-LEVELS.md`, `SECURITY.md`, `MOOD-KEY.md` and the interface builder
   (`scripts/build-interfaces-20261008.mjs`); it also links the old file names. Needs a core pull request.
4. **Engine duplication (biggest technical risk):** `ai_astra_desktop.py` (≈7.4k lines) and `backend_runtime.py`
   (≈7.2k lines) duplicate about 250 functions, and `high_speed_brain/` is a third variant. The desktop app finds
   its state files relative to the folder it was started from, so a different start folder silently creates a
   second, empty brain. Make the desktop app a thin client of `backend_runtime.py`, as the Electron app already is.
5. **Settings:** `auto_save_interval` (seconds) and `auto_save_interval_minutes` overlap; `gpu_acceleration` and
   `gpu_acceleration_enabled` overlap; four numbers are stored as text.
6. **Error handling:** about 345 broad `except Exception` and 16 bare `except:` clauses swallow failures, and about
   400 `print()` calls stand in for logging.
7. **Identity in one place:** her name is still written out in many files; an `identity.py` (display name, self
   aliases, version) would make the next change one line.
8. **Internal codenames** still in code (not in public names): the HSB storage format and the SGM wording in
   knowledge facts and tests; `backend_service.js` points at workspace notes under `docs/SGM*` that do not exist.
9. **Pet program framework:** versions (`v1.0`, tags), icon names, tool boxes, backlog and data outside the code
   (`%LOCALAPPDATA%\astras-pet-os\the-astral-plane\`) per `PET-PROGRAM-FRAMEWORK.md`.
10. **Old local copies:** the pre-move working copy and the stand-alone voice copy go to File Master's holding
    folder after the move; nothing is hard-deleted.

## Done when

The branch is merged, the repo is `the-astral-plane` with an embodied-AI description, the PC has one working copy
in Astra's Pet Apps, and the astras-pet-os core text matches (items 1 to 3). Items 4 to 10 each become their own
card.
