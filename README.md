# Mai: a framework-free statistical AI

Mai aims to be a **lifelike, coherent alternative** to GPT-style pretrained datacenter models: not a fancy autocomplete over frozen internet text, but a mind that keeps living memory, appraises situations, chooses goals, and learns from outcomes. It is built from persistent n-gram statistics, word associations, semantic clustering, symbolic reasoning, graph reasoning, analog attention, adaptive memory, and a sparse Norn-style cognitive controller. The core runtime does **not** import PyTorch, TensorFlow, Keras, or JAX. An optional NumPy MLP is lazy and excluded from predictions until it has received real training; random weights are never blended into answers. Optional CUDA/OpenCL support accelerates bulk pattern processing and is not required for response generation.

This is still a research prototype. The useful engineering direction remains measurable online learning in service of that identity: every response should be attributable to stored evidence, steered by an explicit goal/strategy when needed, evaluated against repeatable prompts, and accepted into durable memory only when it stays clean and does not regress held-out behavior.

## Run it

On Windows, double-click `Run_Mai.bat`. The safety-hardened launcher only validates the existing Python and Electron runtimes and starts the Electron interface. It does not run PowerShell, install packages, or invoke the disabled legacy updater.

For a clean setup:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
cd standalone_frontend
npm ci
cd ..
Run_Mai.bat
```

If the full interface looks wrong, launch `Launch_Mai_Test.cmd`. It opens a deliberately small diagnostic client with backend status, one prompt, one response, cognitive-control fields (goal, action, directive, hygiene, realization mode, continuity, episode), and raw request details. By default it uses a temporary state directory so diagnostic messages do not alter the trained brain; set `MAI_TEST_USE_LIVE_STATE=1` before launching when you explicitly want to test the live state.

Headless commands run from the directory *above* `maimain`:

```powershell
py -3 -m maimain.headless_api status
py -3 -m maimain.headless_api generate "What have you learned about Mai?"
py -3 -m maimain.headless_api train "D:\path\to\training"
py -3 -m maimain.headless_api serve-http --port 8765
```

Training coalesces repeated observations into large sorted upserts, accumulates evidence instead of replacing it, and fingerprints completed files in `training_history`. Re-running the same corpus skips it unless its normalized content changes.

The Electron client gives each backend process a random session secret. Manual loopback HTTP sessions remain available without a secret for command-line clients, but browser-origin requests are rejected. Binding to a non-loopback interface requires `--auth-token` or `MAI_BACKEND_AUTH_TOKEN`.

## Verify changes safely

The runtime stores learned state beside the source code (`mai_phoenix_brain.db` and `mai_*.json`). Set `MAI_STATE_DIR` to keep state elsewhere. Desktop smoke tests automatically use and remove a temporary state directory, so they never train the production brain.

```powershell
py -3 -m unittest discover -s tests -v
py -3 evaluation_harness.py --suite symbolic --label local_check --require-pass --baseline evaluation_baseline.json
py -3 evaluation_harness.py --suite generative --label generative_check --require-pass --baseline evaluation_baseline_generative.json
py -3 evaluation_harness.py --suite gate --label gate_check --require-pass
cd standalone_frontend
npm run smoke -- all
```

The Python suite checks transport contracts, strict JSON output, settings validation, evidence accumulation, symbolic reasoning, cognitive scoring biases, gated learning accept/reject paths, training fingerprints, bounded bonuses, learning hygiene gates, and the framework-free architecture.

There are three held-out evaluation suites:

- **symbolic** (`evaluation_prompts.json` / `evaluation_baseline.json`) — high-confidence compositional and identity fixtures that should keep passing.
- **generative** (`evaluation_prompts_generative.json` / `evaluation_baseline_generative.json`) — paraphrases and novel prompts expected to miss symbolic shortcuts; the harness tracks `symbolic_hit_rate`, `cognitive_action_presence_rate`, and `unexpected_symbolic_count`.
- **gate** (`evaluation_prompts_gate.json`) — compact generative probes used by intentional gated learning before durable memory writes.

Both full suites run without learning from their answers, emit an explicit pass/fail verdict, and can compare against a prior baseline with `--baseline`. The smoke suite launches isolated backend and Electron sessions and verifies that the production database is untouched.

### Gated online learning

Live chat still uses cheap hygiene gates. Intentional durable learning can be proposed through a clone trial:

```powershell
py -3 -m maimain.headless_api gated-learn --user "How could curiosity change answer length?" --response "Curiosity can lengthen exploration while confidence is low, then shorten answers once a coherent relation is found." --dry-run
py -3 -m maimain.headless_api gated-learn --user "..." --response "..." 
```

The flow is: hygiene → learn on an in-memory clone → re-run the gate suite → compare pre/post (and the generative baseline when present) → commit to the live brain only if accepted. Use `--dry-run` to inspect the decision without writing.

## Architecture and research priorities

The response path is deliberately heterogeneous:

1. SQLite/HSB stores context chains, associations, facts, evidence, and memory.
2. A fast symbolic layer handles high-confidence compositional rules, constraints, evidence standards, core causal facts, and architecture introspection.
3. Statistical retrieval and graph spreading activation propose continuations and reasoning paths for learned material.
4. Critic, confidence, anti-loop, replay, and adaptive-learning features score or reinforce outcomes.
5. A sparse Norn-style cognitive core appraises each event, updates bounded chemistry-like control signals, selects a goal and response strategy, biases evidence selection, and can preempt scaffolded continuation when clarification, repair, weak recall, or hypothesis exploration is warranted.
6. Atomic sidecar saves protect vocabulary, optional-model metadata, attention, context, semantic, and settings state from partial writes.

The cognitive core stores scalar state, goals, action values, relations, and a capped 1,200-event history in the existing SQLite brain. Action values are capped at 10,000 rows and learned relations at 25,000 rows. Appraisal itself stays in memory; one small transaction is written only after a response completes. Its current goal, strategy, state, and reward are exposed in response diagnostics, the diagnostic test client, the Electron provenance panel, and the feature-runtime snapshot.

### Compact vocab-ID storage

Repeated TEXT n-gram contexts inflate SQLite. Mai now supports an optional integer lexicon + `dynamic_word_chain_ids` table:

- Setting `vocab_id_storage_mode`: `off` (default), `dual` (write TEXT + IDs, prefer ID reads), or `ids` (ID writes only; for fresh brains after migration).
- Changing the mode requires a backend restart.
- Incremental migration API: `migrate_vocab_id_storage(batch_size, offset_rowid)` / inspect with `get_vocab_id_storage_snapshot`.
- Headless batch migrator:

```powershell
py -3 headless_api.py migrate-vocab --batch-size 5000 --max-batches 10
```

The Norn cognitive core now also **reads** persisted `cognitive_goals` progress/priority when choosing the next goal, so successful strategies can bias later behavior instead of being write-only bookkeeping.

### Episode memory and goal continuity

Multi-turn coherence uses two light mechanisms:

- **Episode recall** — before each appraisal, Mai scores recent `conversation_memory` turns by lexical overlap, recency, and quality. A strong match raises `memory_salience` and is attached to the cognitive trace as `episode_memory`. When the selected action is `recall_episode`, generation can preempt with the stored user/bot excerpt (labeled as memory, not invention) instead of a scaffolded miss.
- **Goal continuity** — after a turn completes, an unfinished prior goal (progress below ~0.78) gets a continuity bias on the next appraisal unless a correction, social, or creative interrupt arrives. The trace exposes `goal_continuity` and a reason string when that bias wins.

Both fields appear in the Electron provenance panel and the diagnostic test client.

### Deferred follow-on work

- Finish large-brain vocab-ID migration in controlled batches, then optionally switch `vocab_id_storage_mode` to `ids` and drop redundant TEXT chain pages once verified.
- Keep new work on the headless/Electron path; avoid growing the legacy `mai_phoenix_desktop.py` monolith.
