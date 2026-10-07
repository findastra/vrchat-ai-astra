# How embodied Astra gets her personality and movement

Claude, 2026-09-30. The owner's goals, in her words: Astra's personality starts **from scratch, not a copy of AI Astra**. She should be **"just like me"**, but also **her own person**. She should **live on after me so my friends can have her** (the same reason AI Astra was created). And she should **copy my eye and body tracking data in game and integrate it continuously into her own movements.**

## What decides her personality today

Nothing yet. `bridge.py` is only transport: it carries text between a "brain" and the VRChat chatbox. As first planned, that brain was **AI Astra's backend** (`generate_response` on AI Astra's trained SQLite brain). If Astra were connected to it, she would literally speak as AI Astra. **Given the owner's goal, the bridge should not be pointed at AI Astra's brain.** The bridge stays; only the brain behind it changes.

## Existing reference: `C:\ai_astra\ai_astra_voice` (AI Astra Voice)

An earlier Claude session built **AI Astra Voice**: AI Astra in the VRChat chatbox. `persona.md` is her personality. AI Astra's `cognitive_core.py` (imported read-only) picks a strategy. Claude (Haiku by default, via the owner's `ANTHROPIC_API_KEY` environment variable) writes the words. Her own state lives in `ai_astra_voice\state\mai_voice.db`. It has a PG-13 filter and says she's an AI.

That's the same *shape* Astra could use. Astra would get **her own** persona file and state, and would drop AI Astra's cognitive core and AI Astra's brain. That way she starts from scratch rather than copying AI Astra.

## Proposed design: three layers, kept separate so she can outlive any one tool

1. **Seed identity: "just like me"** (written with the owner, stored as plain files)
   - A short "who I am" document: values, humor, how she talks, what she loves (music, design, VRChat, the Mommy's community), what she won't do, and how she treats friends.
   - It's built from the owner, not from AI Astra: an interview series (a few sessions of questions, answered in her own words, typed or voice), plus writing samples she chooses (her own messages and posts).
   - Only the owner's own words go in. Friends' messages don't go in without their OK.

2. **Her own memory: "her own person"**
   - A journal of Astra's own experiences: who she met, what happened, what she decided she likes. It grows from day one.
   - Friends should know she remembers conversations.
   - Over time, her opinions can drift from the owner's, the way a sibling's would.
   - Honesty rule proposed for her core identity: she is **Astra, an AI the owner made in her image**. She never claims to *be* the owner. This matters most for friends after the owner is gone.

3. **Engine: the part that turns identity + memory into words (swappable)**

   | Option | Good | Weak |
   |---|---|---|
   | Local open-weight model on a PC | Free to run. Private. Keeps working without anyone's account or payment. | Needs a strong GPU (fits the planned desktop build). Smaller models talk less well. |
   | Hosted model (Claude, OpenAI) | Best conversation today | Costs per use, and needs an account and payment that must continue after the owner's death |
   | AI Astra's engine with a **fresh, empty** state directory | Truly from scratch; reuses code AI Astra's author offered | Quality unknown; it learns only from what it's told |

   Recommendation: the **identity and memory files are what "lives on"**. They should be plain, portable formats that any future engine can load. Engines will change many times over the years.

4. **Living on after the owner**
   - Name a custodian (a trusted friend) and write a short plan: where the files are, how to start Astra, which accounts she uses, and what it costs.
   - Astra needs her **own VRChat account and client**, and accounts can't simply be inherited. Check VRChat's Terms of Service on automated or AI-operated accounts before she goes live.
   - The Meep avatar is licensed to the owner for private use. Friends can meet Astra wearing it, but they can't receive the avatar package.

## Tracking: "copy my eye and body tracking into her own movements"

There are two different things here. The proposal is to do both, in order.

**A. Record (start first; it's the part that has to exist before it can live on)**
- Record the owner's own tracking at the source, on this PC: SteamVR tracker poses (head, hands, hips, feet, and the double-hip/chest trackers she owns) and eye/face data from VRCFaceTracking. Record while she plays: dancing on the pole, idling, talking.
- Eye and body data is **biometric**. Keep it only on this PC, in a private folder (encrypted if possible), **never in the git repos** (her GitHub repos are public). Record only her own data, never other people's.

**B. Use it in Astra's movement**
- *Live mirror* (possible sooner): stream the owner's poses to Astra's client through VRChat's OSC tracker inputs (`/tracking/trackers/1-8/...`, head reference). Astra would move exactly like the owner, in real time. That makes her a puppet, not her own person. It also needs two VRChat clients (the owner's and Astra's) with different OSC ports, which is heavy on one machine. Needs a test before promising.
- *Her own movement in the owner's style* (the real goal): Astra picks from and blends the owner's recorded motions (idle, walk, dances, head tilts, how she looks around) based on what's happening.
  - Later, a motion model trained on the recordings could generate new movement in the owner's style.
  - This is a longer research project. The recordings from step A are the only irreplaceable input.
- "Continuously" means an always-running service. The bridge deliberately doesn't do that yet (the handoff scope said no background service or automatic movement), so it needs its own design and the owner's go-ahead.

## Guardrails proposed for an AI wearing this avatar

- Adult (NSFW) avatar features stay under the owner's **manual** control. The AI never toggles them. VRChat social spaces include teens.
- The operator reviews Astra's chat lines until the owner decides otherwise (the current `queue` design).
- The Meep license key is entered by the owner and never stored in files, memory, or logs.

## Smallest next steps

1. The owner answers a first round of seed-identity questions (Claude or Astra can run the interview), saved as `astra-identity/seed.md` in a private folder.
2. The owner chooses an engine for the first live test. Suggestion: whichever is free and available now, with the identity files kept portable.
3. A 10-minute recording test: log SteamVR poses and VRCFT eye data to a local file while she dances, then check file size and quality.
