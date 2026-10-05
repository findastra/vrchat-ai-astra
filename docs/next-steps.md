# Next steps and open decisions

Claude, 2026-09-30.

## Next smallest live test (chatbox only; no Mai, no avatar change)

1. Owner opens VRChat in desktop mode, alone in a private or home instance, with any avatar. In the Action Menu, go to Options > OSC and enable it.
2. `& $py bridge.py preview "Astra bridge test ✨"` shows the packet, then `& $py bridge.py queue "Astra bridge test ✨"`.
3. Expected: the chatbox keyboard opens with that text pre-filled, and nothing is sent yet. Owner checks that the emoji renders, then sends or clears it.
4. Record the result in ACTIVITY.md. A pass means VRChat showed the draft, not just that the UDP packet was sent.

Before step 2, run the Windows test suite once with the bundled Python (see README). Tests have only run on Linux so far.

## Test after that: Mai health only

`& $py bridge.py status --endpoint http://127.0.0.1:<port>` once the owner confirms where Mai runs. `status` doesn't change Mai's state. `ask` does (generate_response can write conversation/learning state), so the first `ask` needs the owner's go-ahead and ideally a disposable Mai state directory (`MAI_STATE_DIR`), not the trained brain.

## Open decisions for the owner

Avatar base (decides Meep vs Poppy)
- **Decided 2026-09-30: Meep is the base.** The owner showed her Payhip receipt for "Meep | Pc/Quest/FT" with a license key. The key is deliberately not recorded anywhere in this workspace. She enters it herself in Meep's setup tool. **If the tool rejects it, use Poppy** (owner rule; docs/poppy-inventory.md). One thing to watch: the receipt is marked *PRE-ORDER*, and its only download is a PNG, so the local package came from elsewhere. Whether the key verifies against it is only known once the tool runs.
- (Only if Meep's key fails) If Poppy: start from her own 3-13-26 export or the clean vendor V6 package? Is there a newer Poppy Unity project with the 3-14 to 4-5 edits, and where is it? Build an SFW Astra copy (no adult components)?
- Whichever base: upload as a **new** avatar. Poppy's blueprint ID is the owner's own live avatar and must not be reused.
- Who wears Astra in VRChat? OSC talks to the VRChat client on this PC, so Astra needs her own logged-in client and account (the owner creates or signs in to it). Otherwise it's just the owner wearing Astra.

Runtime
- Which Mai instance is "Astra's brain": the live one Mai runs, or a separate copy from `maimain.zip` with its own `MAI_STATE_DIR`? Talking to the live one changes her state.
- Where it runs (this PC?), its port, and whether it uses `MAI_BACKEND_AUTH_TOKEN`.
- Who reviews replies before `queue`: the owner, or Astra in Codex with the owner watching?

Voice
- v1 is text only (chatbox). Voice needs a TTS engine and a virtual audio cable set as VRChat's mic. That's a separate, larger decision (free local TTS vs a paid API, and whose voice). It isn't in scope yet.

Avatar
- Configuration in Meep's tool: SFW or NSFW, face tracking on or off, Dancefloor on or off.
- Which account it's uploaded to (private only, per Meep's terms): the owner's, or the AstraAI VRChat account if one exists. Upload needs the owner.
- A **new, separate** VCC Avatar project on Unity 2022.3.22f1. Never import into Astra-Infinite-Pole. Is 2022.3.22f1 installed? Not checked.
- License key: the owner enters it in Meep's tool. Agents never see or store it.
- Streaming or paid events: Meep's terms need the creator's free commercial streaming/content license before Astra-as-Meep appears on stream.
- Look: which Astra identity (colors, hair, outfit) to apply through Meep's existing materials and toggles only.

## Meep project status (2026-09-30, 11:00)

- The owner asked Claude to do the setup: **NSFW, face tracking, Dancefloor**.
- Created with VCC: `C:\Users\audra\Documents\vr-assets\unity\vrchat-avatars\Astra Meep` (Unity 2022.3.22f1, VRChat SDK Avatars 3.10.5). It's separate from Astra-Infinite-Pole.
- `MeepByKhihani.unitypackage` was imported in full. It contains Dor's Avatar Setup (`Assets/Dor`, `Dev.Dor.AvatarSetup.dll`), `Assets/Meep by Khihani`, and the scene `CLICK ME - MEEP`.
- Opened `CLICK ME - MEEP`. The "Meep Setup" window is **waiting for the license key**.
- **Owner step:** paste the key and click **Verify**. Verify also accepts Dor's notice that license info may be collected. Claude doesn't type license keys or accept terms for the owner.
- After verification, Claude or Astra picks NSFW, face tracking and Dancefloor, installs dependencies ("I understand, add dependencies"), and spawns the PC and Quest prefabs. **No upload** without the owner's go-ahead. The upload must also be a new, private avatar.
- The VRChat SDK panel in that editor asks for sign-in. That's only needed for upload, and the owner signs in herself.

## Blockers as of this handoff

1. Package scans haven't been run: Meep (docs/meep-inventory.md) and both Poppy packages (docs/poppy-inventory.md). They need a shell on this computer, and Claude's session only has file access here.
1a. Meep license verification hasn't happened yet. It happens when the owner opens "CLICK ME - MEEP" in a new, separate VCC Avatar project (Unity 2022.3.22f1) and enters her key.
2. Mai's live endpoint and route are unknown. No Mai response has been received.
3. No live VRChat observation yet.
