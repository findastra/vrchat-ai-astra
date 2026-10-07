# Meep inventory and proposed Astra parameter map

Claude, 2026-09-30. The package was inspected as data only: nothing was imported, extracted, run, or uploaded.
Sections are tagged **Observed** (read from a file) or **Assumption / proposal** (not yet checked).

> **Status (2026-09-30):** Meep is the chosen base. The owner has a Payhip license (receipt shown in chat; the key isn't recorded here) and enters it herself. If verification fails, use Poppy (docs/poppy-inventory.md).

## Source files

| File | Size | Status |
|---|---|---|
| `D:\personal-files\vr-assets\meep\MeepByKhihani.unitypackage` | 971,997,718 bytes | **Not read yet.** It is over the 400 MB limit for moving files into Claude's cloud workspace. The terms also say not to share the package, so it should be inspected on this computer, not copied off it. |
| `D:\personal-files\vr-assets\meep\UPLOAD INFORMATION.txt` | 3,912 bytes | Read in full. |

These are the only two files in that folder.

## Observed (UPLOAD INFORMATION.txt)

Setup:
- Requires Unity **2022.3.22f1**, VRChat Creator Companion, and Android build support for that Unity version.
- Setup: create a new VCC "Unity 2022 Avatar Project", import the package, then open the scene **"CLICK ME - MEEP"**.
- That scene opens the vendor's setup tool. It asks for the **Payhip license key** and then offers to install the needed dependencies and **remove ones that interfere**. The setup tool is vendor editor code, and it changes the project's packages.
- The tool offers these configurations: **NSFW or SFW**, **face tracking on or off**, and **Dancefloor on or off**.
- After configuring, spawn the **PC prefab** and the **Quest prefab**, then upload PC first and Android second. One project covers both.
- Optimization-related notes: "Only change the materials that are already on the model, dont drag other materials on the avatar, that will break the toggles." The creator also recommends uploading before editing anything.

Terms of use that affect this project:
- No sharing, leaking or price-splitting of the package. No sharing the license key.
- **Models and edits must not be made public** (private uploads only).
- **No taking parts** for other models, and no importing into other games.
- **No profit, and no showcasing or streaming the avatar** without a license. The creator offers a **free commercial streaming/content license** on request. This matters if Astra appears in streamed or paid Mommy's events.
- The package carries Payhip Antileak, so every copy is traceable to the buyer. Keep it on this computer only.

Not stated in the file: which parameters, menus or toggles exist, the synced-bit budget, the performance rank, or the exact dependency list. Those come from the package scan below.

## Package scan (pending, read-only)

`tools/inspect_unitypackage.py` streams the gzipped tar once. It records paths, sizes, prefab and scene names, VRChat expression parameters (name, type, default, saved, synced), expression-menu controls, animator parameters and layers, and which parameter, menu and controller assets each avatar descriptor points to. It extracts nothing and runs nothing. Tested on a synthetic package (tests/test_inspect_unitypackage.py).

Run it on this computer (about 1 GB is decompressed in memory as a stream, so allow a minute or two):

```powershell
cd "C:\Users\audra\Documents\ChatGPT\Mommy's Discord\astra-vrchat"
$py = "C:\Users\audra\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py tools\inspect_unitypackage.py "D:\personal-files\vr-assets\meep\MeepByKhihani.unitypackage" --out docs\generated\meep-scan
```

This writes `docs\generated\meep-scan.md` and `.json` (names and metadata only). The report starts with a **lock assessment**. "MOSTLY NOT READABLE" means most Unity assets aren't plain text, probably because they're locked until the vendor tool verifies a license. It also lists file names that look like license or setup tooling. Locked content is never worked around; per the owner's rule, the fallback is Poppy.

Alternative that needs no package reading: once Meep is uploaded privately and worn with OSC enabled, VRChat writes the avatar's real parameter list to
`%USERPROFILE%\AppData\LocalLow\VRChat\VRChat\OSC\usr_<id>\Avatars\avtr_<id>.json` (per docs.vrchat.com/docs/osc-avatar-parameters). That file is the ground truth for OSC mapping.

## Proposed minimal Astra parameter map (assumptions)

Principles: don't add anything to Meep in v1. Drive only what already exists or what VRChat provides, keep every channel explicit and operator-triggered, and respect the 256-bit synced budget until the scan shows Meep's usage.

| Astra signal | VRChat target | Source | Status |
|---|---|---|---|
| Reviewed line of speech | `/chatbox/input` s, F, F (draft) | VRChat built-in | **Implemented** (`bridge.py queue`) |
| "Astra is composing" | `/chatbox/typing` bool | VRChat built-in (documented) | Proposed. It would be a 1-line addition, but it isn't added because the scope says no automatic behaviour |
| Mood / expression | `/avatar/parameters/<Meep face param>` | Meep's existing expression control, name **unknown** | Proposed: map a small enum (neutral, happy, surprised, sad) to whatever Meep uses, after the scan or OSC JSON shows the real name and type |
| Outfit / look | Meep's existing toggles | Meep menu | Set by hand in the expression menu. Not driven by the bridge |
| Speaking glow (optional) | New `AstraSpeaking` Bool, 1 synced bit | Would need an avatar edit | **Deferred**: it changes Meep, needs its own Avatar project, and should wait until the bit budget is known |
| Lip sync / voice | `Viseme`, `Voice` (built-in, driven by the mic) | VRChat built-in | Out of scope (no microphone or TTS in this bridge) |

Sending to a parameter that doesn't exist on the worn avatar does nothing, so parameter output should be added only after the real names are confirmed.
