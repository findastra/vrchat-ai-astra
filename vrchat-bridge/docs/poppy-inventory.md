# Poppy: fallback avatar base for Astra

Claude, 2026-09-30. Owner instruction (2026-09-30): **"If meep is not working due to license keys, use poppy."**
**Status: standby.** The owner has a Meep license, so Poppy is used only if Meep's setup tool rejects her key.
The folder was inspected as data only: nothing was imported, extracted, run, or uploaded. Sections are tagged **Observed** or **Assumption / proposal**.

## When Poppy is used instead of Meep

- Meep is used only if the owner has **Meep's Payhip license key** and enters it herself in Meep's own setup tool (scene "CLICK ME - MEEP").
- If she doesn't have the key, or the tool won't verify it, use Poppy. Nobody gets around Meep's check: no pulling assets out of the package and no editing the vendor tool. Meep's terms forbid taking parts, and working around the check would break them.
- The Meep scan (docs/meep-inventory.md) adds evidence. A "MOSTLY NOT READABLE" result means the content is locked behind the tool. The key is still what decides.

## Observed: `D:\personal-files\virtual-assets\poppy`

| Item | Size / date | What it is |
|---|---|---|
| `poppy\Poppy PC & Quest - Fix V6.unitypackage` | 1,430,865,793 bytes | The vendor package (PC + Quest). Not scanned yet. |
| `poppy-3-13-26\poppy-3-13-26.unitypackage` | 1,701,836,941 bytes, 2026-03-13 | **The owner's own export** of her edited Poppy. Not scanned yet. |
| `poppy-3-13-26`, `-3-14-26`, `-3-31-26`, `-4-5-26` | edit logs (.txt/.md) | Read in full. |
| `poppy-2-26-26`, `-3-14-26`, `-3-31-26`, `-4-5-26` | `.unity` scenes, 14-43 KB | Working scenes only. The 4-5 scene holds one prefab instance (Gesture Manager) and some helper objects, not the avatar prefab. It references assets in a Unity project **that isn't in this folder**. |
| `Face Gestures Pictures\` | 14 PNGs | One picture per face. Names only were used. |
| `poppy-3-14-26\Poppy ANDROID - blueprint ID.png` | image | Not opened (not needed). |

No terms-of-use or upload-instructions file for Poppy is in this folder. The vendor package may contain one; the scan's "Documents" list will show it. Until then, assume the same limits as Meep: private uploads only, and no redistributing or reusing parts.

From the edit logs (the latest is 2026-04-05):
- Unity **2022.3.22f1**, the same version Meep needs.
- Variants: "Poppy 4 VRCFT" (face tracking), "Poppy 2 NO VRCFT", and Poppy ANDROID.
- Already done: Poiyomi shaders; controller set to "Poppy (1)"; choke chain; What's Goodie lanyard (NO VRCFT variant); lewd, tongue and tonguewink gestures removed; height chart at 5'7"; sparkle hands (feet unselected, particle settings updated, an RGB fan created); hydro/blood bending; fans (4 options). Also **adult (SPS) components**.
- Still to do: Marshmellow tech, pole system, eye texture, avatar limb scaling, angel wings.
- Hand gestures to faces (VRCFT variant):
  - Right hand: Fist = Sleep, Open = In love, Point = HUH, Peace = Wink, RockNRoll = none, Gun = Embarrassed, Thumbs up = Fangs.
  - Left hand: Fist = Sleep, Open = Happy, Point, Peace and RockNRoll = none, Gun = Mad, Thumbs up = Tongue.
- Face pictures present: embarrassed, fangs, happy, huh, in-love, lewd, louchage, mad, sad, sleep, surprised, tongue, tonguewink, wink.

## Important cautions (proposal)

1. **Don't overwrite the owner's own Poppy.** Every variant in the edit logs carries the same blueprint ID: the owner's live uploaded avatar. An Astra build must be uploaded as a **new** avatar with a cleared blueprint ID. Otherwise it replaces her Poppy. The upload is the owner's step.
2. **Separate project.** Build it in a new VCC Avatar project (Unity 2022.3.22f1), never in Astra-Infinite-Pole or her Poppy project.
3. **SFW build recommended.** Astra is an AI character that people will meet at Mommy's. Leave out the adult components and keep the removed lewd/tongue faces off. This is the owner's call.
4. **Which starting point:** the owner's export (3-13-26) already has Poiyomi, sparkle hands and the height chart. The 3-14 to 4-5 edits exist only in her Unity project, which isn't in this folder. The vendor package is the clean base. Owner decision.

## Scan commands (read-only, on this computer)

```powershell
cd "C:\Users\audra\Documents\ChatGPT\Mommy's Discord\astra-vrchat"
$py = "C:\Users\audra\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py tools\inspect_unitypackage.py "D:\personal-files\virtual-assets\poppy\poppy-3-13-26\poppy-3-13-26.unitypackage" --out docs\generated\poppy-owner-3-13-26-scan
& $py tools\inspect_unitypackage.py "D:\personal-files\virtual-assets\poppy\poppy\Poppy PC & Quest - Fix V6.unitypackage" --out docs\generated\poppy-vendor-v6-scan
```

These packages are 1.4 to 1.7 GB, so each scan takes a few minutes. The output is names and parameter metadata only.

## Proposed minimal Astra parameter map on Poppy (assumptions)

| Astra signal | Target | Status |
|---|---|---|
| Reviewed line of speech | `/chatbox/input` draft | **Implemented** (bridge `queue`) |
| Mood | Poppy's face system. Proposed small set: neutral, happy, surprised, sad, embarrassed, mad, sleep | **Parameter name unknown.** Faces are gesture-driven (`GestureLeft`/`GestureRight`, VRChat built-ins). VRChat's OSC config marks some parameters as output-only. Check whether these accept input in the avatar's OSC JSON before relying on them. If they don't, a face-override Int on the Astra copy would be needed (8 synced bits, an avatar edit). |
| Look toggles (fans, sparkle hands, lanyard) | Poppy's existing menu | Set by hand, not by the bridge |
| Face tracking | VRCFT variant | Not used for an AI-driven avatar. The NO VRCFT variant is simpler |

Once an Astra copy is uploaded privately and worn with OSC on, VRChat's `...\LocalLow\VRChat\VRChat\OSC\usr_<id>\Avatars\avtr_<id>.json` gives the real parameter names and which ones accept input.
