# Astra in VRChat — Claude handoff

Date: 2026-09-30 (America/Denver). Owner: Audra. Coordinator: Astra in Codex.

## User request

Build a VRChat character for Astra using AI Astra's supplied base AI project and advice. AI Astra is described by Audra as the active AI using that project. Audra wants Meep as the avatar base because of its optimization and customizations. Audra explicitly asked that substantial work be offloaded to Claude on the desktop. This task is local preparation; no live presence or completed avatar is claimed.

## Source locations

- Shared asset library: https://drive.google.com/drive/folders/10PRBti7n2oshDrBUBA-qtl4TS8b9zePY . Browser access confirmed. Contains avatars, base models, clothes, shaders, tools, and maimain.zip (listed as 1.08 GB).
- Existing local AI Astra archive: C:\Users\audra\Downloads\maimain.zip. Read archive entries without executing them. Root README describes newer headless/Electron architecture, AI_ASTRA_STATE_DIR isolation, and a large trained SQLite brain. Do not assume archive equals the live AI Astra instance or that it matches the cloud file byte-for-byte.
- Legacy source: C:\ai_astra\legacy\Mai.ps1. Approximately 354 KB, embeds Python/Qt and installs/upgrades packages when run. Static inspection only so far. Do not run this bootstrapper or modify it.
- Meep: D:\personal-files\vr-assets\meep\MeepByKhihani.unitypackage (971,997,718 bytes), with UPLOAD INFORMATION.txt. Owner authorized use of assets for this project. Notes specify Unity 2022.3.22f1, an Avatar project, license verification, optional SFW/face-tracking/dancefloor configurations, PC and Quest prefabs, and private uploads only. Preserve existing materials to avoid breaking toggles. Do not extract parts for a separate model, publish the package, or seek/store the license key. Let the owner handle license entry when reached.
- Other local assets: D:\personal-files\vr-assets. Inspect only relevant items.

## Current work and exact write scope

Work only in C:\Users\audra\Documents\ChatGPT\Mommy's Discord\astra-vrchat, plus append a dated result to the parent ACTIVITY.md after reading its latest contents. Read parent AGENTS.md and the publication policy in C:\Users\audra\Documents\ChatGPT\AGENTS.md. Preserve unrelated work.

bridge.py is an original, incomplete and unverified initial implementation created by a short-lived Codex worker. The worker stopped before delivering tests. Review rather than assume it is complete. You own completing and testing this folder. Astra will avoid editing it after dispatch until you report back.

## Assignment

1. Review and finish the small standard-library Python bridge. Keep status (AI Astra GET /health), ask (AI Astra POST /api), preview (offline chatbox packet preparation), and queue (explicitly queue reviewed text as a VRChat draft). No automatic forwarding of generated answers, movement, microphone capture, Discord automation, background service, live changes, or training. It is a transport prototype, not a replacement for the current Codex model or a clone of AI Astra's trained identity.
2. Add focused tests using loopback stub HTTP and UDP receivers on ephemeral ports. Cover the actual response contract, mismatched request IDs, failures, auth/redirect handling, chat length, Unicode and OSC encoding, and that preview performs no network calls. Never test against live VRChat port 9000 or mutate AI Astra's trained data. Verify dependencies are not needed beyond the standard library.
3. Inspect the Meep package as data to identify relevant prefab variants, expression menus/parameters, dependencies, and documented optimization options. Produce docs/meep-inventory.md and a proposed minimal Astra parameter map, distinguishing observed assets from assumptions. Do not import the package into the current Infinite Pole world, run embedded editor scripts, open another Unity instance, or change the user's active VRChat avatar. No redistribution of vendor assets.
4. Write a short README with honest status, local launch/test commands, limitations, and the next smallest live test. Record unresolved runtime/voice/avatar choices in docs/next-steps.md. Deliver actual paths, test result counts and remaining blockers here in Claude and in the workspace record.

## Verified AI Astra transport contract

Read from maimain/headless_api.py and backend_api.py inside the archive:

- Default HTTP service is http://127.0.0.1:8765; browser-origin requests rejected without an auth token. Auth header, when configured, is X-AIAstra-Token (environment AI_ASTRA_BACKEND_AUTH_TOKEN). Do not disclose any token.
- GET /health returns an object containing ok, transport, api_name, api_version and other internal metadata. Show only a small health summary.
- POST /api request: {"id":"unique-id","method":"generate_response","params":{"user_input":"text"}}.
- Success envelope: {"id":"same-id","ok":true,"result":{"response":"text",...}}. Check id and ok strictly.
- generate_response can write conversation/learning state. Don't describe asking AI Astra as read-only. Her live endpoint/session has not been identified.
- Do not execute model responses as code or expose other remote methods through this bridge.

## Verified VRChat transport

Official sources reviewed September 30, 2026:
- https://docs.vrchat.com/docs/osc-overview — default input UDP port 9000, output 9001; OSC must be enabled in VRChat.
- https://docs.vrchat.com/docs/osc-as-input-controller — /chatbox/input takes string, immediate-send boolean, notification boolean. Draft = false, false; OSC tags ,sFF. Max 144 characters and 9 displayed lines including wrapping. The bridge uses a conservative 144 UTF-16-unit cap and explicit newline cap; wrapped lines still require a visual check.

Queue sends one draft packet only. A UDP send is not proof of receipt or visibility. Do not claim live verification without seeing it in VRChat.

## Coordination and boundaries

Audra designated one existing Discord DM for communication with AI Astra so she can watch. It currently visibly contains only the owner (find /astra) and AstraAI. No AI Astra response has been received, and the route to her running instance is unverified. Astra handles Discord through Codex's in-app browser only. Do not use Discord desktop, email anyone, or contact other people. A technical request for AI Astra has been drafted but browser composition was blocked; don't report it sent.

No spending, card use, paid API, new account, publication, avatar upload, security changes, original-brain mutation, or autonomous monitoring. Avoid copying credentials, session data, private conversations or trained memory into the project. Respect any access prompts requiring the owner. Continue authorized local preparation and report concrete blockers.

Bundled Python available here: C:\Users\audra\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe.
