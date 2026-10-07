"""Read-only inventory of a .unitypackage (a gzipped tar), standard library only.

It streams the archive once and never extracts files, runs code, or copies
asset contents. It records only structure: asset paths and sizes, prefab and
scene names, and the metadata that matters for an avatar-parameter map
(VRChat expression parameters, expression-menu controls, animator parameters
and layer names, and which parameter/menu/controller assets each avatar
descriptor references).

Usage (on the computer that holds the package; nothing is written next to it):
    python tools/inspect_unitypackage.py "D:\\path\\Package.unitypackage" --out docs/generated/meep
It writes <out>.json and <out>.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

MAX_YAML_BYTES = 64 * 1024 * 1024  # full avatar prefabs/FX controllers can be tens of MB
MAX_JSON_BYTES = 64 * 1024
GUID_RE = re.compile(r"^[0-9a-f]{32}$")
UNITY_SERIALIZED = {".prefab", ".controller", ".asset", ".unity", ".mat", ".anim", ".overridecontroller", ".mask"}
LICENSE_HINT_RE = re.compile(r"licen[sc]|verif|payhip|gumroad|jinxxy|protect|encrypt|decrypt|unlock|setup", re.I)
REF_RE = r"\{fileID: -?\d+, guid: ([0-9a-f]{32}), type: \d+\}"

VRC_VALUE_TYPES = {0: "Int", 1: "Float", 2: "Bool"}
VRC_BIT_COST = {"Int": 8, "Float": 8, "Bool": 1}
VRC_CONTROL_TYPES = {
    101: "Button", 102: "Toggle", 103: "SubMenu", 201: "TwoAxisPuppet", 202: "FourAxisPuppet", 203: "RadialPuppet",
}
ANIMATOR_PARAM_TYPES = {1: "Float", 3: "Int", 4: "Bool", 9: "Trigger"}
# VRChat animator layer slots in the avatar descriptor's baseAnimationLayers/specialAnimationLayers.
VRC_LAYER_TYPES = {0: "Base", 1: "Deprecated0", 2: "Additive", 3: "Gesture", 4: "Action", 5: "FX",
                   6: "Sitting", 7: "TPose", 8: "IKPose"}


def _scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(value[0]) and len(value) >= 2:
        return value[1:-1]
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _list_block(text: str, key: str, indent: str) -> list[dict[str, Any]]:
    """Parse a simple YAML list of mappings under ``<indent><key>:``.

    Unity's force-text YAML is regular enough for this: items start with
    ``<indent>- field: value`` and continue with deeper-indented ``field: value``.
    Nested mappings are flattened as ``parent.child``.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == f"{indent}{key}:")
    except StopIteration:
        inline = re.search(rf"^{re.escape(indent)}{re.escape(key)}: \[\]", text, re.M)
        return [] if inline else []
    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    parent: str | None = None
    parent_indent = -1
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        depth = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if depth < len(indent) or (depth == len(indent) and not stripped.startswith("- ")):
            break
        if depth == len(indent) and stripped.startswith("- "):
            current = {}
            items.append(current)
            parent, parent_indent = None, -1
            stripped = stripped[2:]
            depth += 2
        if current is None or ":" not in stripped:
            continue
        field, _, value = stripped.partition(":")
        if parent is not None and depth <= parent_indent:
            parent = None
        if value.strip() == "":
            parent, parent_indent = field, depth
            continue
        current[f"{parent}.{field}" if parent else field] = _scalar(value)
    return items


def _name(text: str) -> str | None:
    match = re.search(r"^  m_Name: (.*)$", text, re.M)
    return str(_scalar(match.group(1))) if match else None


def extract_yaml(text: str) -> dict[str, Any]:
    """Pull only the avatar-relevant metadata out of one Unity YAML asset."""
    found: dict[str, Any] = {}

    if "\n  parameters:" in text and "valueType:" in text:
        params = []
        for item in _list_block(text, "parameters", "  "):
            if "name" not in item:
                continue
            value_type = VRC_VALUE_TYPES.get(item.get("valueType"), f"unknown({item.get('valueType')})")
            params.append({
                "name": str(item["name"]),
                "type": value_type,
                "default": item.get("defaultValue"),
                "saved": bool(item.get("saved", 0)),
                "synced": bool(item.get("networkSynced", 1)),
            })
        found["expression_parameters"] = {"asset_name": _name(text), "parameters": params}

    if "\n  controls:" in text and "subMenu:" in text:
        controls = []
        for item in _list_block(text, "controls", "  "):
            if "name" not in item:
                continue
            control = {
                "name": str(item["name"]),
                "type": VRC_CONTROL_TYPES.get(item.get("type"), f"unknown({item.get('type')})"),
                "parameter": str(item.get("parameter.name", "") or ""),
                "value": item.get("value"),
            }
            sub = re.search(r"guid: ([0-9a-f]{32})", str(item.get("subMenu", "")))
            if sub:
                control["submenu_guid"] = sub.group(1)
            controls.append(control)
        # subParameters (puppet axes) are a nested list; collect their names separately.
        sub_names = [
            name
            for block in re.findall(r"^ +subParameters:\n((?: +- name: .*\n)+)", text, re.M)
            for name in re.findall(r"- name: (.*)", block)
        ]
        found["expressions_menu"] = {"asset_name": _name(text), "controls": controls,
                                     "puppet_sub_parameters": sorted({str(_scalar(n)) for n in sub_names} - {""})}

    if re.search(r"^AnimatorController:", text, re.M):
        params = [
            {"name": str(item["m_Name"]), "type": ANIMATOR_PARAM_TYPES.get(item.get("m_Type"), str(item.get("m_Type")))}
            for item in _list_block(text, "m_AnimatorParameters", "  ")
            if "m_Name" in item
        ]
        layers = [str(item["m_Name"]) for item in _list_block(text, "m_AnimatorLayers", "  ") if "m_Name" in item]
        controller_doc = text[re.search(r"^AnimatorController:", text, re.M).start():]
        found["animator_controller"] = {"asset_name": _name(controller_doc), "parameters": params, "layers": layers}

    if "expressionParameters:" in text and "expressionsMenu:" in text:
        descriptor: dict[str, Any] = {}
        match = re.search(r"expressionParameters: " + REF_RE, text)
        if match:
            descriptor["parameters_guid"] = match.group(1)
        match = re.search(r"expressionsMenu: " + REF_RE, text)
        if match:
            descriptor["menu_guid"] = match.group(1)
        layers = []
        for layer_match in re.finditer(
            r"- isEnabled: \d\s+type: (\d+)\s+animatorController: (?:\{fileID: 0\}|" + REF_RE + r")\s+mask: [^\n]*\s+isDefault: (\d)",
            text,
        ):
            layers.append({
                "layer": VRC_LAYER_TYPES.get(int(layer_match.group(1)), layer_match.group(1)),
                "controller_guid": layer_match.group(2),
                "is_default": layer_match.group(3) == "1",
            })
        descriptor["layers"] = layers
        found.setdefault("avatar_descriptors", []).append(descriptor)

    return found


def inspect(package: Path, progress: bool = False) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    started = time.monotonic()
    members = 0
    with tarfile.open(package, mode="r|gz") as archive:
        for member in archive:
            members += 1
            if progress and members % 5000 == 0:
                print(f"  ...{members} tar entries, {time.monotonic() - started:.0f}s", file=sys.stderr)
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name.lstrip("./")).parts
            if len(parts) != 2 or not GUID_RE.match(parts[0]):
                entries["_unexpected"].setdefault("names", []).append(member.name)
                continue
            guid, leaf = parts
            record = entries[guid]
            if leaf == "pathname":
                handle = archive.extractfile(member)
                raw = handle.read(4096) if handle else b""
                record["path"] = raw.decode("utf-8", "replace").splitlines()[0].strip() if raw else ""
            elif leaf == "asset":
                record["size"] = member.size
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                head = handle.read(16)
                if head.startswith(b"%YAML"):
                    record["yaml"] = True
                    if member.size > MAX_YAML_BYTES:
                        record["yaml_too_large"] = True
                    else:
                        text = (head + handle.read()).decode("utf-8", "replace")
                        extracted = extract_yaml(text)
                        if extracted:
                            record["extracted"] = extracted
                elif head.lstrip().startswith(b"{") and member.size <= MAX_JSON_BYTES:
                    try:
                        document = json.loads((head + handle.read()).decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        document = None
                    if isinstance(document, dict) and "name" in document and "version" in document:
                        record["package_json"] = {
                            key: document.get(key)
                            for key in ("name", "version", "displayName", "unity", "dependencies", "vpmDependencies")
                            if key in document
                        }
            elif leaf == "preview.png":
                record["has_preview"] = True
    return summarize(entries, package, members, time.monotonic() - started)


def summarize(entries: dict[str, dict[str, Any]], package: Path, members: int, seconds: float) -> dict[str, Any]:
    unexpected = entries.pop("_unexpected", {}).get("names", [])
    by_guid = {guid: rec for guid, rec in entries.items() if "path" in rec}
    path_of = {guid: rec["path"] for guid, rec in by_guid.items()}
    assets = sorted(by_guid.values(), key=lambda rec: rec["path"].lower())

    ext_counts: collections.Counter[str] = collections.Counter()
    ext_bytes: collections.Counter[str] = collections.Counter()
    folder_bytes: collections.Counter[str] = collections.Counter()
    non_yaml_unity = []
    yaml_unity = 0
    for rec in assets:
        path = PurePosixPath(rec["path"])
        ext = path.suffix.lower() or ("(folder)" if "size" not in rec else "(none)")
        ext_counts[ext] += 1
        ext_bytes[ext] += rec.get("size", 0)
        folder = "/".join(path.parts[:3]) if len(path.parts) > 3 else "/".join(path.parts[:-1])
        folder_bytes[folder] += rec.get("size", 0)
        if ext in UNITY_SERIALIZED and "size" in rec:
            if rec.get("yaml"):
                yaml_unity += 1
            else:
                non_yaml_unity.append(rec["path"])

    def pick(ext: str) -> list[str]:
        return [rec["path"] for rec in assets if rec["path"].lower().endswith(ext)]

    def resolve(guid: str | None) -> str | None:
        return path_of.get(guid, f"(guid {guid} not in package)") if guid else None

    parameter_assets, menus, controllers, descriptors, packages = [], [], [], [], []
    for rec in assets:
        extracted = rec.get("extracted", {})
        if "expression_parameters" in extracted:
            params = extracted["expression_parameters"]["parameters"]
            bits = sum(VRC_BIT_COST.get(p["type"], 0) for p in params if p["synced"])
            parameter_assets.append({"path": rec["path"], "synced_bits": bits, "parameters": params})
        if "expressions_menu" in extracted:
            menu = dict(extracted["expressions_menu"])
            for control in menu["controls"]:
                if "submenu_guid" in control:
                    control["submenu"] = resolve(control.pop("submenu_guid"))
            menus.append({"path": rec["path"], **menu})
        if "animator_controller" in extracted:
            controllers.append({"path": rec["path"], **extracted["animator_controller"]})
        for descriptor in extracted.get("avatar_descriptors", []):
            descriptors.append({
                "in": rec["path"],
                "parameters": resolve(descriptor.get("parameters_guid")),
                "menu": resolve(descriptor.get("menu_guid")),
                "layers": [
                    {"layer": layer["layer"], "controller": resolve(layer["controller_guid"]) if layer["controller_guid"]
                     else "(none)", "is_default": layer["is_default"]}
                    for layer in descriptor.get("layers", [])
                ],
            })
        if "package_json" in rec:
            packages.append({"path": rec["path"], **rec["package_json"]})

    return {
        "package": package.name,
        "package_bytes": package.stat().st_size,
        "tar_entries": members,
        "scan_seconds": round(seconds, 1),
        "asset_count": len(assets),
        "orphan_guid_entries": len(entries) - len(by_guid),
        "unexpected_tar_names": unexpected[:50],
        "extensions": {ext: {"count": ext_counts[ext], "bytes": ext_bytes[ext]} for ext in sorted(ext_counts)},
        "largest_folders": [{"folder": f, "bytes": b} for f, b in folder_bytes.most_common(25)],
        "scenes": pick(".unity"),
        "prefabs": pick(".prefab"),
        "shaders": pick(".shader"),
        "scripts": pick(".cs"),
        "assemblies": pick(".dll"),
        "documents": [rec["path"] for rec in assets
                      if PurePosixPath(rec["path"]).suffix.lower() in {".txt", ".md", ".pdf", ".url", ".html"}],
        "non_text_serialized_unity_assets": non_yaml_unity[:200],
        "non_text_serialized_count": len(non_yaml_unity),
        "text_serialized_count": yaml_unity,
        "yaml_too_large_to_parse": [rec["path"] for rec in assets if rec.get("yaml_too_large")],
        "license_or_setup_hints": [rec["path"] for rec in assets
                                   if PurePosixPath(rec["path"]).suffix.lower() in {".cs", ".dll", ".unity", ".txt", ".pdf"}
                                   and LICENSE_HINT_RE.search(PurePosixPath(rec["path"]).name)],
        "embedded_package_manifests": packages,
        "expression_parameter_assets": parameter_assets,
        "expression_menus": menus,
        "animator_controllers": controllers,
        "avatar_descriptors": descriptors,
    }


def lock_assessment(report: dict[str, Any]) -> str:
    """A cautious reading of whether the package's content is usable as plain Unity assets."""
    text, other = report["text_serialized_count"], report["non_text_serialized_count"]
    if text + other == 0:
        return "no Unity serialized assets found"
    if other > text:
        return ("MOSTLY NOT READABLE: most Unity assets aren't text YAML. They're probably locked or encrypted until "
                "the vendor's setup/license tool runs (or the package uses binary serialization).")
    if report["expression_parameter_assets"] or report["avatar_descriptors"]:
        return "readable: avatar parameters/descriptors were found as plain text"
    return "mostly readable text, but no avatar parameters or descriptors were found yet"


def _mb(value: int) -> str:
    return f"{value / 1_048_576:,.1f} MB"


def to_markdown(report: dict[str, Any]) -> str:
    out = [
        f"# Generated inventory: {report['package']}",
        "",
        "Generated by tools/inspect_unitypackage.py (read-only stream; no files extracted, no code run).",
        "Contains names and parameter metadata only, no vendor asset contents.",
        "",
        f"- Package size: {_mb(report['package_bytes'])}; tar entries: {report['tar_entries']}; "
        f"assets: {report['asset_count']}; scan time: {report['scan_seconds']} s",
        f"- Unity serialized assets readable as text YAML: {report['text_serialized_count']}; "
        f"not text (binary, encrypted or locked): {report['non_text_serialized_count']}",
        f"- Lock assessment: {lock_assessment(report)}",
        f"- Text assets too large to parse (> {MAX_YAML_BYTES // 1_048_576} MB): "
        f"{', '.join(report['yaml_too_large_to_parse']) or 'none'}",
        "",
        "## License / setup hints (file names only)",
        "",
        *([f"- {path}" for path in report["license_or_setup_hints"]] or ["- none"]),
        "",
        "## File types",
        "",
        "| Extension | Count | Size |",
        "|---|---:|---:|",
    ]
    for ext, info in sorted(report["extensions"].items(), key=lambda kv: -kv[1]["bytes"]):
        out.append(f"| {ext} | {info['count']} | {_mb(info['bytes'])} |")
    out += ["", "## Largest folders", ""]
    out += [f"- {item['folder']} ({_mb(item['bytes'])})" for item in report["largest_folders"]]
    for title, key in (("Scenes", "scenes"), ("Prefabs", "prefabs"), ("Documents", "documents"),
                       ("Editor/runtime scripts (names only, not run)", "scripts"), ("Assemblies", "assemblies"),
                       ("Shaders", "shaders")):
        values = report[key]
        out += ["", f"## {title} ({len(values)})", ""]
        out += [f"- {value}" for value in values[:300]] or ["- none"]
        if len(values) > 300:
            out.append(f"- ...and {len(values) - 300} more (see JSON)")
    out += ["", "## Embedded package manifests", ""]
    for pkg in report["embedded_package_manifests"] or []:
        out.append(f"- {pkg.get('name')} {pkg.get('version')} ({pkg['path']})")
    if not report["embedded_package_manifests"]:
        out.append("- none found")
    out += ["", "## Avatar descriptors (where prefabs/scenes point)", ""]
    for d in report["avatar_descriptors"] or []:
        out.append(f"- **{d['in']}**: parameters = {d['parameters']}; menu = {d['menu']}")
        for layer in d["layers"]:
            out.append(f"  - {layer['layer']}: {layer['controller']}{' (default)' if layer['is_default'] else ''}")
    if not report["avatar_descriptors"]:
        out.append("- none found in text assets")
    out += ["", "## Expression parameter assets", ""]
    for asset in report["expression_parameter_assets"] or []:
        out += [f"### {asset['path']} ({asset['synced_bits']} synced bits of 256)", "",
                "| Name | Type | Default | Saved | Synced |", "|---|---|---|---|---|"]
        out += [f"| {p['name']} | {p['type']} | {p['default']} | {p['saved']} | {p['synced']} |" for p in asset["parameters"]]
        out.append("")
    if not report["expression_parameter_assets"]:
        out.append("- none found in text assets")
    out += ["", "## Expression menus", ""]
    for menu in report["expression_menus"] or []:
        out.append(f"### {menu['path']}")
        out.append("")
        for control in menu["controls"]:
            extra = f" -> {control['submenu']}" if control.get("submenu") else ""
            out.append(f"- {control['name']} [{control['type']}] param `{control['parameter']}` = {control['value']}{extra}")
        if menu.get("puppet_sub_parameters"):
            out.append(f"- puppet axes: {', '.join(menu['puppet_sub_parameters'])}")
        out.append("")
    if not report["expression_menus"]:
        out.append("- none found in text assets")
    out += ["", "## Animator controllers", ""]
    for controller in report["animator_controllers"] or []:
        out.append(f"### {controller['path']}")
        out.append("")
        out.append(f"- Layers ({len(controller['layers'])}): {', '.join(controller['layers']) or 'none'}")
        out.append("- Parameters: " + (", ".join(f"{p['name']} ({p['type']})" for p in controller["parameters"]) or "none"))
        out.append("")
    if not report["animator_controllers"]:
        out.append("- none found in text assets")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Output path prefix (writes .json and .md)")
    args = parser.parse_args(argv)
    if not args.package.is_file():
        print(f"Error: {args.package} is not a file.", file=sys.stderr)
        return 2
    out_json, out_md = args.out.with_suffix(".json"), args.out.with_suffix(".md")
    package_dir = args.package.resolve().parent
    if package_dir in out_json.resolve().parents:
        print("Error: write the report outside the package's folder.", file=sys.stderr)
        return 2
    print(f"Scanning {args.package.name} (read-only)...", file=sys.stderr)
    report = inspect(args.package, progress=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(to_markdown(report), encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}: {report['asset_count']} assets, "
          f"{len(report['expression_parameter_assets'])} parameter assets, {len(report['expression_menus'])} menus, "
          f"{len(report['avatar_descriptors'])} avatar descriptors.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
