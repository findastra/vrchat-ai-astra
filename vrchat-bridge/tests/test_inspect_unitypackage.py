"""Tests for tools/inspect_unitypackage.py using a small synthetic package.

The fixture below is invented Unity-style YAML, not vendor content.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import inspect_unitypackage as inspector  # noqa: E402

G = {name: (name * 32)[:32] for name in "abcdef0123"}

HEADER = "%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!114 &11400000\nMonoBehaviour:\n  m_ObjectHideFlags: 0\n"

PARAMS = HEADER + """  m_Name: Test Params
  m_EditorClassIdentifier:
  isEmpty: 0
  parameters:
  - name: VRCEmote
    valueType: 0
    saved: 0
    defaultValue: 0
    networkSynced: 1
  - name: Hoodie
    valueType: 2
    saved: 1
    defaultValue: 1
    networkSynced: 1
  - name: HueShift
    valueType: 1
    saved: 1
    defaultValue: 0.25
    networkSynced: 1
  - name: LocalOnly
    valueType: 2
    saved: 0
    defaultValue: 0
    networkSynced: 0
"""

MENU = HEADER + f"""  m_Name: Test Menu
  m_EditorClassIdentifier:
  controls:
  - name: Clothing
    icon: {{fileID: 0}}
    type: 103
    parameter:
      name:
    value: 1
    style: 0
    subMenu: {{fileID: 11400000, guid: {G['e']}, type: 2}}
    subParameters: []
    labels: []
  - name: Hoodie
    icon: {{fileID: 0}}
    type: 102
    parameter:
      name: Hoodie
    value: 1
    style: 0
    subMenu: {{fileID: 0}}
    subParameters: []
    labels: []
  - name: Hue
    icon: {{fileID: 0}}
    type: 203
    parameter:
      name:
    value: 1
    style: 0
    subMenu: {{fileID: 0}}
    subParameters:
    - name: HueShift
    labels: []
"""

CONTROLLER = f"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1102 &-1
AnimatorState:
  serializedVersion: 6
  m_Name: Idle State
--- !u!91 &9100000
AnimatorController:
  m_ObjectHideFlags: 0
  m_Name: Test FX
  serializedVersion: 5
  m_AnimatorParameters:
  - m_Name: Hoodie
    m_Type: 4
    m_DefaultFloat: 0
    m_DefaultInt: 0
    m_DefaultBool: 1
    m_Controller: {{fileID: 9100000}}
  - m_Name: GestureLeft
    m_Type: 3
    m_DefaultFloat: 0
    m_DefaultInt: 0
    m_DefaultBool: 0
    m_Controller: {{fileID: 9100000}}
  m_AnimatorLayers:
  - serializedVersion: 5
    m_Name: Base Layer
    m_StateMachine: {{fileID: 1107}}
    m_Mask: {{fileID: 0}}
    m_Motions: []
  - serializedVersion: 5
    m_Name: Hoodie Toggle
    m_StateMachine: {{fileID: 1108}}
    m_Mask: {{fileID: 0}}
    m_Motions: []
"""

PREFAB = f"""%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!1 &1
GameObject:
  m_Name: Test Avatar PC
--- !u!114 &2
MonoBehaviour:
  m_Name:
  baseAnimationLayers:
  - isEnabled: 0
    type: 0
    animatorController: {{fileID: 0}}
    mask: {{fileID: 0}}
    isDefault: 1
  - isEnabled: 1
    type: 5
    animatorController: {{fileID: 9100000, guid: {G['c']}, type: 2}}
    mask: {{fileID: 0}}
    isDefault: 0
  customExpressions: 1
  expressionsMenu: {{fileID: 11400000, guid: {G['b']}, type: 2}}
  expressionParameters: {{fileID: 11400000, guid: {G['a']}, type: 2}}
"""

SUBMENU = HEADER + "  m_Name: Clothing Menu\n  controls: []\n  subMenu: none\n"

PACKAGE_JSON = json.dumps({"name": "com.example.tool", "version": "1.2.3", "vpmDependencies": {"com.vrchat.avatars": ">=3.5"}})


def build_package(path: Path) -> None:
    files = [
        # (guid, leaf, bytes) -- pathname deliberately sometimes after the asset.
        (G["a"], "asset", PARAMS.encode()),
        (G["a"], "pathname", b"Assets/Test/Params.asset\n00"),
        (G["b"], "pathname", b"Assets/Test/Menu.asset"),
        (G["b"], "asset", MENU.encode()),
        (G["c"], "asset", CONTROLLER.encode()),
        (G["c"], "pathname", b"Assets/Test/Animators/FX.controller"),
        (G["d"], "asset", PREFAB.encode()),
        (G["d"], "asset.meta", b"fileFormatVersion: 2"),
        (G["d"], "preview.png", b"\x89PNG fake"),
        (G["d"], "pathname", b"Assets/Test/Test Avatar PC.prefab"),
        (G["e"], "pathname", b"Assets/Test/Clothing Menu.asset"),
        (G["e"], "asset", SUBMENU.encode()),
        (G["f"], "pathname", b"Assets/Test/Textures/Body.png"),
        (G["f"], "asset", b"\x89PNG" + b"\0" * 5000),
        (G["0"], "pathname", b"Assets/Test/Editor/SetupTool.cs"),
        (G["0"], "asset", b"// not executed by the inspector\nclass X {}"),
        (G["1"], "pathname", b"Packages/com.example.tool/package.json"),
        (G["1"], "asset", PACKAGE_JSON.encode()),
        (G["2"], "pathname", b"Assets/Test/Binary.prefab"),
        (G["2"], "asset", b"\x00\x01binary-serialized"),
        (G["3"], "pathname", b"Assets/Test/CLICK ME.unity"),
        (G["3"], "asset", b"%YAML 1.1\n--- !u!29 &1\nOcclusionCullingSettings:\n  m_Name: \n"),
    ]
    with tarfile.open(path, "w:gz") as archive:
        for guid, leaf, data in files:
            info = tarfile.TarInfo(f"{guid}/{leaf}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


class InspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.package = Path(cls.tmp.name) / "pkg" / "Test.unitypackage"
        cls.package.parent.mkdir()
        build_package(cls.package)
        with contextlib.redirect_stderr(io.StringIO()):
            cls.report = inspector.inspect(cls.package)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts_and_lists(self):
        r = self.report
        self.assertEqual(r["asset_count"], 10)
        self.assertEqual(r["prefabs"], ["Assets/Test/Binary.prefab", "Assets/Test/Test Avatar PC.prefab"])
        self.assertEqual(r["scenes"], ["Assets/Test/CLICK ME.unity"])
        self.assertEqual(r["scripts"], ["Assets/Test/Editor/SetupTool.cs"])
        self.assertEqual(r["non_text_serialized_unity_assets"], ["Assets/Test/Binary.prefab"])
        self.assertEqual(r["extensions"][".png"]["bytes"], 5004)

    def test_expression_parameters(self):
        (asset,) = self.report["expression_parameter_assets"]
        self.assertEqual(asset["path"], "Assets/Test/Params.asset")
        names = [(p["name"], p["type"], p["default"], p["saved"], p["synced"]) for p in asset["parameters"]]
        self.assertEqual(names, [
            ("VRCEmote", "Int", 0, False, True),
            ("Hoodie", "Bool", 1, True, True),
            ("HueShift", "Float", 0.25, True, True),
            ("LocalOnly", "Bool", 0, False, False),
        ])
        self.assertEqual(asset["synced_bits"], 8 + 1 + 8)

    def test_menu_controls_and_submenu_resolution(self):
        menu = next(m for m in self.report["expression_menus"] if m["path"] == "Assets/Test/Menu.asset")
        controls = {c["name"]: c for c in menu["controls"]}
        self.assertEqual(controls["Clothing"]["type"], "SubMenu")
        self.assertEqual(controls["Clothing"]["submenu"], "Assets/Test/Clothing Menu.asset")
        self.assertEqual(controls["Hoodie"]["type"], "Toggle")
        self.assertEqual(controls["Hoodie"]["parameter"], "Hoodie")
        self.assertEqual(controls["Hue"]["type"], "RadialPuppet")
        self.assertEqual(menu["puppet_sub_parameters"], ["HueShift"])

    def test_animator_controller(self):
        (controller,) = self.report["animator_controllers"]
        self.assertEqual(controller["asset_name"], "Test FX")
        self.assertEqual(controller["layers"], ["Base Layer", "Hoodie Toggle"])
        self.assertEqual(controller["parameters"], [{"name": "Hoodie", "type": "Bool"}, {"name": "GestureLeft", "type": "Int"}])

    def test_avatar_descriptor_links(self):
        (descriptor,) = self.report["avatar_descriptors"]
        self.assertEqual(descriptor["in"], "Assets/Test/Test Avatar PC.prefab")
        self.assertEqual(descriptor["parameters"], "Assets/Test/Params.asset")
        self.assertEqual(descriptor["menu"], "Assets/Test/Menu.asset")
        self.assertEqual(descriptor["layers"][1], {"layer": "FX", "controller": "Assets/Test/Animators/FX.controller",
                                                   "is_default": False})

    def test_lock_indicators(self):
        r = self.report
        self.assertEqual(r["license_or_setup_hints"], ["Assets/Test/Editor/SetupTool.cs"])
        self.assertEqual(r["non_text_serialized_count"], 1)
        self.assertEqual(r["text_serialized_count"], 6)
        self.assertTrue(inspector.lock_assessment(r).startswith("readable"))
        locked = dict(r, text_serialized_count=1, non_text_serialized_count=9)
        self.assertTrue(inspector.lock_assessment(locked).startswith("MOSTLY NOT READABLE"))

    def test_package_manifest(self):
        (pkg,) = self.report["embedded_package_manifests"]
        self.assertEqual((pkg["name"], pkg["version"]), ("com.example.tool", "1.2.3"))

    def test_report_contains_no_asset_bodies(self):
        blob = json.dumps(self.report)
        self.assertNotIn("not executed by the inspector", blob)
        self.assertNotIn("binary-serialized", blob)
        self.assertNotIn("m_StateMachine", blob)

    def test_cli_writes_outputs_and_refuses_package_folder(self):
        out = Path(self.tmp.name) / "reports" / "test"
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(inspector.main([str(self.package), "--out", str(out)]), 0)
            self.assertEqual(inspector.main([str(self.package), "--out", str(self.package.parent / "x")]), 2)
        self.assertTrue(out.with_suffix(".json").is_file())
        markdown = out.with_suffix(".md").read_text(encoding="utf-8")
        self.assertIn("| Hoodie | Bool | 1 | True | True |", markdown)
        self.assertIn("FX: Assets/Test/Animators/FX.controller", markdown)
        self.assertFalse((self.package.parent / "x.json").exists())


if __name__ == "__main__":
    unittest.main()
