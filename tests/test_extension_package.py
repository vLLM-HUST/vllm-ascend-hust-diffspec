# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from diffspec import plugin

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "diffspec" / "manifests" / "vllm-hust-extension-v0.2.json"


def test_distribution_registers_runtime_and_manager_entry_points() -> None:
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."vllm.general_plugins"]' in project
    assert 'diffspec = "diffspec.plugin:register"' in project
    assert '[project.entry-points."vllm_hust.extension_bundles"]' in project
    assert '"org.vllm-hust.diffspec" = "diffspec.manifests"' in project
    assert 'diffspec = ["manifests/*.json"]' in project


def test_experimental_manifest_describes_real_runtime_boundaries() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == "0.2-experimental"
    assert manifest["extension_id"] == "org.vllm-hust.diffspec"
    assert manifest["kind"] == "in_process_plugin"
    assert manifest["host"] == {
        "provider": "vllm",
        "name": "vllm-ascend",
        "version_range": ">=0.23,<0.24",
    }
    assert manifest["runtime"]["isolation"] == "trusted_in_process"
    assert manifest["lifecycle_owner"] == "vllm"
    assert manifest["requires_services"] == []
    assert manifest["implementation"] == [
        {
            "type": "python_entry_point",
            "group": "vllm.general_plugins",
            "name": "diffspec",
            "status": "active",
        }
    ]
    component = manifest["components"][0]
    assert component["implementation_ref"] == "diffspec.plugin:register"
    assert component["execution_planes"] == [
        "scheduler",
        "worker",
        "native",
        "device",
    ]
    assert component["permissions"] == ["device_access"]


@pytest.mark.parametrize(
    ("manager_value", "allowed"),
    [
        (None, True),
        ("", False),
        ("org.vllm-hust.bidkv", False),
        ("org.vllm-hust.bidkv,org.vllm-hust.diffspec", True),
    ],
)
def test_manager_enabled_list_controls_registration(
    monkeypatch: pytest.MonkeyPatch,
    manager_value: str | None,
    allowed: bool,
) -> None:
    if manager_value is None:
        monkeypatch.delenv("VLLMHUST_EXT_ENABLED_BUNDLES", raising=False)
    else:
        monkeypatch.setenv("VLLMHUST_EXT_ENABLED_BUNDLES", manager_value)

    assert plugin._manager_allows_registration() is allowed


def test_disabled_manager_launch_does_not_install_runtime_patches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(plugin, "_REGISTERED", False)
    monkeypatch.setenv("VLLMHUST_EXT_ENABLED_BUNDLES", "org.vllm-hust.bidkv")

    plugin.register()

    assert plugin._REGISTERED is False
