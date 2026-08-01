from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from diffspec import plugin


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def reset_plugin_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin, "_REGISTERED", False)


def test_general_plugin_entry_point_is_declared() -> None:
    config = (REPO_ROOT / "pyproject.toml").read_text()

    assert '[project.entry-points."vllm.general_plugins"]' in config
    assert 'diffspec = "diffspec.plugin:register"' in config


def test_register_installs_each_hook_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    vllm_patch = ModuleType("diffspec.vllm_patch")
    vllm_patch.patch_vllm = lambda: calls.append("vllm")  # type: ignore[attr-defined]

    def patch_after_import(module_name: str, callback: object) -> None:
        calls.append((module_name, callback))

    lazy_patch = ModuleType("diffspec.lazy_patch")
    lazy_patch.patch_after_import = patch_after_import  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffspec.vllm_patch", vllm_patch)
    monkeypatch.setitem(sys.modules, "diffspec.lazy_patch", lazy_patch)

    plugin.register()
    plugin.register()

    assert calls[0] == "vllm"
    assert len(calls) == 2
    module_name, callback = calls[1]
    assert module_name == "vllm_ascend.worker.model_runner_v1"
    assert callable(callback)
    assert plugin._REGISTERED is True


def test_failed_registration_can_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise RuntimeError("incompatible vLLM hook")

    vllm_patch = ModuleType("diffspec.vllm_patch")
    vllm_patch.patch_vllm = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "diffspec.vllm_patch", vllm_patch)

    with pytest.raises(RuntimeError, match="incompatible vLLM hook"):
        plugin.register()

    assert plugin._REGISTERED is False
