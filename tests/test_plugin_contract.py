from __future__ import annotations

import json
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


def test_vllm_hust_optimization_manifest_matches_entry_point() -> None:
    manifest = json.loads(
        (REPO_ROOT / ".vllm-hust" / "optimization.json").read_text()
    )

    assert manifest["schema_version"] == 1
    assert manifest["id"] == "diffspec"
    assert manifest["entrypoint"] == {
        "group": "vllm.general_plugins",
        "name": "diffspec",
    }
    assert manifest["parameters"]["draft_model"]["required"] is True
    assert manifest["activation"]["vllm_plugins"] == ["ascend", "diffspec"]
    speculative_config = manifest["activation"]["extra_args"][1]
    assert speculative_config["draft_context_policy"] == "diffspec"


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


@pytest.mark.parametrize(
    ("module_name", "patch_names", "entrypoint_name"),
    [
        (
            "diffspec.vllm_patch",
            ("_patch_config", "_patch_metrics", "_patch_eagle3_model"),
            "patch_vllm",
        ),
        (
            "diffspec.ascend_patch",
            (
                "_patch_forward_context",
                "_patch_rotary_cache",
                "_patch_factory",
                "_patch_attention",
                "_patch_runner",
            ),
            "patch_vllm_ascend",
        ),
    ],
)
def test_patch_state_is_retryable_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    patch_names: tuple[str, ...],
    entrypoint_name: str,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    module = __import__(module_name, fromlist=[entrypoint_name])
    monkeypatch.setattr(module, "_PATCHED", False)
    calls: list[str] = []

    def fail_once() -> None:
        calls.append("failed")
        raise RuntimeError("incompatible runtime hook")

    monkeypatch.setattr(module, patch_names[0], fail_once)
    for name in patch_names[1:]:
        monkeypatch.setattr(module, name, lambda name=name: calls.append(name))

    with pytest.raises(RuntimeError, match="incompatible runtime hook"):
        getattr(module, entrypoint_name)()

    assert module._PATCHED is False

    monkeypatch.setattr(module, patch_names[0], lambda: calls.append("retried"))
    getattr(module, entrypoint_name)()
    assert module._PATCHED is True
    assert calls.count("retried") == 1
