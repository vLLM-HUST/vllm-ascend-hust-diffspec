# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""vLLM general-plugin entry point."""

from __future__ import annotations

_REGISTERED = False


def register() -> None:
    """Install idempotent runtime hooks in every vLLM process."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    from diffspec.vllm_patch import patch_vllm

    patch_vllm()

    # Importing vllm-ascend's runner while its platform plugin is still being
    # initialized creates a circular import through DeviceOperator. Defer the
    # device hooks until the runner module has completed normally.
    from diffspec.lazy_patch import patch_after_import

    def patch_ascend() -> None:
        from diffspec.ascend_patch import patch_vllm_ascend

        patch_vllm_ascend()

    patch_after_import("vllm_ascend.worker.model_runner_v1", patch_ascend)
