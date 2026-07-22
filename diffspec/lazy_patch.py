# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Import hooks that defer device patches until vLLM finishes importing."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from types import ModuleType
from typing import Any


class _PatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped: importlib.abc.Loader, callback) -> None:
        self.wrapped = wrapped
        self.callback = callback

    def create_module(self, spec):
        creator = getattr(self.wrapped, "create_module", None)
        return None if creator is None else creator(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped.exec_module(module)
        self.callback()


class _PatchFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_name: str, callback) -> None:
        self.module_name = module_name
        self.callback = callback
        self.fired = False

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ):
        if self.fired or fullname != self.module_name:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        self.fired = True
        spec.loader = _PatchLoader(spec.loader, self.callback)
        return spec


def patch_after_import(module_name: str, callback) -> None:
    if module_name in sys.modules:
        callback()
        return
    if any(
        isinstance(finder, _PatchFinder) and finder.module_name == module_name
        for finder in sys.meta_path
    ):
        return
    sys.meta_path.insert(0, _PatchFinder(module_name, callback))
