#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import List
from .device import Device
from remix_backend.patterns.base_pattern import BasePattern


@dataclass
class CompileOptions:
    devices: List[Device] = field(default_factory=lambda: [])
    patterns: dict[str, BasePattern] = field(default_factory=lambda: {})
    aten_op_summary_path: str | None = None  # If None, do not generate.
    device_allocation: dict = field(default_factory=lambda: {})
    dynamic_dims: dict = field(default_factory=lambda: {})
    timing: bool = False
    module_tagging : bool = False
    default_allocation:  Device = None
