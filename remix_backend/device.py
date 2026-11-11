#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import List, Callable, Any
import torch
from remix_backend.backends import BaseBackend, PassthroughBackend


@dataclass
class Device:
    name: str
    #supported_kernels: List[torch._ops.OpOverloadPacket] = field(default_factory=[])
    backend: BaseBackend = PassthroughBackend

    def __post_init__(self):
        self.backend.name = self.name

