#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

import torch
from remix_backend.utils import FunctionWrapperModule
from .base_backend import BaseBackend


class PassthroughBackend(BaseBackend):
    def backend(self, graph_module: torch.fx.GraphModule, example_inputs: list[torch.Tensor], base_backend):
        return graph_module
