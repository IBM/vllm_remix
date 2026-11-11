#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC, abstractmethod
from loguru import logger
import torch
from contextlib import nullcontext


class BaseBackend(ABC):
    def __init__(self):
        super().__init__()

    def call(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        #logger.info("Calling %s backend" % self.__class__.__name__)
        return self.backend(graph_module, example_inputs)

    @abstractmethod
    def backend(
        self,
        graph_module: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        raise NotImplementedError

    def move_to_device(self, x):
        return x.cpu()

    @abstractmethod
    def return_latency(self, profiler, meta_ctx) -> float:
        raise NotImplementedError


    def get_profile_ctx(self):
        return nullcontext()


    def to_device(self, x):
        if isinstance(x, torch.Tensor):
            return self.move_to_device(x)

        if isinstance(x, list):
            return [self.to_device(item) for item in x]

        if isinstance(x, tuple):
            return tuple(self.to_device(item) for item in x)

        if isinstance(x, dict):
            return {key: self.to_device(value) for key, value in x.items()}

        return x

