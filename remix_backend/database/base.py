#
# Copyright IBM Corp. 2026 - 2026
# SPDX-License-Identifier: Apache-2.0
#

from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
import math

class Database(ABC):
    def __init__(self, path: str, config: "AutoConfig", *args, **kwargs):
        self.path = Path(path)
        self.config = config
        self.data = defaultdict(dict)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def latency(self, stage: "OpStage", batch_size:int, seq_len:int, is_prefill:bool):
        return math.inf