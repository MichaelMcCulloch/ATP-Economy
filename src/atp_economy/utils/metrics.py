# src/atp_economy/utils/metrics.py
from __future__ import annotations
from typing import Dict, List, Iterable, Optional
import numpy as np


class MetricsRecorder:
    """
    Memory-lean recorder:
    - Keeps only selected small vectors (already numpy) per step.
    - Ring buffer with optional stride to downsample.
    """

    def __init__(
        self, keys: Iterable[str], maxlen: Optional[int] = None, stride: int = 1
    ):
        self.keys = list(keys)
        self.maxlen = maxlen
        self.stride = max(1, stride)
        self._step = 0
        self._store: Dict[str, List[np.ndarray]] = {k: [] for k in self.keys}

    def record(self, metrics: Dict[str, np.ndarray]):
        self._step += 1
        if (self._step - 1) % self.stride != 0:
            return
        for k in self.keys:
            v = metrics.get(k, None)
            if v is None:
                continue
            self._store[k].append(v.copy())
            if self.maxlen is not None and len(self._store[k]) > self.maxlen:
                # pop front (ring buffer)
                self._store[k].pop(0)

    def as_arrays(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for k, seq in self._store.items():
            out[k] = np.stack(seq, axis=0) if len(seq) else np.zeros((0,))
        return out

    def clear(self):
        for k in self.keys:
            self._store[k].clear()
        self._step = 0
