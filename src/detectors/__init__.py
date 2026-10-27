"""Anomaly detectors: IBAD and the baselines it is compared against.

Every detector implements :class:`~src.detectors.base.Detector`, so the
evaluation and the experiments treat them interchangeably. Adding a method
means writing one module here, one config file, and one registry entry.
"""

import importlib

from src.detectors.base import Detector

# name -> "module:Class". Imported lazily, so a run only pulls in the
# dependencies of the methods it actually uses.
REGISTRY = {
    "ibad": "src.detectors.ibad:IBADDetector",
    # θ-space baselines
    "histmean": "src.detectors.theta:HistMean",
    "ghmm": "src.detectors.theta:GHMM",
    "iforest": "src.detectors.theta:IsolationForestDetector",
    "ocsvm": "src.detectors.theta:OCSVMDetector",
    "deepsvdd": "src.detectors.theta:DeepSVDDDetector",
    # sequence baselines
    "bestad": "src.detectors.bestad:BeSTADDetector",
    "tod4traj": "src.detectors.tod4traj:TOD4TrajDetector",
    "icad": "src.detectors.icad:ICADDetector",
    "trajgen": "src.detectors.trajgen:TrajGenDetector",
}


def detector_class(name: str) -> type[Detector]:
    if name not in REGISTRY:
        raise KeyError(f"Unknown method {name!r}; expected one of {list(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)


__all__ = ["Detector", "REGISTRY", "detector_class"]
