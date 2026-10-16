from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_pipelines"


def load_module(filename: str, module_name: str):
    path = SOURCE / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def acquisition_module():
    return load_module("data_acquisition.py", "prt661_stage1")


def preprocessing_module():
    return load_module("preprocessing.py", "prt661_stage2")
