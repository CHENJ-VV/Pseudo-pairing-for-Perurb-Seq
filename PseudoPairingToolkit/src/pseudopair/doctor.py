"""Dependency reporting without importing optional packages globally."""
from __future__ import annotations

import importlib.util
from typing import Any

REQUIREMENTS = {
    "core": ["numpy", "pandas", "scipy", "anndata", "scanpy", "sklearn", "h5py", "tqdm", "yaml"],
    "plotting": ["matplotlib"],
    "mlp": ["torch"],
    "ot": ["ot"],
    "seacells": ["SEACells"],
    "parquet": ["pyarrow"],
}


def dependency_report() -> dict[str, Any]:
    report = {}
    for group, modules in REQUIREMENTS.items():
        status = {module: importlib.util.find_spec(module) is not None for module in modules}
        report[group] = {"available": all(status.values()), "modules": status}
    return report
