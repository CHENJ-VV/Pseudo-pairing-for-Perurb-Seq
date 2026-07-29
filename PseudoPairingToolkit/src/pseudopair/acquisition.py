"""Small, checksum-aware acquisition layer for public or local h5ad files."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Mapping


def sha256sum(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_one(item: Mapping[str, Any], overwrite: bool = False) -> dict[str, Any]:
    output = Path(item["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    expected = str(item.get("sha256", "")).lower().strip() or None

    if output.exists() and not overwrite:
        actual = sha256sum(output) if expected else None
        if expected and actual != expected:
            raise ValueError(f"Checksum mismatch for existing file: {output}")
        return {"output": str(output), "status": "reused", "sha256": actual}

    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    if item.get("source_path"):
        source = Path(item["source_path"])
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, temporary)
    else:
        request = urllib.request.Request(str(item["url"]), headers={"User-Agent": "pseudopair/0.1"})
        with urllib.request.urlopen(request) as response, open(temporary, "wb") as target:
            shutil.copyfileobj(response, target, length=8 * 1024 * 1024)

    actual = sha256sum(temporary)
    if expected and actual != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for downloaded file: {output}")
    temporary.replace(output)
    return {"output": str(output), "status": "downloaded", "sha256": actual}


def run_acquisition(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    block = dict(config.get("acquisition", {}))
    overwrite = bool(block.get("overwrite", False))
    return [acquire_one(item, overwrite=overwrite) for item in block.get("files", [])]
