#!/usr/bin/env python3
"""Add reproducible model-file hashes to an existing run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    model = args.model_dir.resolve()
    files = [
        path for path in sorted(model.rglob("*"))
        if path.is_file() and (path.suffix in {".safetensors", ".pt", ".bin"} or path.name in {"config.json", "model.safetensors.index.json"})
    ]
    inventory = [{"path": str(path.relative_to(model)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    combined = hashlib.sha256()
    for item in inventory:
        combined.update(str(item["path"]).encode())
        combined.update(str(item["bytes"]).encode())
        combined.update(str(item["sha256"]).encode())
    manifest = json.loads(args.manifest.read_text())
    manifest["model_hash_root"] = str(model)
    manifest["model_files_sha256"] = inventory
    manifest["model_digest_sha256"] = combined.hexdigest()
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"enriched {args.manifest}: {len(inventory)} model files, digest={combined.hexdigest()}")


if __name__ == "__main__":
    main()
