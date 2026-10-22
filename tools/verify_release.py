#!/usr/bin/env python3
"""Verify all six release model checksums without loading pickle files."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


MODEL_IDS = [f"{surface}_{sensor}" for surface in ("Bare", "MLVeg", "MHVeg") for sensor in ("Terra", "Aqua")]


def verify(root: Path) -> list[str]:
    failures = []
    for model_id in MODEL_IDS:
        directory = root / "models" / model_id
        model = directory / "model_bundle.pkl"
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("model_id") != model_id:
                raise ValueError("model_id does not match its directory")
            if manifest.get("file") != model.name:
                raise ValueError("file must be model_bundle.pkl")
            if model.stat().st_size != manifest.get("size_bytes"):
                raise ValueError("file size differs from manifest")
            digest = hashlib.sha256()
            with model.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != manifest.get("sha256"):
                raise ValueError("SHA-256 differs from manifest")
            names = manifest.get("feature_names")
            if not isinstance(names, list) or not names or len(names) != len(set(names)):
                raise ValueError("feature_names must be a nonempty list without duplicates")
            print(f"OK {model_id}: {len(names)} features; {model.stat().st_size:,} bytes")
        except (OSError, ValueError, TypeError) as exc:
            failures.append(f"{model_id}: {exc}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = verify(args.root)
    if failures:
        for failure in failures:
            print(f"ERROR {failure}", file=sys.stderr)
        return 1
    print("All six bundled model files match their release manifests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
