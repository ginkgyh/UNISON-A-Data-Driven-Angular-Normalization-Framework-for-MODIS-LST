#!/usr/bin/env python3
"""Inspect a trusted model and optionally write its required CSV header."""

import argparse
import csv
import json
import sys
from pathlib import Path

try:
    from .inference import ValidationError, describe_bundle, load_bundle
except ImportError:
    from inference import ValidationError, describe_bundle, load_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Pickle/joblib loading can execute code: use only trusted model files.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", help="Write a header-only CSV in this model's feature order")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        bundle = load_bundle(args.model)
        description = describe_bundle(bundle, args.model)
        if args.template:
            template = Path(args.template)
            if template.resolve() == Path(args.model).resolve():
                raise ValidationError("Template path cannot overwrite the model.")
            if template.exists() and not args.overwrite:
                raise ValidationError(f"Template already exists; use --overwrite: {template}")
            with template.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(bundle["feature_names"])
            description["template"] = str(template)
        print(json.dumps(description, indent=2, ensure_ascii=False))
    except (ValidationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
