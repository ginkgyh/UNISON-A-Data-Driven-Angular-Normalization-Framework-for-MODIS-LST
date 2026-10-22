#!/usr/bin/env python3
"""Apply a trusted UNISON .pkl bundle to named CSV feature columns."""

import argparse
import json
import sys

try:
    from .inference import ValidationError, load_bundle, predict_csv
except ImportError:
    from inference import ValidationError, load_bundle, predict_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Load only trusted pickle/joblib files. Published model features DSAT, "
        "DSOIL and modisLST must be in Celsius; no automatic input conversion. "
        "Invalid rows retain their original values and get blank predictions."
    ))
    parser.add_argument("--model", required=True, help="Trusted model_bundle.pkl path")
    parser.add_argument("--input", required=True, help="Input CSV with named model features")
    parser.add_argument("--output", required=True, help="Output CSV (input columns are preserved)")
    parser.add_argument("--lst-column", help="Optional observed LST column for nadir = observed + deltaLST")
    parser.add_argument("--lst-unit", choices=("celsius", "kelvin"), default="celsius",
                        help="Unit of the optional observed LST column (default: celsius)")
    parser.add_argument("--output-kelvin", action="store_true", help="Also produce nadir_LST_K; requires --lst-column")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Rows per prediction chunk (default: 10000)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    try:
        result = predict_csv(load_bundle(args.model), args.input, args.output,
                             lst_column=args.lst_column, lst_unit=args.lst_unit,
                             output_kelvin=args.output_kelvin, chunk_size=args.chunk_size,
                             overwrite=args.overwrite)
    except (ValidationError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"output": args.output, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
