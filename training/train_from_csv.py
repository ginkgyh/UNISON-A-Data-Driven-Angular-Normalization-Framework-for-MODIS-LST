#!/usr/bin/env python
"""Fit/export an UNISON-compatible angular model from a prepared labelled CSV.

This is the final regression/export step, not the full paper preprocessing or
an exact rebuild of any published bundle. See training/README.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


# Active recipe in the supplied Terra/Aqua training sources. These parameters
# are not a claim about the provenance of every pre-trained release bundle.
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "seed": 0,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path,
                        help="Prepared CSV containing all features and deltaLST labels.")
    schema = parser.add_mutually_exclusive_group(required=True)
    schema.add_argument("--reference-bundle", type=Path,
                        help="Trusted supplied .pkl; reads its feature order, not its fitted scalers.")
    schema.add_argument("--feature-schema", type=Path,
                        help="JSON object with ordered feature_names and optional descriptive metadata.")
    parser.add_argument("--output", required=True, type=Path,
                        help="New .pkl filename; existing files are never overwritten.")
    parser.add_argument("--temperature-unit", required=True, choices=("celsius", "kelvin"),
                        help="Absolute temperature unit already present in the CSV; no conversion is performed.")
    parser.add_argument("--model-id", help="Optional descriptive ID for the new model.")
    parser.add_argument("--group-column",
                        help="Optional group ID, e.g. ORIG_FID, kept wholly in train or test.")
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--rounds", default=2000, type=int)
    parser.add_argument("--params-json", type=Path,
                        help="Optional JSON object overriding the documented LightGBM defaults.")
    parser.add_argument("--nodata", default=-9999.0, type=float,
                        help="Numeric NoData sentinel to treat as missing, default -9999.")
    return parser.parse_args(argv)


def read_schema(args):
    manifest = {}
    if args.reference_bundle:
        # joblib/pickle can execute code: use only the trusted author-supplied
        # bundles, never arbitrary downloaded pickle files.
        import joblib
        manifest_path = args.reference_bundle.with_name("manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if not isinstance(manifest, dict):
                raise ValueError("Sibling manifest must be a JSON object.")
            expected = manifest.get("sha256")
            if (not isinstance(expected, str) or len(expected) != 64
                    or sha256_file(args.reference_bundle).lower() != expected.lower()):
                raise ValueError("Reference bundle SHA-256 differs from its sibling manifest.")
            if manifest.get("file", args.reference_bundle.name) != args.reference_bundle.name:
                raise ValueError("Manifest filename differs from the reference bundle.")
            if manifest.get("size_bytes", args.reference_bundle.stat().st_size) != args.reference_bundle.stat().st_size:
                raise ValueError("Reference bundle size differs from its sibling manifest.")
        schema = joblib.load(args.reference_bundle)
        if not isinstance(schema, dict):
            raise ValueError("Reference bundle must be a dictionary.")
        origin = args.reference_bundle.name
    else:
        schema = json.loads(args.feature_schema.read_text(encoding="utf-8-sig"))
        origin = args.feature_schema.name
    if not isinstance(schema, dict):
        raise ValueError("Feature schema must be a JSON object.")
    names = schema.get("feature_names")
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name.strip() for name in names)
            or len(names) != len(set(names))):
        raise ValueError("feature_names must be a nonempty ordered list of unique names.")
    if "deltaLST" in names:
        raise ValueError("deltaLST cannot appear among the predictors.")
    if manifest and manifest.get("feature_names", names) != names:
        raise ValueError("Reference feature order differs from its sibling manifest.")
    if schema.get("target_name", "deltaLST") != "deltaLST":
        raise ValueError("This helper trains angular deltaLST models, not reference-transfer models.")
    known_unit = (schema.get("temperature_input_unit") or schema.get("temperature_unit")
                  or schema.get("training_temperature_unit")
                  or manifest.get("temperature_input_unit") or manifest.get("temperature_unit")
                  or manifest.get("training_temperature_unit"))
    aliases = {"C": "celsius", "degC": "celsius", "°C": "celsius", "K": "kelvin"}
    known_unit = aliases.get(known_unit, known_unit)
    if known_unit in ("celsius", "kelvin") and known_unit != args.temperature_unit:
        raise ValueError("CSV temperature unit differs from the reference schema; convert the CSV explicitly.")
    metadata_keys = ("sensor_type", "surface_type", "surface_region", "area_mc", "layer")
    metadata = {key: schema[key] for key in metadata_keys if key in schema}
    return names, metadata, origin


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    args = parse_args(argv)
    if not 0 < args.test_size < 1 or args.rounds < 1:
        raise ValueError("test-size must be between 0 and 1; rounds must be positive.")
    if args.output.suffix.lower() != ".pkl":
        raise ValueError("Output must have a .pkl extension.")
    report_path = args.output.with_suffix(".training.json")
    if args.output.exists() or report_path.exists():
        raise FileExistsError("Choose a new output name; the bundle or report already exists.")

    import joblib
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.preprocessing import MinMaxScaler

    names, metadata, schema_origin = read_schema(args)
    header = pd.read_csv(args.csv, nrows=0).columns.tolist()
    required = names + ["deltaLST"] + ([args.group_column] if args.group_column else [])
    missing = [name for name in required if name not in header]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}. No predictors are synthesized or zero-filled.")
    frame = pd.read_csv(args.csv, usecols=list(dict.fromkeys(required)))
    numeric = frame.loc[:, names + ["deltaLST"]].apply(pd.to_numeric, errors="raise")
    numeric = numeric.replace([np.inf, -np.inf, args.nodata], np.nan)
    valid = numeric.notna().all(axis=1)
    if args.group_column:
        valid &= frame[args.group_column].notna()
    clean = numeric.loc[valid].astype("float32")
    if len(clean) < 10:
        raise ValueError("Fewer than 10 valid labelled rows remain.")
    positions = np.arange(len(clean))
    if args.group_column:
        groups = frame.loc[valid, args.group_column].to_numpy()
        if len(pd.unique(groups)) < 2:
            raise ValueError("A group split requires at least two different group IDs.")
        split = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        train_pos, test_pos = next(split.split(positions, groups=groups))
        split_kind = f"group:{args.group_column}"
    else:
        train_pos, test_pos = train_test_split(
            positions, test_size=args.test_size, random_state=args.seed)
        split_kind = "random_rows"
        print("Random row split: related angular rows may occur in both partitions. "
              "Use --group-column for independent sample groups.", file=sys.stderr)
    if len(test_pos) < 2:
        raise ValueError("At least two held-out rows are needed to report R2.")

    x_train = clean.iloc[train_pos].loc[:, names]
    x_test = clean.iloc[test_pos].loc[:, names]
    y_train = clean.iloc[train_pos]["deltaLST"].to_numpy()
    y_test = clean.iloc[test_pos]["deltaLST"].to_numpy()
    # The held-out partition never contributes to scaler fitting or model fit.
    # Match the offline predictor's raw float32 ndarray contract. Feature order
    # is carried explicitly in the bundle and named LightGBM training frames.
    x_scaler = MinMaxScaler().fit(x_train.to_numpy(dtype="float32"))
    y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
    x_train_scaled = pd.DataFrame(x_scaler.transform(x_train.to_numpy(dtype="float32")), columns=names, index=x_train.index)
    x_test_scaled = pd.DataFrame(x_scaler.transform(x_test.to_numpy(dtype="float32")), columns=names, index=x_test.index)
    y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    params = {**DEFAULT_PARAMS, "seed": args.seed}
    if args.params_json:
        overrides = json.loads(args.params_json.read_text(encoding="utf-8-sig"))
        if not isinstance(overrides, dict):
            raise ValueError("params-json must contain a JSON object.")
        params.update(overrides)
    # No early stopping in the supplied source recipe. A genuinely held-out
    # test set is evaluated only after this fixed-round fit has finished.
    train_set = lgb.Dataset(x_train_scaled, y_train_scaled, feature_name=names)
    model = lgb.train(params, train_set, num_boost_round=args.rounds)
    pred_scaled = np.asarray(model.predict(x_test_scaled)).reshape(-1, 1)
    pred = y_scaler.inverse_transform(pred_scaled).ravel()
    metrics = {
        "rmse_K": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae_K": float(mean_absolute_error(y_test, pred)),
        "bias_prediction_minus_label_K": float(np.mean(pred - y_test)),
        "r2": float(r2_score(y_test, pred)),
    }
    versions = {name: importlib.metadata.version(name)
                for name in ("lightgbm", "scikit-learn", "numpy", "pandas", "joblib")}
    versions["python"] = platform.python_version()
    training = {
        "scope": "prepared_csv_fit_export; not full paper preprocessing or exact published-model rebuild",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": args.csv.name,
        "source_csv_sha256": sha256_file(args.csv),
        "schema_source": schema_origin,
        "input_rows": len(frame),
        "valid_rows": len(clean),
        "dropped_rows": int((~valid).sum()),
        "train_rows": len(train_pos),
        "test_rows": len(test_pos),
        "split": split_kind,
        "test_size": args.test_size,
        "seed": args.seed,
        "num_boost_round": args.rounds,
        "params": params,
        "metrics": metrics,
    }
    bundle = {
        **metadata,
        "lgb_model": model,
        "x_std_transformer": x_scaler,
        "y_std_transformer": y_scaler,
        "feature_names": names,
        "target_name": "deltaLST",
        "target_definition": "near-nadir LST minus directional MODIS LST; add deltaLST to modisLST",
        "target_unit": "K",
        "temperature_input_unit": args.temperature_unit,
        "temperature_unit": args.temperature_unit,
        "bundle_version": "prepared_csv_export_v1",
        "versions": versions,
        "training": training,
    }
    if args.model_id:
        bundle["model_id"] = args.model_id
    report = {key: value for key, value in bundle.items()
              if key not in ("lgb_model", "x_std_transformer", "y_std_transformer")}
    report_text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids inadvertently replacing a supplied bundle.
    with args.output.open("xb") as stream:
        joblib.dump(bundle, stream, compress=3)
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report_text)
    print(json.dumps({"bundle": str(args.output), "report": str(report_path),
                      "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
