"""Load a trusted UNISON bundle and apply its original prediction pipeline.

The published bundles expect DSAT, DSOIL and modisLST in degrees Celsius.
No input unit conversion, missing-value imputation or nadir-zero override is
performed. A correction of 1 K has the same magnitude as a difference of 1 C.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np


REQUIRED_KEYS = ("lgb_model", "x_std_transformer", "y_std_transformer", "feature_names")


class ValidationError(ValueError):
    """A model or input cannot be used without correcting its contents."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(model_path: str | Path) -> dict[str, Any] | None:
    """Check a sibling manifest before deserialization, when it is supplied."""
    path = Path(model_path)
    manifest_path = path.with_name("manifest.json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValidationError(f"Cannot read model manifest: {manifest_path}: {exc}") from exc
    expected = manifest.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValidationError(f"Manifest must supply a SHA-256 checksum: {manifest_path}")
    if manifest.get("file", path.name) != path.name:
        raise ValidationError("Manifest file name does not match the selected model.")
    if "size_bytes" in manifest and manifest["size_bytes"] != path.stat().st_size:
        raise ValidationError("Model size differs from manifest; download the original bundle again.")
    if sha256_file(path).lower() != expected.lower():
        raise ValidationError("Model SHA-256 differs from manifest; download the original bundle again.")
    return manifest


def load_bundle(model_path: str | Path) -> dict[str, Any]:
    """Deserialize only a model obtained from a source you trust.

    Pickle/joblib files can execute code while being loaded. Checksums detect
    accidental changes; they do not establish the publisher's identity.
    """
    path = Path(model_path)
    if not path.is_file():
        raise ValidationError(f"Model file does not exist: {path}")
    manifest = verify_manifest(path)
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValidationError("Model bundle must be a dictionary.")
    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise ValidationError(f"Model bundle is missing required keys: {', '.join(missing)}")
    names = list(bundle["feature_names"])
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValidationError("Model feature_names must be a nonempty list of column names.")
    if len(set(names)) != len(names):
        raise ValidationError("Model feature_names contains duplicate names.")
    if manifest and "feature_names" in manifest and manifest["feature_names"] != names:
        raise ValidationError("Model feature order differs from manifest.")
    for key, method in (("lgb_model", "predict"), ("x_std_transformer", "transform"),
                        ("y_std_transformer", "inverse_transform")):
        if not callable(getattr(bundle[key], method, None)):
            raise ValidationError(f"Bundle {key} does not provide {method}().")
    bundle["feature_names"] = names
    return bundle


def predict_array(bundle: dict[str, Any], values: Any) -> np.ndarray:
    """Apply exactly the app.py formula, including float32 input and output."""
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.asarray(values, dtype=np.float32)
    feature_count = len(bundle["feature_names"])
    if raw.ndim != 2 or raw.shape[1] != feature_count:
        raise ValidationError(f"Expected a matrix with {feature_count} feature columns.")
    if not np.all(np.isfinite(raw)):
        raise ValidationError("Prediction features must be finite float32 values.")
    if raw.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    scaled = bundle["x_std_transformer"].transform(raw)
    pred_scaled = bundle["lgb_model"].predict(scaled)
    result = bundle["y_std_transformer"].inverse_transform(
        np.asarray(pred_scaled).reshape(-1, 1)
    ).ravel().astype(np.float32)
    if result.shape != (raw.shape[0],) or not np.all(np.isfinite(result)):
        raise ValidationError("The model returned a nonfinite result or an unexpected row count.")
    return result


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def temperature_input_unit(bundle: dict[str, Any]) -> str:
    """Read retrained-bundle metadata; the six original bundles use Celsius."""
    unit = bundle.get("temperature_unit") or bundle.get("training_temperature_unit") or "celsius"
    if unit not in ("celsius", "kelvin"):
        raise ValidationError(f"Unsupported model temperature input unit: {unit!r}")
    return unit


def predict_csv(
    bundle: dict[str, Any], input_path: str | Path, output_path: str | Path,
    *, lst_column: str | None = None, lst_unit: str = "celsius",
    output_kelvin: bool = False, chunk_size: int = 10000, overwrite: bool = False,
) -> dict[str, int]:
    """Preserve input rows and mark invalid features with blank predictions."""
    source, destination = Path(input_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValidationError("Input and output paths must differ.")
    if destination.exists() and not overwrite:
        raise ValidationError(f"Output already exists; use --overwrite: {destination}")
    if chunk_size < 1:
        raise ValidationError("Chunk size must be at least 1.")
    if lst_unit not in ("celsius", "kelvin"):
        raise ValidationError("LST unit must be celsius or kelvin.")
    if output_kelvin and not lst_column:
        raise ValidationError("--output-kelvin requires --lst-column.")
    if lst_column == "modisLST" and lst_unit != temperature_input_unit(bundle):
        raise ValidationError(
            f"Model feature modisLST uses {temperature_input_unit(bundle)}; "
            f"--lst-unit {lst_unit} cannot reinterpret that feature. Use the model's "
            "input unit, or supply a separate observed LST column in the other unit."
        )
    names = bundle["feature_names"]
    added = ["deltaLST_K", "prediction_status"]
    temperature_fields: list[str] = []
    if lst_column:
        temperature_fields.append("nadir_LST_C" if lst_unit == "celsius" else "nadir_LST_K")
        if output_kelvin and lst_unit == "celsius":
            temperature_fields.append("nadir_LST_K")
    added += temperature_fields
    counts = {"rows": 0, "predicted": 0, "invalid_features": 0, "invalid_lst": 0}
    temporary: str | None = None
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as input_stream:
            reader = csv.DictReader(input_stream)
            fields = reader.fieldnames
            if not fields:
                raise ValidationError("Input CSV must contain a header.")
            if len(set(fields)) != len(fields):
                raise ValidationError("Input CSV contains duplicate column names.")
            missing = [name for name in names if name not in fields]
            if lst_column and lst_column not in fields:
                missing.append(lst_column)
            if missing:
                raise ValidationError(f"Missing input columns: {', '.join(dict.fromkeys(missing))}")
            collision = [name for name in added if name in fields]
            if collision:
                raise ValidationError(f"Output column already exists in input: {', '.join(collision)}")
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", suffix=".tmp",
                prefix=destination.name + ".", dir=destination.parent, delete=False,
            ) as output_stream:
                temporary = output_stream.name
                writer = csv.DictWriter(output_stream, fieldnames=fields + added)
                writer.writeheader()

                def write_chunk(rows: list[dict[str, Any]]) -> None:
                    valid_indices, values = [], []
                    for index, row in enumerate(rows):
                        counts["rows"] += 1
                        if None in row or any(row.get(field) is None for field in fields):
                            raise ValidationError(f"CSV row {counts['rows'] + 1} has a different number of fields than the header.")
                        converted = [_finite_float(row.get(name)) for name in names]
                        bad = [name for name, value in zip(names, converted)
                               if value is None or abs(value) > float(np.finfo(np.float32).max)]
                        row.update({name: "" for name in added})
                        if bad:
                            row["prediction_status"] = "invalid_features:" + ";".join(bad)
                            counts["invalid_features"] += 1
                        else:
                            valid_indices.append(index)
                            values.append(converted)
                    predictions = predict_array(bundle, values) if values else []
                    for index, correction in zip(valid_indices, predictions):
                        row = rows[index]
                        delta = float(correction)
                        row["deltaLST_K"] = format(delta, ".9g")
                        row["prediction_status"] = "ok"
                        counts["predicted"] += 1
                        if lst_column:
                            observed = _finite_float(row[lst_column])
                            if observed is None:
                                row["prediction_status"] = "invalid_lst:" + lst_column
                                counts["invalid_lst"] += 1
                            else:
                                nadir = observed + delta
                                row[temperature_fields[0]] = format(nadir, ".12g")
                                if output_kelvin and lst_unit == "celsius":
                                    row["nadir_LST_K"] = format(nadir + 273.15, ".12g")
                    writer.writerows(rows)

                chunk: list[dict[str, Any]] = []
                for row in reader:
                    chunk.append(row)
                    if len(chunk) == chunk_size:
                        write_chunk(chunk)
                        chunk = []
                if chunk:
                    write_chunk(chunk)
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return counts


def describe_bundle(bundle: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    """Return JSON-compatible metadata and the fitted input ranges."""
    scaler = bundle["x_std_transformer"]
    ranges = {}
    for index, name in enumerate(bundle["feature_names"]):
        if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
            ranges[name] = {"min": float(scaler.data_min_[index]), "max": float(scaler.data_max_[index])}
    return {
        "file": Path(model_path).name,
        "sha256": sha256_file(model_path),
        "size_bytes": Path(model_path).stat().st_size,
        "feature_names": bundle["feature_names"],
        "metadata": {key: value for key, value in bundle.items() if key not in REQUIRED_KEYS},
        "x_transformer": type(scaler).__name__,
        "y_transformer": type(bundle["y_std_transformer"]).__name__,
        "fitted_input_ranges": ranges,
        "temperature_input_unit": temperature_input_unit(bundle) + " (DSAT, DSOIL, modisLST)",
        "correction_definition": "deltaLST = nadir LST - observed LST; differences are in K",
        "manifest_verified": Path(model_path).with_name("manifest.json").exists(),
    }
