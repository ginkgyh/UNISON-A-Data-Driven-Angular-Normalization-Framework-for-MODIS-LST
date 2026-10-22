"""Compare all released bundles with the original app.py numerical formula.

The default fixtures use each model's fitted input range. To additionally test
real training rows during a local release audit, set UNISON_REFERENCE_CSV_DIR
to the directory holding the original MC_<surface>_DE_<sensor>_... CSV files.
That optional directory is never copied into the release by these tests.
"""

import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from offline.inference import load_bundle, predict_array, predict_csv


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(os.environ.get("UNISON_MODEL_DIR", str(ROOT / "models")))
MODEL_IDS = [f"{surface}_{sensor}" for surface in ("Bare", "MLVeg", "MHVeg") for sensor in ("Terra", "Aqua")]


def original_formula(bundle, values):
    """Keep the deployment source calculation explicit for comparison."""
    raw = np.asarray(values, dtype=np.float32)
    x_scaled = bundle["x_std_transformer"].transform(raw)
    pred_scaled = bundle["lgb_model"].predict(x_scaled)
    return bundle["y_std_transformer"].inverse_transform(
        np.asarray(pred_scaled).reshape(-1, 1)
    ).ravel().astype(np.float32)


class RealModelTests(unittest.TestCase):
    def test_all_six_bundles_match_original_pipeline_and_csv_chunks(self):
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                path = MODEL_ROOT / model_id / "model_bundle.pkl"
                self.assertTrue(path.is_file(), f"Release model is missing: {path}")
                bundle = load_bundle(path)
                scaler = bundle["x_std_transformer"]
                fractions = np.array([0.2, 0.5, 0.8], dtype=np.float32).reshape(-1, 1)
                values = np.asarray(scaler.data_min_ + fractions * (scaler.data_max_ - scaler.data_min_), dtype=np.float32)
                expected = original_formula(bundle, values)
                np.testing.assert_array_equal(predict_array(bundle, values), expected)
                self.check_csv(bundle, values, expected)

    def check_csv(self, bundle, values, expected):
        names = bundle["feature_names"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["row_id"] + list(reversed(names)))
                writer.writeheader()
                for index, row in enumerate(values):
                    writer.writerow({"row_id": index, **dict(zip(names, [format(float(v), ".9g") for v in row]))})
            baseline = None
            for chunk_size in (1, 2, 10000):
                output = Path(directory) / f"output_{chunk_size}.csv"
                result = predict_csv(bundle, source, output, lst_column="modisLST",
                                     output_kelvin=True, chunk_size=chunk_size)
                self.assertEqual(result["predicted"], len(values))
                with output.open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                actual = np.array([float(row["deltaLST_K"]) for row in rows], dtype=np.float32)
                np.testing.assert_array_equal(actual, expected)
                lst_index = names.index("modisLST")
                for index, row in enumerate(rows):
                    self.assertEqual(row["prediction_status"], "ok")
                    self.assertAlmostEqual(float(row["nadir_LST_C"]), float(values[index, lst_index]) + float(expected[index]), places=5)
                    self.assertAlmostEqual(float(row["nadir_LST_K"]), float(row["nadir_LST_C"]) + 273.15, places=7)
                if baseline is None:
                    baseline = output.read_bytes()
                else:
                    self.assertEqual(output.read_bytes(), baseline)

    @unittest.skipUnless(os.environ.get("UNISON_REFERENCE_CSV_DIR"), "Original local training CSVs were not requested")
    def test_real_source_rows_match_all_six_original_pipelines(self):
        directory = Path(os.environ["UNISON_REFERENCE_CSV_DIR"])
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                surface, sensor = model_id.split("_")
                layer = 16 if surface == "Bare" else 0
                region = "highLAT" if surface == "MHVeg" else "GA_lowLAT"
                source = directory / f"MC_{surface}_DE_{sensor}_Layer_{layer}_{region}_Summer_Y2013_2023.csv"
                bundle = load_bundle(MODEL_ROOT / model_id / "model_bundle.pkl")
                sample = []
                with source.open(encoding="utf-8-sig", newline="") as stream:
                    for line_number, row in enumerate(csv.DictReader(stream)):
                        try:
                            values = np.asarray([float(row[name]) for name in bundle["feature_names"]], dtype=np.float32)
                        except (ValueError, TypeError):
                            continue
                        if np.all(np.isfinite(values)):
                            sample.append(values)
                        if len(sample) == 5:
                            break
                        if line_number >= 20000:
                            self.fail(f"No sufficient finite sample rows in first 20000 lines: {source}")
                self.assertEqual(len(sample), 5)
                values = np.stack(sample)
                expected = original_formula(bundle, values)
                np.testing.assert_array_equal(predict_array(bundle, values), expected)
                self.check_csv(bundle, values, expected)


if __name__ == "__main__":
    unittest.main()
