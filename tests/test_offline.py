"""Behavioral tests; run with: python -m unittest discover -s tests -v."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from offline.inference import ValidationError, describe_bundle, load_bundle, predict_array, predict_csv, verify_manifest


class IdentityScaler:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


class SumModel:
    def predict(self, values):
        return np.sum(values, axis=1)


def simple_bundle():
    return {"lgb_model": SumModel(), "x_std_transformer": IdentityScaler(),
            "y_std_transformer": IdentityScaler(), "feature_names": ["a", "b"]}


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "input.csv"
        self.destination = self.root / "output.csv"

    def input(self, fields, rows):
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def output(self):
        with self.destination.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_named_columns_order_sign_and_both_temperature_scales(self):
        self.input(["id", "b", "a", "observed"], [["first", 2, -5, 20], ["second", 1, 5, 10]])
        result = predict_csv(simple_bundle(), self.source, self.destination,
                             lst_column="observed", output_kelvin=True, chunk_size=1)
        rows = self.output()
        self.assertEqual(result, {"rows": 2, "predicted": 2, "invalid_features": 0, "invalid_lst": 0})
        self.assertEqual([row["id"] for row in rows], ["first", "second"])
        self.assertEqual(float(rows[0]["deltaLST_K"]), -3)
        self.assertEqual(float(rows[0]["nadir_LST_C"]), 17)
        self.assertAlmostEqual(float(rows[0]["nadir_LST_K"]), 290.15)
        self.assertEqual(float(rows[1]["nadir_LST_C"]), 16)

    def test_invalid_features_remain_masked_and_are_never_zero_filled(self):
        self.input(["id", "a", "b"], [[1, "", 2], [2, "NaN", 3], [3, "inf", 4],
                                       [4, "-inf", 4], [5, "bad", 4], [6, "1e100", 4], [7, 0, 0]])
        result = predict_csv(simple_bundle(), self.source, self.destination, chunk_size=2)
        rows = self.output()
        self.assertEqual(result["invalid_features"], 6)
        self.assertTrue(all(row["deltaLST_K"] == "" and row["prediction_status"] == "invalid_features:a" for row in rows[:6]))
        self.assertEqual(rows[0]["a"], "")
        self.assertEqual(rows[4]["a"], "bad")
        self.assertEqual(rows[-1]["deltaLST_K"], "0")
        self.assertEqual(rows[-1]["prediction_status"], "ok")

    def test_invalid_optional_lst_preserves_valid_correction(self):
        self.input(["a", "b", "observed"], [[2, 3, ""], [2, 3, "NaN"], [2, 3, 280]])
        result = predict_csv(simple_bundle(), self.source, self.destination,
                             lst_column="observed", lst_unit="kelvin")
        rows = self.output()
        self.assertEqual(result["invalid_lst"], 2)
        self.assertEqual(rows[0]["deltaLST_K"], "5")
        self.assertEqual(rows[0]["nadir_LST_K"], "")
        self.assertEqual(rows[0]["prediction_status"], "invalid_lst:observed")
        self.assertEqual(rows[2]["nadir_LST_K"], "285")

    def test_missing_feature_is_an_error_before_output(self):
        self.input(["a"], [[1]])
        with self.assertRaisesRegex(ValidationError, "Missing input columns: b"):
            predict_csv(simple_bundle(), self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_duplicate_columns_and_output_collisions_are_errors(self):
        for header in (["a", "a", "b"], ["a", "b", "deltaLST_K"]):
            with self.subTest(header=header):
                self.input(header, [[1, 2, 3]])
                with self.assertRaises(ValidationError):
                    predict_csv(simple_bundle(), self.source, self.destination)

    def test_ragged_csv_fails_atomically(self):
        self.input(["a", "b"], [[1, 2], [3, 4, 5]])
        with self.assertRaisesRegex(ValidationError, "different number of fields"):
            predict_csv(simple_bundle(), self.source, self.destination, chunk_size=1)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_output_protection_and_atomic_overwrite(self):
        self.input(["a", "b"], [[1, 2]])
        self.destination.write_text("original output", encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "already exists"):
            predict_csv(simple_bundle(), self.source, self.destination)
        self.assertEqual(self.destination.read_text(), "original output")
        predict_csv(simple_bundle(), self.source, self.destination, overwrite=True)
        self.assertEqual(self.output()[0]["deltaLST_K"], "3")
        with self.assertRaisesRegex(ValidationError, "must differ"):
            predict_csv(simple_bundle(), self.source, self.source, overwrite=True)

    def test_header_only_input_is_valid(self):
        self.input(["a", "b"], [])
        self.assertEqual(predict_csv(simple_bundle(), self.source, self.destination)["rows"], 0)
        self.assertEqual(self.output(), [])

    def test_array_validation_and_float32_contract(self):
        self.assertEqual(predict_array(simple_bundle(), [[1.2, 3.4]]).dtype, np.float32)
        for values in ([[1]], [[float("nan"), 2]], [[1e100, 2]], [1, 2]):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                predict_array(simple_bundle(), values)

    def test_manifest_blocks_modified_model_before_pickle_load(self):
        model = self.root / "model_bundle.pkl"
        model.write_bytes(b"not a pickle")
        (self.root / "manifest.json").write_text(json.dumps({
            "file": model.name, "sha256": "0" * 64, "size_bytes": model.stat().st_size,
        }))
        with patch("offline.inference.joblib.load") as loader:
            with self.assertRaisesRegex(ValidationError, "SHA-256"):
                load_bundle(model)
            loader.assert_not_called()

    def test_bundle_manifest_and_schema_validation(self):
        model = self.root / "model_bundle.pkl"
        joblib.dump(simple_bundle(), model)
        manifest = {"file": model.name, "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                    "size_bytes": model.stat().st_size, "feature_names": ["a", "b"]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        self.assertEqual(load_bundle(model)["feature_names"], ["a", "b"])
        manifest["feature_names"] = ["b", "a"]
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValidationError, "feature order"):
            load_bundle(model)

    def test_configuration_validation(self):
        self.input(["a", "b"], [[1, 2]])
        for kwargs in ({"chunk_size": 0}, {"output_kelvin": True}, {"lst_unit": "fahrenheit"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                predict_csv(simple_bundle(), self.source, self.destination, **kwargs)

    def test_inspection_uses_retrained_temperature_metadata_and_original_fallback(self):
        model = self.root / "model_bundle.pkl"
        model.write_bytes(b"metadata test placeholder")
        bundle = simple_bundle()
        self.assertEqual(describe_bundle(bundle, model)["temperature_input_unit"], "celsius (DSAT, DSOIL, modisLST)")
        bundle["training_temperature_unit"] = "kelvin"
        self.assertEqual(describe_bundle(bundle, model)["temperature_input_unit"], "kelvin (DSAT, DSOIL, modisLST)")
        bundle["temperature_unit"] = "celsius"
        self.assertEqual(describe_bundle(bundle, model)["temperature_input_unit"], "celsius (DSAT, DSOIL, modisLST)")
        bundle["temperature_unit"] = "kelvin"
        self.assertEqual(describe_bundle(bundle, model)["temperature_input_unit"], "kelvin (DSAT, DSOIL, modisLST)")

    def test_modislst_unit_cannot_be_reinterpreted_but_external_kelvin_is_allowed(self):
        bundle = simple_bundle()
        bundle["feature_names"] = ["modisLST", "b"]
        self.input(["modisLST", "b", "observed_K"], [[20, 1, 293.15]])
        with self.assertRaisesRegex(ValidationError, "modisLST uses celsius"):
            predict_csv(bundle, self.source, self.destination, lst_column="modisLST", lst_unit="kelvin")
        self.assertFalse(self.destination.exists())
        predict_csv(bundle, self.source, self.destination, lst_column="observed_K", lst_unit="kelvin")
        self.assertAlmostEqual(float(self.output()[0]["nadir_LST_K"]), 314.15)
        bundle["temperature_unit"] = "kelvin"
        self.input(["modisLST", "b"], [[293.15, 1]])
        with self.assertRaisesRegex(ValidationError, "modisLST uses kelvin"):
            predict_csv(bundle, self.source, self.destination, lst_column="modisLST", overwrite=True)
        predict_csv(bundle, self.source, self.destination, lst_column="modisLST", lst_unit="kelvin", overwrite=True)
        self.assertEqual(self.output()[0]["prediction_status"], "ok")


if __name__ == "__main__":
    unittest.main()
