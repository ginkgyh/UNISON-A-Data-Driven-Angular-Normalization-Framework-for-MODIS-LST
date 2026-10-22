# Run UNISON locally

Use the six trusted bundles under [models/](../models/). The library versions in [requirements.txt](../requirements.txt) match the recorded model environment; Python 3.13 is recommended. On Linux, LightGBM may also require the system OpenMP runtime (`libgomp`).

## Run the example

From the repository root, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python tools/verify_release.py
python offline/predict.py --model models/Bare_Terra/model_bundle.pkl --input examples/Bare_Terra.csv --output result.csv --lst-column modisLST --lst-unit celsius --output-kelvin
```

`result.csv` retains the input columns and adds `deltaLST_K`, `prediction_status`, `nadir_LST_C` and `nadir_LST_K`. The example is synthetic and demonstrates software use only. For observations, prepare the required physical predictors as described in the [input contract](../docs/input_contract.md).

## Prepare your CSV

Inspect the selected bundle and create a header-only template:

```bash
python offline/inspect_model.py --model models/Bare_Terra/model_bundle.pkl --template my_input.csv
```

Fill the template with real prepared observations, or use an existing CSV with matching field names. CSV column order may differ: the code selects the required columns in the bundle's order. Extra columns such as pixel IDs, dates and coordinates are preserved. Do not rename the required features. Terra/Aqua and surface groups have different feature lists.

For the released models, `modisLST`, `DSAT` and `DSOIL` must be **Celsius**. Do not supply Kelvin inputs or pre-scaled values. Both fitted transforms are already included in the bundle. The predicted correction is near-nadir minus observed LST, so normalized LST is observed LST **plus** correction.

To predict only the correction, omit `--lst-column`. To add the correction to a separate observed-temperature column in Kelvin, supply that column's name and `--lst-unit kelvin`; required model input features remain in their trained units. The unit option declares the observed addition column, not an automatic conversion of model features. A conflicting declaration for the `modisLST` feature is rejected.

## Invalid rows and large inputs

The CSV reader preserves rows. Missing, nonnumeric, NaN, infinite or float32-overflowing required values receive blank predictions and `invalid_features:<names>`. Missing required column names stop the run. If a separate observed-LST column is invalid, the correction can still be present while normalized LST is blank and the status is `invalid_lst:<name>`.

Mask finite NoData codes such as `-9999` before calling the program; they are not automatically inferred. Never fill missing physical inputs with zero. Use `--chunk-size 10000` to control the number of rows predicted at once. Existing outputs require `--overwrite`, and the input file cannot be overwritten.

## Python use

```python
from offline.inference import load_bundle, predict_array

bundle = load_bundle("models/Bare_Terra/model_bundle.pkl")
# X must be a finite N × F matrix in bundle['feature_names'] order.
delta = predict_array(bundle, X)
```

Sibling manifest checksums are verified before deserialization. Checksums detect changed files; they do not make arbitrary pickle files safe. Load trusted sources only. The client preserves the raw model response at all angles, without a forced zero correction at 0°.

## Software checks

```bash
python -m unittest discover -s tests -v
```

The suite covers invalid rows, feature ordering, unit handling and equivalence to the supplied model formula. Optional tests against the author's original CSVs run only when `UNISON_REFERENCE_CSV_DIR` points to those files. These software checks do not replace the paper's scientific validation.
