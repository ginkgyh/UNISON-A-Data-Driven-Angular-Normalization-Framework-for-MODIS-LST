# Build a model bundle from prepared training samples

Routine application uses the supplied trained `.pkl` files; retraining is optional.
This folder provides the concise final fit/export step for an already prepared,
labelled CSV. It does **not** reconstruct the full satellite-data preparation,
conditional label generation, or cross-surface learning procedure, and does not
claim to regenerate any released model byte for byte.

## Method in brief

1. Quality-control daytime summer MODIS observations and align the environmental
   predictors to the MODIS grid. Prepare Terra and Aqua separately.
2. Within each land-cover type and surface-air-temperature class, divide LAI
   (vegetation) or DTR (bare surfaces) into six surface-state bins, and signed VZA
   into 13 bins spanning -65 to +65 degrees. The central -5 to +5 degree bin is the
   near-nadir reference.
3. Construct the conditional angular label as near-nadir mean LST minus directional
   mean LST, then smooth its angular response using LOESS. Keep the surface and
   meteorological predictors associated with the samples.
4. Fit type-specific models, then perform the study's cross-surface joint modeling
   for MLVeg, MHVeg and Bare, including transition-zone samples. Prepare one final
   labelled table for each sensor and surface group.
5. Fit the final regression and save its model, fitted input/target transformations,
   ordered feature names, and metadata together. The helper below implements this
   last step. Landsat/VIIRS cross-sensor reference-transfer models are used for
   evaluation and are separate from the angular model.

Use the manuscript for the scientific method. An exact rebuild also needs the
original preprocessing settings, sample assignments, conditional labels, and
corresponding source-data version; these cannot be inferred from a fitted model.

## Required CSV

- One numeric column for every ordered feature in the selected model schema.
- `deltaLST`, in kelvin differences: near-nadir LST minus directional MODIS LST.
- Optionally a persistent sampling-unit identifier, such as `ORIG_FID`, for a
  grouped train/test split.

Temperature differences have the same numeric value in kelvin and degrees Celsius.
Absolute temperatures in `modisLST`, `DSAT` and `DSOIL` must use the unit expected by
the chosen schema. `--temperature-unit` records the unit already in the CSV; the
helper performs **no** conversion. A newly trained Celsius-input model requires
Celsius inputs at prediction time, including online use.

The legacy `Traning Data of UNISON.zip` is retained for provenance. It is not a
verified complete rebuild dataset for the current release. In particular, its
Terra low-latitude vegetation entry duplicates the Bare Terra entry. Do not
relabel, fabricate, or zero-fill missing predictors. Supply corrected, compatible
training tables before training the affected configurations.

## Run

Install the repository's Python dependencies, then run from the repository root.
For example, use the released Bare Terra feature schema:

```bash
python training/train_from_csv.py \
  --csv prepared_training.csv \
  --reference-bundle models/Bare_Terra/model_bundle.pkl \
  --temperature-unit celsius \
  --output models/my_new_model.pkl
```

`--reference-bundle` reads feature order and descriptive sensor/surface metadata;
it never reuses the reference model's fitted scalers. Only load trusted `.pkl`
files: joblib/pickle loading can execute code. When a sibling `manifest.json` is
present, its checksum is verified before loading and its temperature input unit
is checked against `--temperature-unit`.

Alternatively, use `--feature-schema schema.json` instead of `--reference-bundle`.
A minimal schema is a JSON object with an ordered `feature_names` array; for
example `{"feature_names": ["LON", "LAT", "modisLST", "modisAngle"]}`. This
illustrates the schema format, not a recommended predictor set or a reproduction
of UNISON. Optional schema keys are `target_name` (must be `deltaLST`),
`temperature_input_unit` or `temperature_unit` (`celsius` or `kelvin`), `sensor_type`, `surface_type`,
`surface_region`, `area_mc`, and `layer`.

## Recipe and validation

Defaults follow the active supplied Terra/Aqua regression scripts: an 80%/20%
random row split with seed 0; input and target `MinMaxScaler` fitted **only** on
training rows; 2,000 LightGBM rounds; learning rate 0.03; 31 leaves; maximum depth
8; minimum 100 rows per leaf; feature fraction 0.85; bagging fraction 0.8 with
frequency 1; L1/L2 penalties 0.1/1.0; deterministic, column-wise training.
There is no early stopping. The held-out set is used only after the fixed-round
fit to report RMSE, MAE, prediction-minus-label Bias in K, and R².

The six released LightGBM Boosters also store these parameter values and 2,000
trees; their manifests record `model_parameters` and `num_trees`. These values
differ from the hyperparameters printed in the manuscript's Table A3. A fitted
Booster does not establish the original sample split or all upstream decisions.
The helper's exported JSON records its actual recipe; reconcile the intended
paper/release version before claiming exact reproducibility. `--params-json` can
override the LightGBM parameters; `--rounds`, `--test-size`, and `--seed` are explicit.

Related VZA rows from the same sampling unit can enter both sides of a random
row split. Such metrics describe interpolation among prepared labels and do not
establish spatial or temporal independence. To keep a sampling unit wholly on
one side, add `--group-column ORIG_FID` (or another scientifically appropriate ID).
This uses `GroupShuffleSplit` and changes the split from the original row recipe.
A spatial or temporal block validation needs appropriate group IDs prepared
upstream. Avoid splitting related observations before computing shared training
labels if an independent validation claim is intended.

Rows containing missing/non-finite required predictors or targets, or the
configurable NoData sentinel (default -9999), are excluded and counted. No missing
predictor is synthesized. The helper saves a new `.pkl` plus `.training.json`
with feature order, units, parameters, split counts, metrics, source CSV hash and
library versions. Existing output files are never overwritten.

The released pretrained bundles are unchanged by this helper.
