# Input and output contract

The released `.pkl` bundles are the authority for feature names and order. The code selects CSV columns by those names before applying the saved transforms. Each bundle has a neighbouring `manifest.json` and `input_template.csv`; the [registry](../models/registry.json) lists all six.

## Prepare raw predictors

Use the study's quality control, land-cover grouping, observation matching and approximately 1 km common grid. For MODIS daytime data, convert `LST_Day_1km` to °C as `raw * 0.02 - 273.15`. Convert `Day_view_angl` to signed degrees using `raw - 65`. Do not replace signed VZA with its absolute value.

The released model preprocessing uses:

- `modisLST`, `DSAT`, `DSOIL`: **°C**, including ERA5 air and topsoil temperatures after subtraction of 273.15. The fitted scaler ranges and supplied application preprocessing support this convention.
- `DSR`, `DLR`: W m−2, matched to the MODIS observation using the study's five-hour window. Convert ERA5 accumulated radiation to flux before use.
- `DVSOIL`: volumetric soil moisture, m³ m−3.
- `DTR`: temperature difference; K and °C differences have equal numerical values.
- `LON`, `LAT`: longitude and latitude in degrees.
- `DEM`, `ELESD`, `TH`: metres; `SLOPE`: degrees.
- `LAI`, `EVI`, `NDWI`: scaled physical index values, not unscaled product integers.
- `GPP`: the study's physical GPP representation, kg C m−2; retain the training temporal aggregation and product scale.

Use only the variables required by the selected bundle. Bare Aqua omits NDWI; Bare Terra includes it. Vegetation groups have additional canopy variables, and Terra and Aqua place NDWI differently. TRI is absent from these bundles.

The manuscript expresses absolute temperatures in Kelvin. For these supplied executable bundles, temperature **inputs** are Celsius; convert final LST to Kelvin if required. Do not feed Kelvin features into the Celsius-trained scalers. This repository does not refit those scalers.

Match the study's fixed backgrounds: tree height (2019), summer GPP (2021–2023 mean), and the seasonal background drivers. CSV inference starts after predictor preparation; it does not download, align or derive satellite products automatically.

## Prediction

The sequence is raw float32 features → saved input transform → LightGBM → saved target inverse transform → float32 temperature difference. The compatibility keys `x_std_transformer` and `y_std_transformer` contain **MinMaxScaler**, not StandardScaler. Do not fit new scalers for inference or scale inputs twice.

The correction is `near_nadir_LST - observed_LST` and is **added** to the observation:

```text
nadir_LST_C = modisLST + deltaLST_K
nadir_LST_K = nadir_LST_C + 273.15
```

The code does not impose a zero-angle override. A learned correction may be nonzero for an angle of 0°.

## Missing values and scope

Missing required columns are a schema error. Invalid required values are retained with an invalid status and blank prediction. Supply a common valid-pixel mask online. Numeric fill values such as −9999 must be masked or converted to NaN before inference; finite sentinel values cannot all be identified automatically.

Fitted min/max ranges describe training inputs, not proof of scientific validity for every combination inside them. The [summer daytime and surface-domain limits](../README.md#supported-application) still apply.

## Check the bundle

Required keys: `lgb_model`, `x_std_transformer`, `y_std_transformer`, `feature_names`. The bundles also record sensor, surface grouping, target definition and library versions. Season is absent from original metadata. The model files are unchanged; verify their SHA-256 values against each manifest or [SHA256SUMS.txt](../models/SHA256SUMS.txt). Pickle loading can execute code: load trusted bundles only.
