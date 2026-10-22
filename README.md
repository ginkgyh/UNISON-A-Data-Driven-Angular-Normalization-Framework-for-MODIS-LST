# UNISON: angular normalization of MODIS land surface temperature

UNISON estimates the correction from a directional Terra or Aqua MODIS land surface temperature (LST) observation to a near-nadir equivalent. This repository offers **three ways to use or study the method**. The pretrained `.pkl` bundles and local application are available now; the author-hosted Earth Engine/Vertex AI service is **not yet deployed**; and a concise training workflow explains how to create a new bundle from prepared, labelled samples.

**Start here:** [run a pretrained model locally](#1-run-a-pretrained-model-locally) · [online service status](#2-author-hosted-online-inference-planned) · [build a new bundle](#3-build-a-new-pkl-bundle) · [input contract](docs/input_contract.md) · [中文说明](README.zh-CN.md)

## What is included

| Path | Purpose |
| --- | --- |
| [`models/`](models/) | Six pretrained bundles, manifests, feature orders, units, and checksums |
| [`offline/`](offline/) | Local batch CSV inference and Python API |
| [`examples/`](examples/) | Synthetic format examples for the six models; **not** validation data |
| [`cloud/Bare_Aqua/`](cloud/Bare_Aqua/) | A complete custom-container example using the supplied Aqua bare-surface bundle |
| [`online/`](online/) | Earth Engine client template and endpoint registry; all endpoints currently inactive |
| [`training/`](training/) | Brief method and final fit/export helper for prepared labelled CSV data |
| [`docs/`](docs/) | Detailed predictor contract and legacy-data notes |

Choose the bundle matching both the satellite and surface group. The six IDs are `Bare_Aqua`, `Bare_Terra`, `MLVeg_Aqua`, `MLVeg_Terra`, `MHVeg_Aqua`, and `MHVeg_Terra`. Each bundle has its **own ordered feature list**. The published application scope is clear-sky **summer daytime** observations over the studied natural surfaces. Other seasons, night-time observations, excluded land cover, and other sensors need separate validation.

The repository retains the earlier [`Traning Data of UNISON.zip`](Traning%20Data%20of%20UNISON.zip) for continuity. It is not a verified complete rebuild dataset; see [its limitations](docs/legacy_data.md).

## 1. Run a pretrained model locally

This route needs **no hosted endpoint**. Download this repository, or use **Code → Download ZIP** on GitHub and extract it. Use Python 3.13 and the pinned versions in the root [`requirements.txt`](requirements.txt). Run the following commands from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\verify_release.py
.\.venv\Scripts\python.exe offline\predict.py --model models\Bare_Aqua\model_bundle.pkl --input examples\Bare_Aqua.csv --output result.csv --lst-column modisLST --lst-unit celsius --output-kelvin
```

On macOS/Linux, create the environment with `python3.13 -m venv .venv`, install with `.venv/bin/python -m pip install -r requirements.txt`, and use `/` in paths. The example CSV is synthetic and checks the software format only. Replace it with your own prepared observations for research use.

`result.csv` includes `deltaLST_K`, `prediction_status`, `nadir_LST_C`, and `nadir_LST_K`. For another model, replace both the model path and example/input CSV. To prepare your own CSV, run `python offline/inspect_model.py --model models/Bare_Aqua/model_bundle.pkl --template my_input.csv`, fill all required predictors, and read the [local inference guide](offline/README.md). The application selects CSV columns in the bundle's saved order; the input file's column order may differ.

### Input and output contract

For the released bundles, absolute-temperature inputs `modisLST`, `DSAT`, and `DSOIL` are in **degrees Celsius**. For raw MODIS LST digital numbers, apply the source product's scale factor first; where the scale is `0.02 K`, `modisLST_C = raw × 0.02 − 273.15`. Other predictors must retain their trained physical definitions and units. `modisAngle` is a **signed** viewing angle in degrees. Do not pre-scale input variables: the fitted input and output scalers are already saved in each bundle. Invalid or missing predictors should be masked, never replaced by zero.

```text
deltaLST = near-nadir LST − observed directional LST
nadir_LST_C = observed_modisLST_C + deltaLST
nadir_LST_K = nadir_LST_C + 273.15
```

The correction is a temperature **difference**; its numerical value is the same in K and °C. Do not interpret `deltaLST` as the final absolute LST. For a complete feature definition, including spatial and temporal preparation, read [`docs/input_contract.md`](docs/input_contract.md). Load `.pkl` files only from a trusted source: Python pickle/joblib deserialization can execute code.

### Worked example: `Bare_Aqua`

The exact feature order inside the supplied `Bare_Aqua` bundle is:

```text
LON, LAT, DSAT, DSR, DLR, DSOIL, DVSOIL, DTR, DEM, ELESD, SLOPE, modisLST, modisAngle
```

This is **13 bands; `NDWI` is not an input to this particular bundle**. Check [`models/Bare_Aqua/manifest.json`](models/Bare_Aqua/manifest.json) for units, source metadata, training ranges, and SHA-256. The Cloud example contains the same `.pkl` bytes, plus the author's `app.py`, `Dockerfile`, and container `requirements.txt`; its [readme](cloud/Bare_Aqua/README.md) shows how to run and deploy it. Other model IDs require their matching bundle and verified application code/configuration.

## 2. Author-hosted online inference (planned)

**Current status: `not_deployed`. There is no usable public prediction endpoint or Earth Engine script link yet.** The author will deploy and maintain an endpoint and publish a tested access path here when it is ready. Until then, use the offline route above. The reserved [`online/endpoint_registry.json`](online/endpoint_registry.json) entries have `null` endpoint and script URL fields, and [`online/gee_client.js`](online/gee_client.js) deliberately stops before making a cloud call.

When activated, a user will (1) select the appropriate model ID, (2) prepare an `ee.Image` with the exact bands, units, masks, and order described in its manifest, (3) use the published Earth Engine client/link, and (4) inspect the returned `pred_deltaLST` and `correctedLST`. The online wrapper applies the same saved scalers as the local route; GEE inputs must remain in raw physical units. The final corrected LST is `modisLST + pred_deltaLST` in °C; add `273.15` only if a Kelvin product is needed.

An active endpoint and a shared Code Editor URL alone do not grant inference access. Direct `ee.Model.fromVertexAi` callers need eligible Earth Engine access and permission on the host Google Cloud project, including the role specified in [Google's Earth Engine hosted-model guidance](https://developers.google.com/earth-engine/guides/ee-vertex-overview). The author must confirm IAM and billing arrangements for the intended audience. Hosted prediction and other Google Cloud components can incur charges, and service availability depends on the author's deployment. The [online guide](online/README.md) documents these conditions; the [Bare Aqua owner deployment guide](cloud/Bare_Aqua/README.md) gives the setup steps. No current public link should be described as active before end-to-end verification.

## 3. Build a new `.pkl` bundle

The [`training/README.md`](training/README.md) summarizes the scientific steps: conditional multi-angle LST differences, a near-nadir reference, surface-state and signed-view-angle bins, response smoothing, and separate surface-group/sensor models. The included helper implements the **final fit and export step** once a complete labelled CSV has been prepared. It does not regenerate the full upstream GEE sampling, original labels, or the six released files byte for byte.

For example, after preparing a table with every `Bare_Aqua` feature and a `deltaLST` target in K differences:

```powershell
.\.venv\Scripts\python.exe training\train_from_csv.py --csv prepared_training.csv --reference-bundle models\Bare_Aqua\model_bundle.pkl --temperature-unit celsius --output my_bare_aqua_model.pkl
```

The helper records feature order, fitted transformations, model, metadata, library versions, and a training report. The provided `--reference-bundle` supplies the schema; it does **not** reuse that model or its fitted scalers. Read the [training guide](training/README.md) before interpreting its split metrics or claiming full scientific reproduction.

## Reproducibility and reporting

The software checks verify file integrity, input handling, and the local formula; they do not replace the manuscript's scientific validation. Cite the accompanying UNISON paper when using the method, and report the model ID, bundle/checksum or release version, input product definitions, observation period, and whether inference was local or hosted. The model manifests contain the supplied bundle metadata; they do not encode every upstream preprocessing choice.

The previous public GEE scripts and training-data archive are described in [`docs/legacy_data.md`](docs/legacy_data.md). For this release, the durable public landing page is the [UNISON GitHub repository](https://github.com/ginkgyh/UNISON-A-Data-Driven-Angular-Normalization-Framework-for-MODIS-LST).
