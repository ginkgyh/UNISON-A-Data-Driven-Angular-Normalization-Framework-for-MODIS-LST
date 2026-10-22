# Optional online inference

**Status: `not_deployed`. No online endpoint or Earth Engine share link is currently published.** The entries in [endpoint_registry.json](endpoint_registry.json) reserve this option for a later owner deployment. Use the pretrained models and local application code for inference now. 当前线上服务尚未部署，建议使用离线模型。

The owner will host the trained UNISON models and publish their endpoint resource names and optional Earth Engine Code Editor links. Users will call those models without building the training pipeline or deploying their own backend. Service availability depends on the owner's active deployment, quotas, maintenance, and cloud costs; continuous availability is not guaranteed.

Google Cloud documentation now calls this product **Gemini Enterprise Agent Platform**. The Earth Engine connector and prediction API still use `ee.Model.fromVertexAi` and `aiplatform.googleapis.com`; these identifiers are used by this client. See [Google's custom model inference documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/get-online-predictions).

## Access and charges

A published endpoint name or Code Editor link does not itself authorize inference. Every caller using `ee.Model.fromVertexAi` needs access to Earth Engine and sufficient IAM permissions in the cloud project hosting the model. Google's Earth Engine guide specifies the Vertex AI User role for that project. The owner must configure and verify caller access before publishing an active entry. See [Earth Engine hosted models and endpoint IAM permissions](https://developers.google.com/earth-engine/guides/ee-vertex-overview).

The service uses billable Google Cloud resources. Hosting and prediction charges apply according to the hosting project's billing configuration, while callers may incur costs for their own Earth Engine or other cloud operations. This is not a requirement for every person to purchase an individual Agent Platform subscription. Consult [Agent Platform pricing](https://cloud.google.com/products/gemini-enterprise-agent-platform/pricing) for current charges and [IAM access control](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/access-control) for permissions.

## Published client files

- `endpoint_registry.json` lists the six model IDs, exact predictor order, deployment status, and future endpoint and Code Editor link fields. All endpoint and code URL fields are currently `null`.
- `gee_client.js` is a minimal Google Earth Engine client template. Its deliberate guard stops before creating an `ee.Model` while the selected model is not deployed or its endpoint is missing. It contains no deployment tools or private Earth Engine asset references.

## Use after the owner activates the service

1. Select the matching `MODEL_ID` from the registry: `Bare_Terra`, `Bare_Aqua`, `MLVeg_Terra`, `MLVeg_Aqua`, `MHVeg_Terra`, or `MHVeg_Aqua`. Model choice must also match the verified training domain and season documented for that deployment.
2. Copy the published active entry into the corresponding `MODEL_REGISTRY` entry in `gee_client.js`. The owner must publish a valid `projects/.../locations/.../endpoints/...` resource name and change its status to `active` only after deployment, access, and local versus online agreement have been verified.
3. Supply your own prepared `ee.Image`. Its bands must use the exact names, order, units, spatial definitions, and temporal aggregation used for model training. The client selects names in the correct order. It does not generate predictors or apply satellite scale factors. **For these released bundles, `DSAT`, `DSOIL`, and `modisLST` must be in degrees Celsius.** Convert MODIS integer LST to `raw * 0.02 - 273.15` before constructing `modisLST`; do not pass Kelvin values. Radiation predictors use W m-2, coordinates decimal degrees, elevation and elevation variability meters, slope degrees, and signed `modisAngle` degrees. Surface state predictors must retain their training definitions.
4. Mask missing data and unsuitable pixels upstream. If your source uses finite numeric NoData sentinels, put those values in `CFG.NODATA_VALUES` or mask them before calling. The client keeps the intersection of predictor masks and excludes nonfinite values. A missing predictor band or absent image is an error; no input is silently filled with zero. Select a grid, projection, and pixel size consistent with the prepared predictors.
5. Call `unisonPredict(preparedPredictorImage)` after the configuration block. Returned bands are `pred_deltaLST` and `correctedLST`, with `correctedLST = modisLST + pred_deltaLST`. The correction is a temperature difference in K (numerically identical in degrees Celsius); the corrected nadir LST is in degrees Celsius. Add `273.15` to the final corrected LST only when a Kelvin output is desired. The raw model response is preserved, including at a view angle of zero.

The hosted wrapper is expected to apply the saved input scaler, LightGBM model, and inverse output scaler. The client supplies raw physical predictors; callers must not scale them a second time. Its `ND_ARRAYS` request contains each predictor as a `[height, width, 1]` tile, in registry order, and expects one scalar `pred_deltaLST` per pixel. See the [Earth Engine connector API](https://developers.google.com/earth-engine/apidocs/ee-model-fromvertexai) for current payload and tiling options.

The template's syntax and disabled guard can be checked locally. An end to end cloud result cannot be validated before the owner deploys the service and grants IAM access.
