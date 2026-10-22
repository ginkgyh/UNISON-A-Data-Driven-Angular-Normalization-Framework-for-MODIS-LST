# Bare Aqua: custom-container and Earth Engine deployment example

**Service status: not deployed.** This directory is an author-owned deployment example, not a live public endpoint. Readers who only want predictions can use the [pretrained local route](../../README.md#1-run-a-pretrained-model-locally). The owner can follow the steps below, validate the cloud result, configure access, and then publish a usable Earth Engine entry point.

This example copies the supplied `Vertex_LGB_Bare_Aqua` application files: `app.py`, `Dockerfile`, `requirements.txt`, and `model_bundle.pkl`. It adds `smoke_test.py` for a portable local/API check. The source ZIP duplicates the same files and is omitted here. Do **not** upload `.venv`, caches, credentials, local CSVs, or the original `test_local.py` with its author-specific absolute path.

## 0. Understand the contract

The FastAPI container loads the trusted bundle once, serves `GET /health` and `POST /predict`, and listens on **port 8081**. Its `/predict` body has an `instances` array; each instance is either an ordered list of 13 bands or a name-to-band dictionary. For Earth Engine `ND_ARRAYS`, each band has shape `[height, width, 1]`. The response is `{"predictions": [tile, ...]}`, where each tile has one `deltaLST` value per pixel. `app.py` performs the saved input scaling, LightGBM inference, and inverse output scaling.

The **exact** `Bare_Aqua` order is:

```text
LON, LAT, DSAT, DSR, DLR, DSOIL, DVSOIL, DTR, DEM, ELESD, SLOPE, modisLST, modisAngle
```

There are 13 features; this bundle does **not** contain `NDWI`. The model input `modisLST`, `DSAT`, and `DSOIL` must be in °C. `modisAngle` is signed degrees. Predicted `deltaLST` is near-nadir minus observed LST; corrected LST in °C is `modisLST + deltaLST`. Mask missing/invalid predictors before inference. Check the [manifest](../../models/Bare_Aqua/manifest.json) and [input contract](../../docs/input_contract.md) before assembling a GEE image. The `.pkl` in this directory and the `models/Bare_Aqua` copy should have SHA-256 `c2fecd2839521d39ac85fae8e452379b0c8cce38155466212bde467749b6baef`.

## 1. Validate the published files locally

In Windows PowerShell, start from the repository root:

```powershell
cd .\cloud\Bare_Aqua
Get-FileHash .\model_bundle.pkl -Algorithm SHA256
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8081
```

Keep that terminal open. In a **second** PowerShell window, from the same directory:

```powershell
.\.venv\Scripts\python.exe smoke_test.py --request-file request.json
```

Expect `PASS: 13 bands, deltaLST=... K`. The script checks `/health`, the exact feature order, and equality of the API result to direct inference from this bundle (within `1e-5`). Its generated 1×1 request is a **synthetic software test**, not an observed MODIS pixel or accuracy validation. `request.json` can be reused for cloud endpoint testing. The root `.gitignore` should keep `.venv` out of a Git commit; remove local `request.json` from any browser upload selection.

Optional Docker check from this directory:

```powershell
docker build -t unison-bare-aqua:local .
docker run --rm -p 8081:8081 unison-bare-aqua:local
```

Then run `smoke_test.py` in a second terminal. The Dockerfile and model upload in this example must use **8081** throughout. Generic port values from other examples do not apply.

## 2. Prepare a billable Google Cloud project

Install/sign in to the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), select a project with billing enabled, and confirm that your account has the permissions to enable services, create an Artifact Registry repository, run Cloud Build, upload models, and manage endpoints. Check quota and choose a supported region; `us-central1` is used below because Google's [Earth Engine hosted-model guide](https://developers.google.com/earth-engine/guides/ee-vertex-overview) recommends it for proximity to Earth Engine. Replace **every** angle-bracket value with your own IDs. The sample uses no private IDs or keys.

```powershell
$ProjectId = '<PROJECT_ID>'
$Region = 'us-central1'
$Repo = 'unison-serving'
$Image = 'bare-aqua-summer'
$Tag = 'v1'
gcloud auth login
gcloud config set project $ProjectId
gcloud services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project=$ProjectId
```

## 3. Build the image and register the model

Create a Docker repository once, or reuse one you control in the **same region**. Run the build from this directory so the Dockerfile can copy its adjacent `app.py`, `requirements.txt`, and `model_bundle.pkl`.

```powershell
gcloud artifacts repositories create $Repo --repository-format=docker --location=$Region --project=$ProjectId --description='UNISON serving images'
$ImageUri = '{0}-docker.pkg.dev/{1}/{2}/{3}:{4}' -f $Region,$ProjectId,$Repo,$Image,$Tag
gcloud builds submit . --region=$Region --tag=$ImageUri --project=$ProjectId
gcloud ai models upload --region=$Region --project=$ProjectId --display-name='UNISON-Bare-Aqua-summer' --container-image-uri=$ImageUri --container-ports=8081 --container-health-route=/health --container-predict-route=/predict
```

If the repository already exists, skip the create command. If Cloud Build reports a permissions error, fix the build service account's access to the image repository as described in the [Cloud Build build-and-push guide](https://docs.cloud.google.com/build/docs/build-push-docker-image); do not paste credentials into GitHub. Record the **uploaded Model ID** from the command output or the Google Cloud console. Building/uploading a model does not create an active endpoint.

## 4. Create and deploy an endpoint

Create a **public prediction endpoint** in the same project and region; the GEE connector uses its resource name. Record the **Endpoint ID** from the result. Deployment starts billable serving resources. The single CPU replica below mirrors the supplied workflow; choose a machine and replica count after checking current pricing, quota, and expected load.

```powershell
gcloud ai endpoints create --region=$Region --project=$ProjectId --display-name='UNISON-Bare-Aqua-endpoint'
$ModelId = '<MODEL_ID_FROM_UPLOAD>'
$EndpointId = '<ENDPOINT_ID_FROM_CREATE>'
gcloud ai endpoints deploy-model $EndpointId --region=$Region --project=$ProjectId --model=$ModelId --display-name='UNISON-Bare-Aqua-v1' --machine-type=n1-standard-2 --min-replica-count=1 --max-replica-count=1 --traffic-split=0=100
gcloud ai endpoints describe $EndpointId --region=$Region --project=$ProjectId
```

Use the same request created in step 1 to test the **actual endpoint**:

```powershell
gcloud ai endpoints predict $EndpointId --region=$Region --project=$ProjectId --json-request=request.json
```

Compare its one-pixel `predictions` value with the local smoke-test output. A working `/health` and successful deployment alone do not prove the GEE image contract. Check logs and the deployed model state if the request fails.

## 5. Connect Earth Engine and publish only after validation

1. In [`online/gee_client.js`](../../online/gee_client.js), set `CFG.MODEL_ID` to `Bare_Aqua` and replace its `MODEL_REGISTRY.Bare_Aqua` endpoint with `projects/<PROJECT_ID>/locations/us-central1/endpoints/<ENDPOINT_ID>`. Change status to `active` **only after** the endpoint test, GEE image test, and access check succeed. Put the same resource name/status and optional shared Code Editor URL in [`online/endpoint_registry.json`](../../online/endpoint_registry.json).
2. In Earth Engine, prepare a real `ee.Image` with the **13 exact bands** above in the saved order and raw physical units. Apply source scale factors, convert required absolute temperatures to °C, preserve a signed `modisAngle`, and mask missing values. The client selects the bands and sends `ND_ARRAYS` tiles of 64×64 pixels to `ee.Model.fromVertexAi`. Call `unisonPredict(preparedPredictorImage)`. It returns `pred_deltaLST` and `correctedLST` (°C). Start with a small area and compare representative pixels against local inference before scaling up.
3. Grant intended users the required Earth Engine access and at least the **Vertex AI User** role on the hosting Cloud project, as specified by [Google's hosted-model guide](https://developers.google.com/earth-engine/guides/ee-vertex-overview). Test with an intended external account. A GitHub or Code Editor URL alone cannot bypass IAM. Document who pays for hosting and cloud usage. Keep a nonworking/unavailable service marked `not_deployed` or `inactive` in the public registry.
4. Publish the tested Earth Engine Code Editor share URL (if you choose to share one), service region, access instructions, and update date in `online/README.md` and `online/endpoint_registry.json`. The `gee_client.js` file is a **template**: it does not generate the scientific predictor image or grant the reader access to your private assets.

To stop ongoing serving charges, undeploy the model when no longer needed and change the public registry/status at once. A later redeployment can use a new endpoint or model version; recheck the client and access every time.

## References

- [Earth Engine predictions from hosted models, IAM, regions, and charges](https://developers.google.com/earth-engine/guides/ee-vertex-overview)
- [`ee.Model.fromVertexAi` argument reference](https://developers.google.com/earth-engine/apidocs/ee-model-fromvertexai)
- [Cloud Build: build and push a Docker image](https://docs.cloud.google.com/build/docs/build-push-docker-image)
- [`gcloud ai models upload` flags](https://docs.cloud.google.com/sdk/gcloud/reference/ai/models/upload)
- [Endpoint creation and model deployment](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/predictions/deploy-model-api)
- [`gcloud ai endpoints predict` request format](https://docs.cloud.google.com/sdk/gcloud/reference/ai/endpoints/predict)
