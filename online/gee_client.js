/* UNISON optional Earth Engine client template.
 * Current release: NOT DEPLOYED. A deliberate guard below prevents cloud calls.
 * Use offline inference until the owner publishes a validated endpoint.
 *
 * This template takes an already prepared, masked predictor image in raw
 * physical units and exact training definitions. It does not construct data,
 * rescale raw MODIS integers, or depend on the owner's Earth Engine assets.
 * DSAT, DSOIL and modisLST must be degrees Celsius for these released bundles.
 * Do not supply Kelvin or raw MODIS integer LST. Other units/definitions must
 * match the released model feature contract.
 */

// The owner must keep these entries synchronized with endpoint_registry.json.
var MODEL_REGISTRY = {
  "Bare_Aqua": {
    "status": "not_deployed",
    "endpoint": null,
    "code_url": null,
    "feature_names": [
      "LON",
      "LAT",
      "DSAT",
      "DSR",
      "DLR",
      "DSOIL",
      "DVSOIL",
      "DTR",
      "DEM",
      "ELESD",
      "SLOPE",
      "modisLST",
      "modisAngle"
    ]
  },
  "Bare_Terra": {
    "status": "not_deployed",
    "endpoint": null,
    "code_url": null,
    "feature_names": [
      "LON",
      "LAT",
      "DSAT",
      "DSR",
      "DLR",
      "DSOIL",
      "DVSOIL",
      "NDWI",
      "DTR",
      "DEM",
      "ELESD",
      "SLOPE",
      "modisLST",
      "modisAngle"
    ]
  },
  "MHVeg_Aqua": {
    "status": "not_deployed",
    "endpoint": null,
    "code_url": null,
    "feature_names": [
      "LON",
      "LAT",
      "DSAT",
      "DSR",
      "DLR",
      "DSOIL",
      "DVSOIL",
      "DTR",
      "DEM",
      "ELESD",
      "SLOPE",
      "modisLST",
      "modisAngle",
      "LAI",
      "EVI",
      "GPP",
      "NDWI"
    ]
  },
  "MHVeg_Terra": {
    "status": "not_deployed",
    "endpoint": null,
    "code_url": null,
    "feature_names": [
      "LON",
      "LAT",
      "DSAT",
      "DSR",
      "DLR",
      "DSOIL",
      "DVSOIL",
      "NDWI",
      "DTR",
      "DEM",
      "ELESD",
      "SLOPE",
      "modisLST",
      "modisAngle",
      "LAI",
      "EVI",
      "GPP"
    ]
  },
  "MLVeg_Aqua": {
    "status": "not_deployed",
    "endpoint": null,
    "code_url": null,
    "feature_names": [
      "LON",
      "LAT",
      "DSAT",
      "DSR",
      "DLR",
      "DSOIL",
      "DVSOIL",
      "DTR",
      "DEM",
      "ELESD",
      "SLOPE",
      "modisLST",
      "modisAngle",
      "LAI",
      "EVI",
      "TH",
      "GPP",
      "NDWI"
    ]
  },
  "MLVeg_Terra": {
    "status": "not_deployed",
    "endpoint": null,
    "code_url": null,
    "feature_names": [
      "LON",
      "LAT",
      "DSAT",
      "DSR",
      "DLR",
      "DSOIL",
      "DVSOIL",
      "NDWI",
      "DTR",
      "DEM",
      "ELESD",
      "SLOPE",
      "modisLST",
      "modisAngle",
      "LAI",
      "EVI",
      "TH",
      "GPP"
    ]
  }
};

var CFG = {
  MODEL_ID: 'Bare_Terra',
  // Mask the source's numeric NoData codes before calling this client, or list
  // them here. Leave empty only when NoData is already represented by masks.
  NODATA_VALUES: [],
  TILE_SIZE: [64, 64]
};

var selected = MODEL_REGISTRY[CFG.MODEL_ID];
if (!selected) {
  throw new Error('Unknown MODEL_ID: ' + CFG.MODEL_ID);
}
if (selected.status !== 'active' || !selected.endpoint) {
  throw new Error(
    'UNISON online inference is not deployed for ' + CFG.MODEL_ID +
    '. Use offline inference; an active endpoint must be published by the owner.'
  );
}
if (!/^projects\/[^/]+\/locations\/[^/]+\/endpoints\/[^/]+$/.test(selected.endpoint)) {
  throw new Error('Expected a projects/.../locations/.../endpoints/... resource name.');
}

// The guard above runs before any ee.Model construction or prediction request.
var cloudModel = ee.Model.fromVertexAi({
  endpoint: selected.endpoint,
  inputTileSize: CFG.TILE_SIZE,
  inputOverlapSize: [0, 0],
  outputTileSize: CFG.TILE_SIZE,
  outputBands: {
    pred_deltaLST: {type: ee.PixelType.float(), dimensions: 0}
  },
  payloadFormat: 'ND_ARRAYS',
  maxPayloadBytes: 1500000
});

function unisonPredict(preparedImage) {
  if (!preparedImage) {
    throw new Error('Provide a prepared ee.Image with all required predictor bands.');
  }
  // select() enforces both names and order. A missing band is an error.
  var x = ee.Image(preparedImage).select(selected.feature_names).toFloat();
  var valid = x.mask().reduce(ee.Reducer.min());

  // NaN fails equality to itself; infinity exceeds the float32 finite limit.
  // Keep source masks and require every predictor to be finite.
  var finite = x.eq(x).and(x.abs().lte(3.4028234663852886e38))
    .reduce(ee.Reducer.min());
  valid = valid.and(finite);
  CFG.NODATA_VALUES.forEach(function(nodata) {
    valid = valid.and(x.neq(nodata).reduce(ee.Reducer.min()));
  });
  x = x.updateMask(valid);

  // The hosted wrapper must apply the saved x scaler, regressor, and inverse
  // y scaler, returning one deltaLST scalar per pixel (Kelvin difference).
  var delta = cloudModel.predictImage(x).select('pred_deltaLST');
  var finiteDelta = delta.eq(delta).and(delta.abs().lte(3.4028234663852886e38));
  delta = delta.updateMask(valid.and(finiteDelta)).rename('pred_deltaLST');
  var corrected = x.select('modisLST').add(delta).rename('correctedLST');

  // deltaLST = correctedLST - observed modisLST; correctedLST is nadir LST.
  // correctedLST is in degrees Celsius. For Kelvin, use:
  // corrected.add(273.15).rename('correctedLST_K').
  // Return the model response unchanged, including at modisAngle = 0.
  return delta.addBands(corrected).copyProperties(preparedImage, ['system:time_start']);
}

// After the owner publishes an active entry and grants access, supply your
// prepared predictor image and call:
// var result = unisonPredict(preparedPredictorImage);
// Map.addLayer(result.select('correctedLST'), {min: 0, max: 60}, 'UNISON nadir LST (C)');
