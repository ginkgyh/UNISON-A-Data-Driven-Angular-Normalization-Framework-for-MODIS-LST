import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request


MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "model_bundle.pkl"
)


# ---------------------------------------------------------
# 1. 程序启动时加载一次模型
# ---------------------------------------------------------
bundle = joblib.load(MODEL_PATH)

required_keys = [
    "lgb_model",
    "x_std_transformer",
    "y_std_transformer",
    "feature_names",
]

missing_keys = [
    key for key in required_keys
    if key not in bundle
]

if missing_keys:
    raise RuntimeError(
        f"Model bundle is missing keys: {missing_keys}"
    )


model = bundle["lgb_model"]
x_scaler = bundle["x_std_transformer"]
y_scaler = bundle["y_std_transformer"]
feature_names = list(bundle["feature_names"])

n_features = len(feature_names)


app = FastAPI(
    title="UNISON LightGBM deltaLST correction model"
)


# ---------------------------------------------------------
# 2. Vertex AI用这个地址判断容器是否正常
# ---------------------------------------------------------
@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features": n_features,
        "feature_names": feature_names,
        "sensor_type": bundle.get("sensor_type"),
        "surface_type": bundle.get("surface_type"),
        "area_mc": bundle.get("area_mc"),
        "layer": bundle.get("layer"),
    }


# ---------------------------------------------------------
# 3. 整理单个输入波段
# ---------------------------------------------------------
def normalize_band(value: Any) -> np.ndarray:
    """
    GEE影像预测时，每个输入波段一般为：
        height × width × 1

    LightGBM最终需要：
        pixel_count × feature_count
    """

    array = np.asarray(value, dtype=np.float32)

    # 去掉最后一个长度为1的波段维度
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]

    # 为了便于手工测试，也接受单个标量
    if array.ndim == 0:
        array = array.reshape(1, 1)

    if array.ndim != 2:
        raise ValueError(
            "Each input band must have shape "
            f"[height, width, 1] or [height, width]. "
            f"Received shape: {array.shape}"
        )

    return array


# ---------------------------------------------------------
# 4. 解析一个GEE影像瓦片
# ---------------------------------------------------------
def parse_instance(instance: Any) -> list[np.ndarray]:
    """
    支持两种输入：

    1. ND_ARRAYS：
       不包含波段名称，严格依赖GEE的波段顺序。

    2. RAW_JSON：
       使用字典和波段名称，适合小规模调试。
    """

    if isinstance(instance, list):
        if len(instance) != n_features:
            raise ValueError(
                f"Model requires {n_features} bands, "
                f"but received {len(instance)}."
            )

        raw_bands = instance

    elif isinstance(instance, dict):
        missing = [
            name for name in feature_names
            if name not in instance
        ]

        if missing:
            raise ValueError(
                f"Missing feature bands: {missing}"
            )

        raw_bands = [
            instance[name]
            for name in feature_names
        ]

    else:
        raise ValueError(
            "Each instance must be a list or dictionary."
        )

    bands = [
        normalize_band(value)
        for value in raw_bands
    ]

    reference_shape = bands[0].shape

    for name, band in zip(feature_names, bands):
        if band.shape != reference_shape:
            raise ValueError(
                f"Band {name} has shape {band.shape}, "
                f"but expected {reference_shape}."
            )

        if not np.all(np.isfinite(band)):
            raise ValueError(
                f"Band {name} contains NaN or infinity."
            )

    return bands


# ---------------------------------------------------------
# 5. 完整预测：缩放X → LightGBM → 反缩放y
# ---------------------------------------------------------
def predict_tile(instance: Any) -> np.ndarray:
    bands = parse_instance(instance)

    height, width = bands[0].shape

    # H × W × F
    feature_cube = np.stack(
        bands,
        axis=-1
    )

    # (H×W) × F
    X_raw = feature_cube.reshape(
        -1,
        n_features
    )

    # 必须与训练阶段完全一致
    X_scaled = x_scaler.transform(X_raw)

    pred_scaled = model.predict(X_scaled)

    pred_delta_lst = y_scaler.inverse_transform(
        np.asarray(pred_scaled).reshape(-1, 1)
    ).ravel()

    # 返回 H × W × 1
    return pred_delta_lst.reshape(
        height,
        width,
        1
    ).astype(np.float32)


# ---------------------------------------------------------
# 6. Vertex AI真正调用的预测地址
# ---------------------------------------------------------
@app.post("/predict")
async def predict(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()

        instances = payload.get("instances")

        if not isinstance(instances, list):
            raise ValueError(
                "Request must contain an instances list."
            )

        if len(instances) == 0:
            raise ValueError(
                "The instances list is empty."
            )

        predictions = []

        for instance in instances:
            prediction_tile = predict_tile(instance)

            # 一个instance对应一个预测瓦片
            predictions.append(
                prediction_tile.tolist()
            )

        return {
            "predictions": predictions
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Internal prediction error: "
                f"{type(exc).__name__}: {exc}"
            )
        ) from exc