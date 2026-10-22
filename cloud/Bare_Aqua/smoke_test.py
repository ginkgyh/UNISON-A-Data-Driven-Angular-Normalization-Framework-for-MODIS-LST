"""Check the Bare Aqua HTTP container against its own saved bundle.

Run this only with the trusted bundle in this directory. The synthetic test
input is the midpoint of each fitted scaler range, not a science benchmark.
"""

import argparse
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--request-file", type=Path)
    args = parser.parse_args()

    bundle = joblib.load(Path(__file__).with_name("model_bundle.pkl"))
    names = list(bundle["feature_names"])
    scaler = bundle["x_std_transformer"]
    x = ((scaler.data_min_ + scaler.data_max_) / 2).astype(np.float32)

    with urlopen(f"{args.url.rstrip('/')}/health", timeout=30) as response:
        health = json.load(response)
    if health.get("status") != "ok" or health.get("feature_names") != names:
        raise AssertionError("Health check or feature order mismatch")

    # One 1x1 tile: a list of bands in the model's exact feature order.
    payload = {"instances": [[[[[float(value)]]] for value in x]]}
    if args.request_file:
        args.request_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    request = Request(
        f"{args.url.rstrip('/')}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    actual = float(result["predictions"][0][0][0][0])
    x_scaled = scaler.transform(x.reshape(1, -1))
    raw = bundle["lgb_model"].predict(x_scaled)
    expected = float(bundle["y_std_transformer"].inverse_transform(
        np.asarray(raw).reshape(-1, 1)
    )[0, 0])
    if not math.isfinite(actual) or not np.isclose(actual, expected, atol=1e-5):
        raise AssertionError(f"API {actual} != direct bundle {expected}")
    print(f"PASS: {len(names)} bands, deltaLST={actual:.6f} K")


if __name__ == "__main__":
    main()
