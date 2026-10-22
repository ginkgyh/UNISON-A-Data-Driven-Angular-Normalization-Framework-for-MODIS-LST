# Trained model bundles

Six original author-provided bundles are included. Select the sensor and surface group explicitly; no automatic land-cover classification is assumed.

The [registry](registry.json) lists relative paths and feature counts. Each directory contains `model_bundle.pkl`, `manifest.json` and `input_template.csv`. Manifests record ordered input names, units, fitted input ranges, original metadata and SHA-256 hashes. The bundles retain the original bytes provided for this release.

Temperature features `modisLST`, `DSAT`, `DSOIL` are **Celsius**. The target is a temperature difference. Apply the bundled scalers, then add the difference to observed LST; see the [input contract](../docs/input_contract.md) and [offline guide](../offline/README.md).

This release is documented for the paper's summer daytime natural-surface domain. Original metadata does not encode season. Load trusted pickle files only.
