"""Signal model artifact contract v1.0.0, consumer side (AC-H-10).

Copyable to market-making-sim: it uses only lightgbm, json, csv, hashlib and the standard
library. For every committed version folder under artifacts/models it checks the files,
the SHA-256 values, the LightGBM major.minor and the feature order, then reproduces
parity_sample.csv's predictions from its normalised features within 1e-9 absolute, and the
normalised features from the raw ones by the spec's normalisation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2] / "artifacts" / "models"
FOLDERS = sorted(p.parent for p in ROOT.glob("*/*/metadata.json"))
TOL = 1e-9


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not FOLDERS, reason="no committed artifact")
@pytest.mark.parametrize("folder", FOLDERS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_artifact_contract(folder: Path) -> None:
    import lightgbm as lgb

    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    spec = json.loads((folder / "feature_spec.json").read_text(encoding="utf-8"))
    assert spec["spec_version"].split(".")[0] == "1"
    assert spec["model_name"] == folder.parent.name
    assert spec["model_version"] == folder.name
    assert _sha(folder / "model.txt") == meta["model_sha256"]
    for name in ("feature_spec.json", "parity_sample.csv"):
        assert _sha(folder / name) == meta["files_sha256"][name]
    assert lgb.__version__.split(".")[:2] == meta["lightgbm_version"].split(".")[:2]
    assert meta["valid_end"] < min(meta.get("test_days", [meta["valid_end"] + "~"]))
    booster = lgb.Booster(model_file=str(folder / "model.txt"))
    names = [f["name"] for f in spec["features"]]
    assert booster.feature_name() == names

    with (folder / "parity_sample.csv").open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) >= 1000
    norm_rows = []
    for r in rows:
        norm = []
        for f in spec["features"]:
            raw = float(r[f["name"]])
            nm = f["normalization"]
            if nm["method"] == "zscore_clip":
                z = (raw - nm["mean"]) / nm["std"]
                want = min(max(z, -nm["clip"]), nm["clip"])
            else:
                want = raw
            got = float(r[f["name"] + "__norm"])
            assert math.isclose(got, want, rel_tol=TOL, abs_tol=TOL)
            norm.append(got)
        norm_rows.append(norm)
    pred = booster.predict(norm_rows)
    for p, r in zip(pred, rows, strict=True):
        assert abs(float(p) - float(r["prediction"])) <= TOL
