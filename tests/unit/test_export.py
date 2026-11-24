"""Model artifact round trip: parity predictions, SHA-256 values, schema (AC-W-11)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from lob_alpha import export
from lob_alpha.models.train import Runner, RunSpec

sys.path.insert(0, str(Path(__file__).parents[1] / "fixtures"))
from make_panels import FEATURES, SYMBOL, TARGET, VENUE, make_days

SCHEMA = Path(__file__).parents[2] / "docs" / "feature_spec.schema.json"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    root = tmp_path_factory.mktemp("export")
    days = make_days(root, date(2026, 4, 6), 7)
    cfg = {
        "n_estimators": 30,
        "early_stopping_rounds": 10,
        "num_threads": 1,
        "min_data_in_leaf": 50,
        "num_leaves": 7,
    }
    spec = RunSpec(VENUE, SYMBOL, TARGET, "lgbm", cfg, list(FEATURES), days, 0.1, 5.0)
    res = Runner(spec, root, root / "results").run()
    out = root / "artifacts"
    r = export.export_model(
        run_id=res.run_id,
        results=root / "results",
        data_root=root,
        out_root=out,
        version="0.0.0-pre.20260412",
        tick=0.1,
        lot=0.001,
        extra_metadata={"provisional": True},
    )
    assert r.max_abs_diff <= 1e-12
    return root, r.folder, res.run_id


def test_round_trip_and_schema(artifact: tuple[Path, Path, str]) -> None:
    _, folder, _ = artifact
    assert sorted(p.name for p in folder.iterdir()) == [
        "feature_spec.json",
        "metadata.json",
        "model.txt",
        "parity_sample.csv",
    ]
    assert export.verify_artifact(folder, SCHEMA) == []
    spec = json.loads((folder / "feature_spec.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(spec)
    assert spec["model_name"] == f"lob-{VENUE}-{SYMBOL}-{TARGET}"
    assert [f["name"] for f in spec["features"]] == FEATURES
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    assert meta["provisional"] is True
    assert meta["valid_end"] < meta["test_days"][0]
    assert meta["train_end"] < meta["valid_start"]
    names, raw, norm, pred = export.read_parity(folder)
    assert names == FEATURES
    assert len(pred) >= 1000 and not np.isnan(raw).any()
    model = export.load_model(folder)
    assert np.max(np.abs(model.predict(norm) - pred)) <= 1e-9
    micro = next(f for f in spec["features"] if f["kind"] == "microprice")
    assert micro["params"]["table"][0][:2] == [1.0, 0.0]


def test_tamper_and_immutability(artifact: tuple[Path, Path, str], tmp_path: Path) -> None:
    root, folder, run_id = artifact
    with pytest.raises(FileExistsError):
        export.export_model(
            run_id=run_id,
            results=root / "results",
            data_root=root,
            out_root=folder.parents[1],
            version=folder.name,
            tick=0.1,
            lot=0.001,
        )
    copy = tmp_path / "m" / "1.0.0"
    copy.mkdir(parents=True)
    for p in folder.iterdir():
        (copy / p.name).write_bytes(p.read_bytes())
    (copy / "parity_sample.csv").write_bytes((copy / "parity_sample.csv").read_bytes() + b"\n")
    assert any("SHA-256" in p for p in export.verify_artifact(copy, SCHEMA))


def test_ridge_run_cannot_be_exported(artifact: tuple[Path, Path, str]) -> None:
    root, _, _ = artifact
    days = sorted(
        date.fromisoformat(d.name[5:])
        for d in (root / "processed/features" / f"venue={VENUE}" / f"symbol={SYMBOL}").iterdir()
    )
    spec = RunSpec(
        VENUE, SYMBOL, TARGET, "ridge", {"alphas": [1.0]}, list(FEATURES), days, 0.1, 5.0
    )
    res = Runner(spec, root, root / "results").run()
    with pytest.raises(ValueError, match="LightGBM"):
        export.export_model(
            run_id=res.run_id,
            results=root / "results",
            data_root=root,
            out_root=root / "artifacts",
            version="9.9.9",
            tick=0.1,
            lot=0.001,
        )
