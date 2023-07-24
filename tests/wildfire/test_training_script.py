from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from train_wildfire_ignition_model import build_demo_dataset, train_model
from wildfire_service.model_bundle import ModelBundle


def test_training_pipeline_smoke(tmp_path):
    frame = build_demo_dataset(rows=180, seed=7)
    bundle, metrics = train_model(frame)

    output = tmp_path / "model_bundle.joblib"
    bundle.metadata["model_version"] = "smoke"
    bundle.save(output)

    loaded = ModelBundle.load(output)
    assert loaded.model_version == "smoke"
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] >= 0.0


def test_demo_dataset_has_required_target():
    frame = build_demo_dataset(rows=30, seed=1)
    assert "ignition_next_24h" in frame.columns
    assert isinstance(frame, pd.DataFrame)
