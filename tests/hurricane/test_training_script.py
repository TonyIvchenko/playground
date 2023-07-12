from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from hurricane_service.model_bundle import ModelBundle
from train_hurricane_intensity_model import build_demo_dataset, train_models


def test_training_pipeline_smoke(tmp_path):
    frame = build_demo_dataset(rows=120, seed=10)
    bundle, metrics = train_models(frame)

    output = tmp_path / "model_bundle.joblib"
    bundle.metadata["model_version"] = "smoke"
    bundle.save(output)

    loaded = ModelBundle.load(output)
    assert loaded.model_version == "smoke"
    assert "ri_roc_auc" in metrics
    assert metrics["ri_roc_auc"] >= 0.0


def test_demo_dataset_has_required_targets():
    frame = build_demo_dataset(rows=20, seed=1)
    assert "ri_next_24h" in frame.columns
    assert "vmax_tplus24_kt" in frame.columns
    assert "vmax_tplus48_kt" in frame.columns
    assert isinstance(frame, pd.DataFrame)
