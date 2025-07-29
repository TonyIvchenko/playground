import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient
import gradio as gr

from src.disasters.main import HURRICANES_LABEL, api, demo
from src.disasters.main import (
    HURICAINES_MODEL_VERSION,
    WILDFIRES_MODEL_VERSION,
    _toggle_model_panel,
    predict_huricaines,
    predict_wildfires,
)


def test_health_endpoint_contract():
    client = TestClient(api)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "Disasters"


def test_predict_wildfires_shape_and_values():
    result = predict_wildfires(
        region_id="norcal",
        temp_c=34,
        humidity_pct=22,
        wind_kph=28,
        ffmc=92.0,
        dmc=180.0,
        drought_code=640.0,
        isi=12.0,
    )
    assert result["region_id"] == "norcal"
    assert result["model_version"] == WILDFIRES_MODEL_VERSION
    assert 0.0 <= result["ignition_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}


def test_predict_huricaines_shape_and_values():
    result = predict_huricaines(
        storm_id="AL09",
        vmax_kt=70,
        min_pressure_mb=980,
        lat=22.5,
        lon=-65.0,
        month=9,
        dvmax_6h=5.0,
        dpres_6h=-3.0,
    )
    assert result["storm_id"] == "AL09"
    assert result["model_version"] == HURICAINES_MODEL_VERSION
    assert 0.0 <= result["ri_probability_24h"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}


def test_demo_uses_hurricanes_user_facing_labels():
    components = demo.config.get("components", [])

    radio_component = next(
        component for component in components if component.get("type") == "radio"
    )
    choices = radio_component.get("props", {}).get("choices", [])
    choice_labels = [
        choice[0] if isinstance(choice, (list, tuple)) else str(choice)
        for choice in choices
    ]
    assert HURRICANES_LABEL in choice_labels
    assert "Huricaines" not in "".join(choice_labels)

    json_labels = [
        component.get("props", {}).get("label")
        for component in components
        if component.get("type") == "json"
    ]
    assert f"{HURRICANES_LABEL} Prediction" in json_labels

    button_values = [
        component.get("props", {}).get("value")
        for component in components
        if component.get("type") == "button"
    ]
    assert f"Predict {HURRICANES_LABEL}" in button_values

    html_values = [
        component.get("props", {}).get("value", "")
        for component in components
        if component.get("type") == "html"
    ]
    assert any(">Hurricanes</option>" in value for value in html_values)


def test_toggle_model_panel_accepts_hurricanes_label():
    hurricanes_panel, wildfires_panel = _toggle_model_panel(HURRICANES_LABEL)

    assert hurricanes_panel == gr.update(visible=True)
    assert wildfires_panel == gr.update(visible=False)
