from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

WEB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WEB_DIR.parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "deauth_model.joblib"

app = FastAPI(
    title="WIDS Console",
    description="Wireless intrusion detection system dashboard",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")
templates = Jinja2Templates(directory=WEB_DIR / "templates")


def load_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict):
        raise ValueError(f"Invalid model artifact at {MODEL_PATH}")
    return artifact


def artifact_summary(artifact: dict[str, Any] | None) -> dict[str, Any]:
    if artifact is None:
        return {
            "status": "not_loaded",
            "live_capture": False,
            "response_action": "disabled",
        }
    return {
        "status": "ready_for_offline_replay",
        "primary_model": "behavioural",
        "live_capture": False,
        "response_action": "disabled",
        "features": artifact.get("features", []),
        "metrics": artifact.get("metrics", {}),
        "dataset": artifact.get("dataset", {}),
    }


def render_page(
    request: Request,
    template: str,
    page_title: str,
    active_page: str,
    context: dict[str, Any] | None = None,
) -> HTMLResponse:
    values: dict[str, Any] = {
        "request": request,
        "page_title": page_title,
        "active_page": active_page,
    }
    if context:
        values.update(context)
    return templates.TemplateResponse(
        request=request,
        name=template,
        context=values,
    )


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    artifact = load_artifact()
    metrics = artifact.get("metrics", {}) if artifact else {}
    dataset = artifact.get("dataset", {}) if artifact else {}
    comparison = artifact.get("model_comparison", {}) if artifact else {}
    protocol_metrics = comparison.get("protocol_baseline", {})
    dataset["protocol_baseline_accuracy"] = protocol_metrics.get("accuracy", 0)
    dataset["protocol_baseline_recall"] = protocol_metrics.get(
        "classification_report", {}
    ).get("Deauth", {}).get("recall", 0)
    return render_page(
        request,
        "dashboard.html",
        "Overview",
        "overview",
        {
            "system_state": "Offline model only" if artifact else "Not configured",
            "model_state": "Loaded" if artifact else "Not trained",
            "dataset_state": (
                f"{dataset.get('rows', 0):,} rows validated"
                if dataset
                else "Pending validation"
            ),
            "recent_alerts": [],
            "accuracy": metrics.get("accuracy"),
            "live_capture": False,
        },
    )


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Expose an explicit readiness state for scripts and demonstrations."""
    return artifact_summary(load_artifact())


@app.get("/api/model")
async def model_api() -> dict[str, Any]:
    artifact = load_artifact()
    if artifact is None:
        return {"status": "not_loaded", "metrics": None, "dataset": None}
    return {
        "status": "ready_for_offline_replay",
        "primary_model": "behavioural",
        "features": artifact.get("features", []),
        "metrics": artifact.get("metrics", {}),
        "model_comparison": artifact.get("model_comparison", {}),
        "dataset": artifact.get("dataset", {}),
    }


@app.get("/detections", response_class=HTMLResponse)
async def detections(request: Request) -> HTMLResponse:
    artifact = load_artifact()
    detection_data = artifact.get("alerts", {}) if artifact else {}
    return render_page(
        request,
        "detections.html",
        "Detections",
        "detections",
        {
            "alerts": detection_data.get("items", []),
            "alert_total": detection_data.get("total", 0),
            "detection_mode": detection_data.get("mode", "No model replay available"),
        },
    )


@app.get("/api/detections")
async def detections_api() -> dict[str, Any]:
    artifact = load_artifact()
    detection_data = artifact.get("alerts", {}) if artifact else {}
    return {
        "mode": detection_data.get("mode", "No model replay available"),
        "total": detection_data.get("total", 0),
        "items": detection_data.get("items", []),
        "live_capture": False,
        "response_action": "disabled",
    }


@app.get("/model", response_class=HTMLResponse)
async def model(request: Request) -> HTMLResponse:
    artifact = load_artifact()
    metrics = artifact.get("metrics") if artifact else None
    dataset_info = dict(artifact.get("dataset", {})) if artifact else {}
    comparison = artifact.get("model_comparison", {}) if artifact else {}
    protocol_metrics = comparison.get("protocol_baseline", {})
    dataset_info["protocol_baseline_accuracy"] = protocol_metrics.get(
        "accuracy",
        0,
    )
    dataset_info["protocol_baseline_recall"] = protocol_metrics.get(
        "classification_report",
        {},
    ).get("Deauth", {}).get("recall", 0)
    return render_page(
        request,
        "model.html",
        "Model",
        "model",
        {
            "model_status": "Trained" if artifact else "Not trained",
            "metrics": metrics,
            "dataset": dataset_info,
        },
    )


@app.get("/dataset", response_class=HTMLResponse)
async def dataset(request: Request) -> HTMLResponse:
    artifact = load_artifact()
    dataset_info = artifact.get("dataset", {}) if artifact else {}
    return render_page(
        request,
        "dataset.html",
        "Dataset",
        "dataset",
        {
            "dataset_name": "AWID3",
            "dataset_status": "Validated" if dataset_info else "Awaiting validation",
            "row_count": dataset_info.get("rows"),
            "feature_count": dataset_info.get("features"),
            "class_count": dataset_info.get("classes"),
            "label_counts": dataset_info.get("label_counts", {}),
        },
    )
