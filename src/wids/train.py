from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "CSV" / "1.Deauth"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "deauth_model.joblib"

CHUNK_SIZE = 100_000
NORMAL_SAMPLE_FRAC = 0.10
RANDOM_STATE = 42

FEATURES = [
    "frame.len",
    "frame.time_delta",
    "radiotap.channel.freq",
    "radiotap.datarate",
    "wlan.duration",
    "wlan.fc.frag",
    "wlan.fc.moredata",
    "wlan.fc.protected",
    "wlan.fc.pwrmgt",
    "wlan.fc.retry",
    "wlan.fc.type",
    "wlan.fc.subtype",
    "wlan.fixed.reason_code",
    "wlan_radio.signal_dbm",
    "frames_per_second",
    "source_frames_per_second",
    "destination_frames_per_second",
]

LABEL = "Label"
METADATA_COLUMNS = [
    "frame.time_relative",
    "wlan.sa",
    "wlan.da",
]
RAW_FEATURES = FEATURES[:14]
DECISION_THRESHOLDS = (0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20)

# These complete captures are reserved for final testing. Files 25-28 are
# used only for choosing the alert threshold.
VALIDATION_FILES = {
    "Deauth_25.csv",
    "Deauth_26.csv",
    "Deauth_27.csv",
    "Deauth_28.csv",
}
TEST_FILES = {
    "Deauth_29.csv",
    "Deauth_30.csv",
    "Deauth_31.csv",
    "Deauth_32.csv",
}


def load_dataset() -> pd.DataFrame:
    frames = []
    files = sorted(
        DATA_DIR.glob("Deauth_*.csv"),
        key=lambda path: int(path.stem.split("_")[1]),
    )

    if not files:
        raise FileNotFoundError(f"No Deauth CSV files found in {DATA_DIR}")

    print(f"Found {len(files)} CSV files")

    for file in files:
        print(f"Reading {file.name}")

        header = pd.read_csv(
            file,
            nrows=0,
            encoding="ISO-8859-1",
        )

        available = [feature for feature in RAW_FEATURES if feature in header.columns]
        metadata = [column for column in METADATA_COLUMNS if column in header.columns]

        if LABEL not in header.columns:
            print(f"  Skipping {file.name}: missing {LABEL}")
            continue

        if not available:
            print(f"  Skipping {file.name}: no selected features found")
            continue

        for chunk in pd.read_csv(
            file,
            usecols=available + metadata + [LABEL],
            chunksize=CHUNK_SIZE,
            encoding="ISO-8859-1",
            low_memory=False,
            on_bad_lines="skip",
        ):
            chunk[LABEL] = chunk[LABEL].astype(str).str.strip()
            chunk = add_traffic_features(chunk, metadata)

            attacks = chunk[chunk[LABEL] == "Deauth"]

            normal = chunk[chunk[LABEL] == "Normal"].sample(
                frac=NORMAL_SAMPLE_FRAC,
                random_state=RANDOM_STATE,
            )

            selected = pd.concat(
                [normal, attacks],
                ignore_index=True,
            )

            if not selected.empty:
                selected["_source_file"] = file.name
                frames.append(selected)

    if not frames:
        raise ValueError("No usable rows were found.")

    data = pd.concat(frames, ignore_index=True)

    for feature in FEATURES:
        if feature in data.columns:
            data[feature] = pd.to_numeric(
                data[feature],
                errors="coerce",
            )

    usable_features = [
        feature
        for feature in FEATURES
        if feature in data.columns and data[feature].notna().any()
    ]

    if not usable_features:
        raise ValueError("No selected features contain usable numeric values.")

    metadata = [column for column in METADATA_COLUMNS if column in data.columns]
    data = data[usable_features + [LABEL, "_source_file"] + metadata].dropna(
        subset=[LABEL]
    )

    print("\nDataset label counts:")
    print(data[LABEL].value_counts())

    print("\nUsing features:")
    for feature in usable_features:
        print(f"  {feature}")

    return data


def _window_counts(times: pd.Series, keys: pd.Series, window: float = 1.0) -> pd.Series:
    """Count frames sharing a key during the trailing time window."""
    result = pd.Series(0, index=times.index, dtype="int64")
    ordered = times.sort_values().index
    queues: dict[str, list[float]] = {}
    for index in ordered:
        key = str(keys.loc[index])
        if key in {"nan", "None", ""}:
            continue
        timestamp = float(times.loc[index])
        queue = queues.setdefault(key, [])
        queue.append(timestamp)
        while queue and queue[0] < timestamp - window:
            queue.pop(0)
        result.loc[index] = len(queue)
    return result


def add_traffic_features(
    frame: pd.DataFrame,
    metadata: list[str],
) -> pd.DataFrame:
    """Add inference-safe traffic-rate features without using the label."""
    if "frame.time_relative" not in frame.columns:
        return frame

    times = pd.to_numeric(frame["frame.time_relative"], errors="coerce")
    valid_times = times.ffill().fillna(0.0)
    frame["frames_per_second"] = _window_counts(
        valid_times,
        pd.Series("_all", index=frame.index),
    )
    if "wlan.sa" in metadata:
        frame["source_frames_per_second"] = _window_counts(
            valid_times,
            frame["wlan.sa"],
        )
    else:
        frame["source_frames_per_second"] = frame["frames_per_second"]
    if "wlan.da" in metadata:
        frame["destination_frames_per_second"] = _window_counts(
            valid_times,
            frame["wlan.da"],
        )
    else:
        frame["destination_frames_per_second"] = frame["frames_per_second"]
    return frame


def _test_files(files: list[Path]) -> set[str]:
    """Use the documented AWID3 holdout when it exists, otherwise hold out 20%."""
    available = {file.name for file in files}
    selected = TEST_FILES & available
    if selected:
        return selected
    holdout_count = max(1, round(len(files) * 0.2))
    return {file.name for file in files[-holdout_count:]}


def _validation_files(files: list[Path], test_files: set[str]) -> set[str]:
    available = {file.name for file in files}
    selected = VALIDATION_FILES & available - test_files
    if selected:
        return selected
    remaining = sorted(available - test_files)
    return set(remaining[-max(1, round(len(remaining) * 0.2)):])


def _report(
    y_true: pd.Series,
    probabilities: Any,
    threshold: float,
) -> tuple[dict[str, Any], list[list[int]]]:
    predictions = ["Deauth" if value >= threshold else "Normal" for value in probabilities]
    report = classification_report(
        y_true,
        predictions,
        labels=["Normal", "Deauth"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=["Normal", "Deauth"])
    return report, matrix.tolist()


def _model_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[LABEL, "_source_file", *METADATA_COLUMNS], errors="ignore")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data = load_dataset()

    # Split by complete CSV chunks, not individual rows.
    files = sorted(
        DATA_DIR.glob("Deauth_*.csv"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    test_files = _test_files(files)
    validation_files = _validation_files(files, test_files)
    split = data["_source_file"]
    is_test = split.isin(test_files)
    is_validation = split.isin(validation_files)

    train_data = data[~is_test & ~is_validation]
    validation_data = data[is_validation]
    test_data = data[is_test]

    if train_data.empty or test_data.empty:
        raise ValueError("Training or testing dataset is empty.")

    X_train = _model_features(train_data)
    y_train = train_data[LABEL]

    X_validation = _model_features(validation_data)
    X_test = _model_features(test_data)
    y_validation = validation_data[LABEL]
    y_test = test_data[LABEL]

    print("\nTraining label counts:")
    print(y_train.value_counts())

    print("\nTesting label counts:")
    print(y_test.value_counts())

    if y_train.nunique() != 2:
        raise ValueError("Training data must contain Normal and Deauth.")

    if y_test.nunique() != 2:
        raise ValueError(
            "Testing data must contain Normal and Deauth. "
            "Check the label distribution in the held-out files."
        )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    deauth_index = list(model.classes_).index("Deauth")
    validation_probabilities = model.predict_proba(X_validation)[:, deauth_index]

    # Select the lowest threshold with the best validation F1. This favors
    # finding attacks when several thresholds perform equally well.
    candidates = []
    for candidate in DECISION_THRESHOLDS:
        candidate_report, _ = _report(
            y_validation,
            validation_probabilities,
            candidate,
        )
        candidates.append((candidate, candidate_report))
    threshold = max(
        candidates,
        key=lambda item: (
            item[1]["Deauth"]["f1-score"],
            item[1]["Deauth"]["recall"],
            -item[0],
        ),
    )[0]

    print(f"\nSelected alert threshold from validation data: {threshold:.2f}")
    test_probabilities = model.predict_proba(X_test)[:, deauth_index]
    report, matrix_list = _report(y_test, test_probabilities, threshold)
    test_predictions = test_probabilities >= threshold
    alert_rows = test_data.loc[test_predictions].copy()
    alert_rows["_confidence"] = test_probabilities[test_predictions]
    alert_rows = alert_rows.sort_values("_confidence", ascending=False).head(20)
    alerts = [
        {
            "timestamp": float(row["frame.time_relative"])
            if pd.notna(row.get("frame.time_relative"))
            else None,
            "capture": str(row["_source_file"]),
            "source": str(row.get("wlan.sa", "Unknown")),
            "destination": str(row.get("wlan.da", "Unknown")),
            "confidence": float(row["_confidence"]),
            "severity": "High" if row["_confidence"] >= 0.8 else "Medium",
        }
        for _, row in alert_rows.iterrows()
    ]

    print("\nClassification report:")
    print(pd.DataFrame(report).transpose().round(4).to_string())
    print("Confusion matrix (Normal, Deauth):")
    print(matrix_list)
    validation_report, validation_matrix = _report(
        y_validation,
        validation_probabilities,
        threshold,
    )

    artifact = {
        "model": model,
        "features": list(X_train.columns),
        "classes": list(model.classes_),
        "metrics": {
            "accuracy": float(report["accuracy"]),
            "classification_report": report,
            "confusion_matrix": matrix_list,
            "confusion_matrix_labels": ["Normal", "Deauth"],
            "decision_threshold": threshold,
            "validation": {
                "accuracy": float(validation_report["accuracy"]),
                "classification_report": validation_report,
                "confusion_matrix": validation_matrix,
            },
        },
        "dataset": {
            "name": "AWID3",
            "rows": int(len(data)),
            "features": int(len(X_train.columns)),
            "classes": int(y_train.nunique()),
            "label_counts": {
                str(label): int(count)
                for label, count in data[LABEL].value_counts().items()
            },
            "validation_files": sorted(validation_files),
            "test_files": sorted(test_files),
            "split_strategy": "train/validation/test capture-file holdout",
            "included_protocol_features": [
                "wlan.fc.type",
                "wlan.fc.subtype",
            ],
        },
        "alerts": {
            "mode": "offline test replay",
            "total": int(test_predictions.sum()),
            "items": alerts,
        },
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
