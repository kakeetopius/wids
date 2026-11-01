# Wireless Intrusion Detection System

This project uses two scikit-learn Random Forest models to detect
deauthentication traffic in the AWID3 wireless dataset. The behavioural model
is the primary detector; a protocol-feature model is retained as a comparison
baseline.

## Quick setup

These steps run the included trained model and dashboard without the raw dataset:

```bash
git clone https://github.com/kakeetopius/wids.git
cd wids
uv sync
uv run uvicorn wids.web.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. Stop the server with `Ctrl+C`.

## Run the existing trained model

Requirements:

- Linux, macOS, or Windows
- Python 3.14
- [uv](https://docs.astral.sh/uv/)

Install the exact locked dependencies:

```bash
git clone https://github.com/kakeetopius/wids.git
cd wids
uv sync
```

Start the dashboard:

```bash
uv run uvicorn wids.web.app:app --reload
```

Open <http://127.0.0.1:8000>.

- `/` — project overview
- `/model` — accuracy, precision, recall, F1-score, and confusion matrix
- `/dataset` — dataset summary and evaluation split
- `/detections` — top alerts from the offline test replay

```text
models/deauth_model.joblib
```

## Reported test result

The included model was evaluated on capture files that were not used for
training or threshold selection:

| Metric                     | Result |
| -------------------------- | -----: |
| Model                 | Accuracy | Deauth precision | Deauth recall | Deauth F1 |
| --------------------- | -------: | ---------------: | ------------: | --------: |
| Behavioural detector  | 98.14%   | 99.99%           | 96.51%        | 98.22%    |
| Protocol baseline     | 100.00%  | 100.00%          | 100.00%       | 100.00%   |

Confusion matrix:

| Actual / predicted | Normal | Deauth |
| ------------------ | -----: | -----: |
| Normal             | 15,843 |      1 |
| Deauth             |    629 | 17,396 |

The Detections page displays predictions from this offline test replay. It does not capture live wireless traffic or automatically block devices.

## Optional: retrain the model

Retraining is not required to run the dashboard. To reproduce or change the
model, obtain the same AWID3 CSV release and place the deauthentication files
in:

```text
CSV/1.Deauth/
```

The trainer expects files named like `Deauth_0.csv` through `Deauth_32.csv`,
with a `Label` column containing `Normal` and `Deauth`.

Run:

```bash
uv run python -m wids.train
```

The model uses capture-file groups:

- Training: files outside the validation and test groups
- Validation: `Deauth_25.csv` through `Deauth_28.csv`
- Final test: `Deauth_29.csv` through `Deauth_32.csv`

The primary behavioural model excludes `wlan.fc.type` and `wlan.fc.subtype`.
The comparison baseline includes them and reaches 100% on this holdout because
they are highly informative for this AWID3 task. The baseline is not used for
the dashboard replay alerts. Both models derive one-second overall, source,
and destination traffic-rate features.

## A1 poster

![A1 poster preview](poster/A1_poster.jpg)
