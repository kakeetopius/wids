from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/deauth_model.joblib")

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

classifier = model.named_steps["classifier"]

importance = pd.DataFrame(
    {
        "feature": features,
        "importance": classifier.feature_importances_,
    }
).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance.to_string(index=False))
