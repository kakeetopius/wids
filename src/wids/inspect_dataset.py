import pandas as pd

df = pd.read_csv("AWID3_DualPath_IDS_Release/data/processed__awid3.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nClass distribution:")
print(df["Durum"].value_counts(dropna=False))

print("\nClass proportions:")
print(df["Durum"].value_counts(normalize=True, dropna=False))

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

print("\nSample rows:")
print(df.head())

# Frame type and subtype values associated with each label.
print("\nFrame type counts by label:")
print(pd.crosstab(df["Durum"], df["wlan.fc.type"]))

print("\nFrame subtype counts by label:")
print(pd.crosstab(df["Durum"], df["wlan.fc.subtype"]))

# Focus on normal and deauthentication-labelled frames.
focused = df[df["Durum"].isin(["Normal", "Saldiri (Deauth)"])]

print("\nFocused dataset class counts:")
print(focused["Durum"].value_counts())

print("\nFocused dataset frame type/subtype counts:")
print(
    focused.groupby(
        ["Durum", "wlan.fc.type", "wlan.fc.subtype"],
        observed=True,
    )
    .size()
    .sort_values(ascending=False)
    .head(25)
)

# Check whether frame numbers are ordered in the CSV.
print(
    "\nFrame numbers monotonically increasing:",
    df["frame.number"].is_monotonic_increasing,
)

# Inspect deauthentication-labelled records that do not have
# the expected management-frame type/subtype combination.
deauth = df[df["Durum"] == "Saldiri (Deauth)"]

expected = deauth[(deauth["wlan.fc.type"] == 0) & (deauth["wlan.fc.subtype"] == 12)]

unexpected = deauth.drop(expected.index)

print("\nDeauth-labelled rows:", len(deauth))
print("Management deauthentication combination:", len(expected))
print("Other combinations:", len(unexpected))

print("\nUnexpected combinations:")
print(
    unexpected.groupby(["wlan.fc.type", "wlan.fc.subtype"])
    .size()
    .sort_values(ascending=False)
)

print("\nExample unexpected deauth-labelled rows:")
print(unexpected.head(10).to_string(index=False))

print("\nExpected combination by label:")
print(
    df.groupby("Durum").apply(
        lambda group: (
            (group["wlan.fc.type"] == 0) & (group["wlan.fc.subtype"] == 12)
        ).sum(),
        include_groups=False,
    )
)
