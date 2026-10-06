# %% [markdown]
# # Feature engineering
# Turns the raw columns into features the models can use, based on the EDA findings.
# Train and test are combined first so each card's history carries on from the train
# period into the test period, then they are split back apart at the end.

# %%
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data")
dates = ["trans_date_trans_time", "dob"]

train = pd.read_csv(DATA / "fraudTrain.csv", index_col=0, parse_dates=dates)
test = pd.read_csv(DATA / "fraudTest.csv", index_col=0, parse_dates=dates)

train["split"] = "train"
test["split"] = "test"

df = pd.concat([train, test], ignore_index=True)
df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

print(df.shape)
print(df.groupby("split")["trans_date_trans_time"].agg(["min", "max"]))

# %% [markdown]
# 1,852,394 transactions in total. Train runs from 1 Jan 2019 to 21 Jun 2020 and test
# runs from 21 Jun 2020 to 31 Dec 2020, so there is no overlap between them.

# %% [markdown]
# ## Time features

# %%
ts = df["trans_date_trans_time"]
df["hour"] = ts.dt.hour
df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 3)).astype(int)
df["day_of_week"] = ts.dt.dayofweek

print(df.groupby("is_night")["is_fraud"].mean())

# %% [markdown]
# The fraud rate is 0.10% in the day compared to 1.88% at night, so the night-time flag
# picks up the pattern from the EDA, with fraud around 18x higher at night.

# %% [markdown]
# ## Age

# %%
df["age"] = (ts - df["dob"]).dt.days // 365

# %% [markdown]
# Age is kept in for now, but the model will be tested with and without it, as it is a
# protected characteristic.

# %% [markdown]
# ## Distance between cardholder and merchant

# %%
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))

df["distance_km"] = haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"])

print(df.groupby("is_fraud")["distance_km"].describe())

# %% [markdown]
# Distance between cardholder and merchant shows no meaningful difference between fraud
# and legitimate transactions, with a median of 78km for both and near-identical spread,
# so it was dropped as a feature.

# %% [markdown]
# ## Per-card behaviour

# %%
card = df.groupby("cc_num")

# Average of this card's previous amounts (shift() excludes the current transaction)
card_avg = card["amt"].transform(lambda s: s.shift().expanding().mean())
df["amt_vs_card_avg"] = (df["amt"] / card_avg).fillna(1.0)

# Number of transactions this card made in the previous 24 hours
counts = (df.set_index("trans_date_trans_time")
            .groupby("cc_num")["amt"]
            .rolling("24h")
            .count())
assert (counts.index.get_level_values("cc_num") == df["cc_num"]).all()
df["txn_count_24h"] = counts.to_numpy() - 1

print(df.groupby("is_fraud")[["amt_vs_card_avg", "txn_count_24h"]].median())

# %% [markdown]
# The fraudulent transactions are typically over 5x the card's usual average amount,
# compared to 0.67x for legitimate transactions, indicating that comparing a transaction
# to the card's own spending history is a strong characteristic to include in the model.
# This could help where an amount looks normal across the whole dataset but is unusual for
# that specific card. The number of transactions in the previous 24 hours was only slightly
# higher for fraud, with a median of 4 compared to 3, so it looks like a weaker
# characteristic, but it will be kept in for now to see if the model uses it.

# %% [markdown]
# ## Merchant category to numbers

# %%
df = pd.get_dummies(df, columns=["category"], prefix="cat", dtype=int)
cat_cols = [c for c in df.columns if c.startswith("cat_")]
print(cat_cols)

# %% [markdown]
# Category is one-hot encoded into 14 columns of 1s and 0s, so the model doesn't read
# an order into the categories that doesn't exist.

# %% [markdown]
# ## Save

# %%
# Gender is left out as it is a protected characteristic, and distance had no signal
features = (["amt", "hour", "is_night", "day_of_week", "age",
             "amt_vs_card_avg", "txn_count_24h"] + cat_cols)

out = df[features + ["is_fraud", "split", "trans_date_trans_time"]]
train_out = out[out["split"] == "train"].drop(columns="split")
test_out = out[out["split"] == "test"].drop(columns="split")

train_out.to_parquet(DATA / "train_features.parquet")
test_out.to_parquet(DATA / "test_features.parquet")

print(train_out.shape, f"{train_out['is_fraud'].mean():.4%}")
print(test_out.shape, f"{test_out['is_fraud'].mean():.4%}")

# %% [markdown]
# ## Takeaway
# The feature engineering created 21 features for the model, built from the findings in
# the EDA. The strongest new characteristics look to be the night-time flag, where fraud
# is around 18x higher at night, and comparing each transaction to the card's own average
# spend, where fraud is typically over 5x the usual amount. Distance between the
# cardholder and merchant was dropped as it showed no difference between fraud and
# legitimate transactions, and gender was left out as it is a protected characteristic.
# Age has been kept in for now, but the model will be tested with and without it to check
# fairness. The test set also has a lower fraud rate than the training set, 0.39%
# compared to 0.58%, which shows fraud levels can shift over time.
