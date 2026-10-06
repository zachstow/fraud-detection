# %% [markdown]
# # Exploratory data analysis
# A first look at the training data before building any model: size, missing values,
# how rare fraud is, and which variables differ between fraud and legitimate transactions.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA = Path("data")
FIG = Path("figures")
FIG.mkdir(exist_ok=True)

train = pd.read_csv(
    DATA / "fraudTrain.csv",
    index_col=0,
    parse_dates=["trans_date_trans_time", "dob"],
)
print(train.shape)
print(train.dtypes)
print(train.isna().sum().sum(), "missing values")

# %% [markdown]
# 1,296,675 transactions and 22 columns, with no missing values.

# %% [markdown]
# ## Class imbalance
# Fraud is very rare compared to legitimate transactions. A model that just predicts
# "not fraud" every time would still score over 99% accuracy while catching no fraud,
# so accuracy can't be used to judge the model. Precision and recall will be used instead.

# %%
fraud_rate = train["is_fraud"].mean()
print(f"Fraud rate: {fraud_rate:.4%}  ({train['is_fraud'].sum():,} of {len(train):,})")

# %% [markdown]
# Fraud rate is 0.58% (7,506 of 1,296,675), about 1 in every 173 transactions.

# %% [markdown]
# ## Transaction amount

# %%
print(train.groupby("is_fraud")["amt"].describe())

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data=train, x="amt", hue="is_fraud", log_scale=True,
             stat="density", common_norm=False, bins=60, ax=ax)
ax.set_title("Transaction amount by class (log scale)")
fig.savefig(FIG / "amount_by_class.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# Fraud transactions cluster around $10-25, which is consistent with fraudsters testing
# whether the card works. The next clusters are a lot higher, between $250 and $1,100
# (median $397 vs $47 for legitimate). There is little overlap here between legitimate and
# fraudulent transactions, so amount alone separates most of the larger fraud. However, the
# fraud around $10-25 overlaps with everyday card transactions, so amount on its own won't
# be enough to separate legitimate from fraudulent transactions.

# %% [markdown]
# ## Merchant category

# %%
cat_rate = (train.groupby("category")["is_fraud"]
            .agg(rate="mean", n="size")
            .sort_values("rate", ascending=False))
print(cat_rate)

fig, ax = plt.subplots(figsize=(8, 5))
cat_rate["rate"].plot.barh(ax=ax)
ax.invert_yaxis()
ax.set_title("Fraud rate by merchant category")
fig.savefig(FIG / "fraud_rate_by_category.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# The riskiest category (shopping_net, 1.76%) has a fraud rate nearly 11x the lowest
# (health_fitness, 0.15%), so category should be a solid feature to include in the model.
# Out of the 3 riskiest categories, 2 are online, known as "card-not-present" transactions,
# which suggests online purchases carry more risk. In-person grocery is the exception at 1.41%.

# %% [markdown]
# ## Hour of day

# %%
train["hour"] = train["trans_date_trans_time"].dt.hour
hour_rate = train.groupby("hour")["is_fraud"].mean()

fig, ax = plt.subplots(figsize=(8, 4))
hour_rate.plot(marker="o", ax=ax)
ax.set_title("Fraud rate by hour of day")
ax.set_ylabel("Fraud rate")
fig.savefig(FIG / "fraud_rate_by_hour.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# Fraud is particularly concentrated during the night: about 2.9% between 22:00-23:59 and
# 1.5% between 00:00-03:59, against about 0.1% during the day. This could be because people
# typically aren't awake to react to fraud alerts, and there are also far fewer genuine
# transactions at night, which pushes the rate up. Fraud is high at both ends of the clock,
# so hour isn't a straight-line relationship.

# %% [markdown]
# ## Cardholder age

# %%
train["age"] = (train["trans_date_trans_time"] - train["dob"]).dt.days // 365
train["age_band"] = pd.cut(train["age"], bins=[0, 25, 35, 45, 55, 65, 100])
print(train.groupby("age_band", observed=True)["is_fraud"].mean())

# %% [markdown]
# Older cardholders have the highest fraud rates (about 0.77% for over-55s), but under-25s
# are also above average (0.62%) and 35-45 is the lowest (0.43%), so it's U-shaped. The
# differences are small compared to category and hour, and age is a protected
# characteristic, so it needs a fairness check if used in the model.

# %% [markdown]
# ## Takeaway
# Fraud is only 0.58% of transactions, so accuracy can't be used to judge the model.
# Fraud clusters by amount, where the higher cluster provides a stronger indication of
# fraud, while the lower cluster is weaker due to the overlap with legitimate transactions.
# Fraud varying by merchant category and hour of the day are very strong characteristics
# due to their ranges in fraud rate. These patterns aren't straight-line relationships,
# which suggests using engineered features and a tree-based model.
