# %% [markdown]
# # Baseline model: logistic regression
# A simple model to set the benchmark. The main model in Step 4 has to clearly beat this.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, confusion_matrix, PrecisionRecallDisplay)

DATA = Path("data")
FIG = Path("figures")

train = pd.read_parquet(DATA / "train_features.parquet")
test = pd.read_parquet(DATA / "test_features.parquet")

target = "is_fraud"
drop_cols = [target, "trans_date_trans_time"]

X_train = train.drop(columns=drop_cols)
y_train = train[target]
X_test = test.drop(columns=drop_cols)
y_test = test[target]

print(X_train.shape, X_test.shape)

# %% [markdown]
# 21 features in both train (1,296,675 rows) and test (555,719 rows).

# %% [markdown]
# ## Log-transform skewed features

# %%
for X in (X_train, X_test):
    X["amt"] = np.log1p(X["amt"])
    X["amt_vs_card_avg"] = np.log1p(X["amt_vs_card_avg"])

print(X_train[["amt", "amt_vs_card_avg"]].describe())

# %% [markdown]
# Amounts are heavily skewed, so the log squashes the largest values to stop them
# dominating a linear model. The max amount went from around 600x the median to under 3x.

# %% [markdown]
# ## Train the model

# %%
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000),
)
model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluate on the test set

# %%
proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"PR-AUC:    {average_precision_score(y_test, proba):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, proba):.4f}")
print(f"Precision: {precision_score(y_test, pred):.4f}  (at 0.5 threshold)")
print(f"Recall:    {recall_score(y_test, pred):.4f}  (at 0.5 threshold)")
print(confusion_matrix(y_test, pred))

# %% [markdown]
# The logistic regression model has a PR-AUC of 0.239, which is around 60x better than
# random guessing (0.0039), showing it has learned something, but it is still weak overall.
# The ROC-AUC of 0.931 looks a lot better, which shows how misleading ROC-AUC can be when
# fraud is this rare. At the 0.5 threshold, the model caught 87.9% of fraud, but only 2.1%
# of the transactions it flagged were actually fraud, which works out at around 46 false
# alerts for every real fraud and 16% of genuine transactions being blocked. This would not
# be usable for a bank, and is likely because a linear model cannot pick up the amount bands
# or the combinations of features found in the EDA.

# %% [markdown]
# ## Precision-recall curve

# %%
fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(y_test, proba, name="Logistic regression", ax=ax)
ax.axhline(y_test.mean(), linestyle="--", color="grey", label="Random guessing")
ax.legend()
ax.set_title("Precision-recall curve (test set)")
fig.savefig(FIG / "pr_curve_logreg.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# The precision-recall curve shows the trade-off between catching fraud and false alerts.
# At around 20% recall, precision peaks at about 57%, but to catch half the fraud precision
# falls to around 22%, and beyond 70% recall it drops to 2-3%. The curve is also unusual on
# the far left, as the transactions the model is most confident about are mostly legitimate.
# This is likely because logistic regression can only learn that a higher amount means more
# risk, so it scores the largest transactions highest, even though fraud never goes above
# $1,376 while legitimate transactions go up to $28,949. This is more evidence that a
# non-linear model is needed.

# %% [markdown]
# ## What the model learned

# %%
coefs = pd.Series(model[-1].coef_[0], index=X_train.columns).sort_values()
print(coefs)

# %% [markdown]
# The strongest feature in the logistic regression is the night-time flag (+1.22),
# followed by the transaction amount compared to the card's average (+0.74), which shows
# the features engineered from the EDA are the most useful to the model. Amount on its own
# has a smaller weight (+0.21), but as both amount features are positive, the model still
# scores the biggest transactions highest. The category weights don't always match the fraud
# rates from the EDA, for example shopping_net has the highest fraud rate but a weight of
# almost zero. This is because the coefficients show each feature's effect after taking the
# others into account, so the amount features already explain most of the online shopping
# fraud.

# %% [markdown]
# ## Save predictions for comparison in Step 5

# %%
pd.DataFrame({"is_fraud": y_test.to_numpy(), "proba": proba}).to_parquet(
    DATA / "preds_logreg.parquet")

# %% [markdown]
# ## Takeaway
# The logistic regression baseline has a PR-AUC of 0.239, around 60x better than random
# guessing, which sets the benchmark for the LightGBM model in Step 4 to beat. The
# night-time flag and comparing amounts to the card's average were the strongest features,
# showing the feature engineering worked. However, at the default threshold the model would
# flag around 46 genuine transactions for every real fraud, and it scores the largest
# transactions highest even though they are legitimate. Both of these come from the model
# only being able to learn straight-line relationships, so a non-linear model should do a
# lot better.
