# %% [markdown]
# # Main model: LightGBM
# Gradient-boosted trees, which can learn the amount bands and feature combinations
# that the logistic regression baseline couldn't.

# %%
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
                             recall_score, confusion_matrix, PrecisionRecallDisplay)

DATA = Path("data")
FIG = Path("figures")

train = pd.read_parquet(DATA / "train_features.parquet").sort_values("trans_date_trans_time")
test = pd.read_parquet(DATA / "test_features.parquet")

target = "is_fraud"
drop_cols = [target, "trans_date_trans_time"]

# Hold back the most recent 20% of the training period as a validation set
cutoff = train["trans_date_trans_time"].quantile(0.8)
fit = train[train["trans_date_trans_time"] < cutoff]
val = train[train["trans_date_trans_time"] >= cutoff]

def split_xy(df):
    return df.drop(columns=drop_cols), df[target]

X_fit, y_fit = split_xy(fit)
X_val, y_val = split_xy(val)
X_test, y_test = split_xy(test)

print("Validation starts:", cutoff)
for name, y in [("fit", y_fit), ("val", y_val), ("test", y_test)]:
    print(f"{name:5} {len(y):>9,} rows  {y.mean():.4%} fraud")

# %% [markdown]
# The most recent 20% of the training period (from 6 Mar 2020) is held back as a validation
# set for early stopping, so the test set stays unseen until the final evaluation.

# %% [markdown]
# ## Train with early stopping

# %%
params = dict(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    metric="average_precision",
    random_state=42,
    verbose=-1,
)

model = lgb.LGBMClassifier(**params)
model.fit(
    X_fit, y_fit,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
)
print("Best number of trees:", model.best_iteration_)

# %% [markdown]
# Validation PR-AUC improved from 0.843 at 100 trees to 0.907 at 529 trees, then stopped
# improving, so the model kept 529 trees.

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
# LightGBM has a PR-AUC of 0.862, over 3.5x the logistic regression baseline, and at the
# 0.5 threshold it caught more fraud (93.7%) with 18x fewer false alerts, around 2.4 for
# every real fraud compared to 46. The PR-AUC is lower than on the validation set (0.907),
# which reflects the lower fraud rate in the test period.

# %% [markdown]
# ## Compare with the baseline

# %%
logreg = pd.read_parquet(DATA / "preds_logreg.parquet")

fig, ax = plt.subplots(figsize=(6, 5))
PrecisionRecallDisplay.from_predictions(logreg["is_fraud"], logreg["proba"],
                                        name="Logistic regression", ax=ax)
PrecisionRecallDisplay.from_predictions(y_test, proba, name="LightGBM", ax=ax)
ax.axhline(y_test.mean(), linestyle="--", color="grey", label="Random guessing")
ax.legend()
ax.set_title("Precision-recall curves (test set)")
fig.savefig(FIG / "pr_curve_comparison.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Feature importance

# %%
importance = pd.Series(model.booster_.feature_importance("gain"), index=X_fit.columns)
importance = (importance / importance.sum()).sort_values(ascending=False)
print(importance.round(4))

# %% [markdown]
# Amount is by far the most important feature (67% of the gain), followed by the night-time
# flag (11%), showing the trees could learn the amount bands that the linear model couldn't.
# The amount compared to the card's average adds less here, as the trees can already split
# amount into bands directly.

# %% [markdown]
# ## Fairness check: with and without age

# %%
no_age_params = {**params, "n_estimators": model.best_iteration_}
model_no_age = lgb.LGBMClassifier(**no_age_params)
model_no_age.fit(X_fit.drop(columns="age"), y_fit)
proba_no_age = model_no_age.predict_proba(X_test.drop(columns="age"))[:, 1]

print(f"PR-AUC with age:    {average_precision_score(y_test, proba):.4f}")
print(f"PR-AUC without age: {average_precision_score(y_test, proba_no_age):.4f}")

age_band = pd.cut(X_test["age"], bins=[0, 25, 35, 45, 55, 65, 100])
fairness = pd.DataFrame({
    "fraud_rate": y_test.groupby(age_band, observed=True).mean(),
    "flagged_with_age": pd.Series(proba >= 0.5, index=X_test.index)
                          .groupby(age_band, observed=True).mean(),
    "flagged_without_age": pd.Series(proba_no_age >= 0.5, index=X_test.index)
                             .groupby(age_band, observed=True).mean(),
})
print(fairness.round(4))

# %% [markdown]
# Removing age dropped the PR-AUC from 0.862 to 0.803. The model with age actually flags
# older customers the least, so age was kept in, with the trade-off that younger customers
# get more false alerts relative to their fraud rate, which a bank would need to monitor.

# %% [markdown]
# ## Save predictions and model

# %%
pd.DataFrame({"is_fraud": y_test.to_numpy(), "proba": proba}).to_parquet(
    DATA / "preds_lgbm.parquet")
model.booster_.save_model(str(DATA / "lgbm_model.txt"))

# %% [markdown]
# ## Takeaway
# LightGBM is a big improvement on the baseline, with a PR-AUC of 0.862 compared to 0.239,
# because the trees can learn the amount bands and combinations of features that a
# straight-line model can't. At the default threshold it would flag around 2.4 genuine
# transactions for every real fraud, which is far closer to something a bank could use.
# The fairness check showed age improves the model without penalising older customers, so
# it was kept in.
