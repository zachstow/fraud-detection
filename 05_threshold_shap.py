# %% [markdown]
# # Threshold tuning and SHAP explainability
# Picks the threshold that minimises the cost of fraud plus false alerts, then explains
# what drives the LightGBM model overall and for a single flagged transaction.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, confusion_matrix

DATA = Path("data")
FIG = Path("figures")

test = pd.read_parquet(DATA / "test_features.parquet")
preds = pd.read_parquet(DATA / "preds_lgbm.parquet")

y = preds["is_fraud"].to_numpy()
proba = preds["proba"].to_numpy()
amt = test["amt"].to_numpy()

# %% [markdown]
# ## Cost-based threshold
# Assumption: a missed fraud costs the transaction amount, and each false alert costs
# a fixed $10 (review time and an annoyed customer).

# %%
FALSE_ALERT_COST = 10
thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

def total_cost(t, fa_cost=FALSE_ALERT_COST):
    flagged = proba >= t
    missed_fraud = amt[(y == 1) & ~flagged].sum()
    false_alerts = ((y == 0) & flagged).sum() * fa_cost
    return missed_fraud + false_alerts

costs = pd.Series([total_cost(t) for t in thresholds], index=thresholds)
best_t = costs.idxmin()

print(f"Cost with no model (all fraud missed): ${amt[y == 1].sum():,.0f}")
print(f"Cost at default 0.50 threshold:        ${costs[0.5]:,.0f}")
print(f"Cost at best threshold {best_t:.2f}:          ${costs[best_t]:,.0f}")

fig, ax = plt.subplots(figsize=(7, 4))
costs.plot(ax=ax)
ax.axvline(best_t, linestyle="--", color="grey", label=f"Best threshold ({best_t:.2f})")
ax.set_xlabel("Threshold")
ax.set_ylabel("Total cost ($)")
ax.set_title("Cost of missed fraud plus false alerts by threshold")
ax.legend()
fig.savefig(FIG / "threshold_cost.png", dpi=150, bbox_inches="tight")

# %% [markdown]
# ## Results at the chosen threshold

# %%
pred = (proba >= best_t).astype(int)
cm = confusion_matrix(y, pred)
print(f"Precision: {precision_score(y, pred):.4f}")
print(f"Recall:    {recall_score(y, pred):.4f}")
print(cm)
print(f"False alerts per real fraud caught: {cm[0, 1] / cm[1, 1]:.2f}")

# %% [markdown]
# Using a cost-based threshold, assuming each false alert costs $10 and missed fraud costs
# the transaction amount, the best threshold is 0.65. At this threshold the model catches
# 92.0% of fraud with a precision of 36.4%, which works out at 1.74 false alerts for every
# real fraud caught, and the total cost falls from $1.13m with no model to $62,312, a
# reduction of around 95%.

# %% [markdown]
# ## How sensitive is the threshold to the cost assumption?

# %%
for fa in [5, 10, 25, 50]:
    c = pd.Series([total_cost(t, fa) for t in thresholds], index=thresholds)
    t = c.idxmin()
    p = (proba >= t).astype(int)
    print(f"False alert cost ${fa:>2}: best threshold {t:.2f}, "
          f"precision {precision_score(y, p):.3f}, recall {recall_score(y, p):.3f}")

# %% [markdown]
# The best threshold changes a lot with the false alert cost, from 0.49 at $5 to 0.94 at
# $50, so the threshold is as much a business decision as a technical one.

# %% [markdown]
# ## SHAP: what drives the model overall

# %%
booster = lgb.Booster(model_file=str(DATA / "lgbm_model.txt"))
X_test = test.drop(columns=["is_fraud", "trans_date_trans_time"])

# Explain a sample: all frauds plus 10,000 random legitimate transactions
sample = pd.concat([
    X_test[y == 1],
    X_test[y == 0].sample(min(10_000, (y == 0).sum()), random_state=42),
])

explainer = shap.TreeExplainer(booster)
sv = explainer(sample)

plt.figure()
shap.plots.beeswarm(sv, max_display=12, show=False)
plt.title("SHAP: impact of each feature on the fraud score")
plt.savefig(FIG / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.show()

mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=sample.columns)
print(mean_abs.sort_values(ascending=False).round(3).head(10))

# %% [markdown]
# SHAP shows amount is by far the biggest driver of the model, with high amounts pushing the
# fraud score up the most, followed by the night-time flag. Some of the results go against
# what I expected from the EDA, as being in shopping_net lowers the fraud score even though
# it had the highest fraud rate, and a high number of transactions in the previous 24 hours
# also lowers the score. This is because SHAP shows each feature's effect after the others
# are taken into account, so once the amount is known, a large online shopping purchase is
# less suspicious than a large grocery purchase.

# %% [markdown]
# ## SHAP: why one transaction was flagged

# %%
flagged_frauds = np.where((y == 1) & (proba >= best_t))[0]
i = flagged_frauds[0]
row = X_test.iloc[[i]]
print(row.T.squeeze()[lambda s: s != 0])
print(f"Fraud score: {proba[i]:.3f}")

plt.figure()
shap.plots.waterfall(explainer(row)[0], max_display=10, show=False)
plt.savefig(FIG / "shap_waterfall_example.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# For the example fraud, a $337 grocery transaction at 1am, the amount (+9.1), the grocery
# category (+4.23) and the night-time flag (+1.03) pushed the score to almost 1.

# %% [markdown]
# ## Takeaway
# The default 0.5 threshold isn't the best choice. Costing missed fraud and false alerts
# gives a threshold of 0.65 that cuts fraud losses by around 95%, but the right threshold
# depends on how a bank values false alerts. SHAP makes the model explainable, both overall
# and for a single transaction, which is what a bank would need for fraud analysts and
# regulators.
