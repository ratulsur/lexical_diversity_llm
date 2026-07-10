"""Headless end-to-end run of Experiment 1 (train) + Experiment 3 (score tiers).

Mirrors the notebook logic so the whole pipeline can be verified without Jupyter
and without the Anthropic API (using the offline tier stand-in). Saves the
AUC-vs-tier chart to exp3_offline_demo.png and prints the numbers.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

import lexical_metrics as lm

SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
DIVERSITY_COLS = ["ttr", "herdan_c", "yule_k", "mattr", "mtld"]
LANG = "English"


def join_text(row):
    parts = [str(row.get("Upside_Review", "") or ""), str(row.get("Downside_Review", "") or "")]
    return " ".join(p.strip() for p in parts if p and p.lower() != "nan").strip()


# --- Load MAiDE-up, English subset ---
raw = pd.read_csv("../all_data.csv")
raw["text"] = raw.apply(join_text, axis=1)
raw["label"] = raw["source"].astype(int)
df = raw[(raw["Review_Language"] == LANG) & (raw["text"].str.len() > 0)].reset_index(drop=True)

feats = pd.DataFrame(df["text"].map(lm.features).tolist())
feats["label"] = df["label"].values
print(f"English reviews: {len(df)}  (real={int((feats.label==0).sum())}, llm={int((feats.label==1).sum())})")

# --- Experiment 1: train the lexical baseline (split on INDICES so we can hold
#     out the exact rows the model never saw, for an honest Experiment 3) ---
X = feats[DIVERSITY_COLS].values
y = feats["label"].values
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)
model = GradientBoostingClassifier(random_state=SEED).fit(X[tr_idx], y[tr_idx])
p = model.predict_proba(X[te_idx])[:, 1]
print(f"\nExperiment 1 (held-out MAiDE-up test):")
print(f"  ROC-AUC = {roc_auc_score(y[te_idx], p):.3f}   PR-AUC = {average_precision_score(y[te_idx], p):.3f}")

# Texts the model saw during training -- anything matching these is excluded from
# Experiment 3 so no evaluated example was also a training example.
train_texts = set(df.iloc[tr_idx]["text"])

# --- Experiment 3: score difficulty tiers on HELD-OUT data only ---
tiers = pd.read_csv("difficulty_tiers.csv")
tiers["text"] = tiers["text"].astype(str)
tiers = tiers[(tiers["text"].str.len() > 0) & (~tiers["text"].isin(train_texts))].reset_index(drop=True)
tier_feats = pd.DataFrame(tiers["text"].map(lm.features).tolist())
tier_feats["tier"] = tiers["tier"].values

# Negatives = held-out real reviews only (never seen in training).
held_out_real = feats.iloc[te_idx]
real = held_out_real[held_out_real.label == 0][DIVERSITY_COLS].values
real_y = np.zeros(len(real))
print(f"  (Exp 3 negatives: {len(real)} held-out real reviews; "
      f"{len(tiers)} tier fakes after dropping any seen in training)")
TIER_ORDER = ["lazy", "careful", "frontier"]
auc_by_tier = {}
for t in TIER_ORDER:
    fake = tier_feats[tier_feats.tier == t][DIVERSITY_COLS].values
    if len(fake) == 0:
        continue
    X_eval = np.vstack([real, fake])
    y_eval = np.concatenate([real_y, np.ones(len(fake))])
    auc_by_tier[t] = roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])

print("\nExperiment 3 (lexical detector AUC vs. generation difficulty):")
for t in TIER_ORDER:
    if t in auc_by_tier:
        print(f"  {t:9s} AUC = {auc_by_tier[t]:.3f}")

order = [t for t in TIER_ORDER if t in auc_by_tier]
vals = [auc_by_tier[t] for t in order]
plt.figure(figsize=(6, 4))
plt.plot(order, vals, "o-", color="#DD8452", lw=2, ms=9)
plt.axhline(0.5, ls="--", color="gray", lw=0.9, label="chance (AUC=0.5)")
for x, v in zip(order, vals):
    plt.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 10), ha="center")
plt.ylim(0.45, 1.0); plt.ylabel("lexical detector ROC-AUC")
plt.xlabel("generation difficulty  (lazy → careful → frontier)")
plt.title("Where the cheap method breaks (OFFLINE demo stand-in)")
plt.legend(); plt.tight_layout()
plt.savefig("exp3_offline_demo.png", dpi=120)
print("\nSaved chart -> exp3_offline_demo.png")
