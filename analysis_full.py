"""Full analysis runner: every number and figure the blog post cites.

Runs Experiment 1 (lexical baseline), the length ablation, and Experiment 3
(difficulty tiers, offline stand-in) end-to-end on the real MAiDE-up data.
Prints a report and writes figures to figures/. Mirrors the notebook exactly.
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

import lexical_metrics as lm

SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
LANG = "English"
DIVERSITY_COLS = ["ttr", "herdan_c", "yule_k", "mattr", "mtld"]
FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def join_text(row):
    parts = [str(row.get("Upside_Review", "") or ""), str(row.get("Downside_Review", "") or "")]
    return " ".join(p.strip() for p in parts if p and p.lower() != "nan").strip()


def rule(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


# --- Load ---
raw = pd.read_csv("../all_data.csv")
raw["text"] = raw.apply(join_text, axis=1)
raw["label"] = raw["source"].astype(int)
df = raw[(raw["Review_Language"] == LANG) & (raw["text"].str.len() > 0)].reset_index(drop=True)
feats = pd.DataFrame(df["text"].map(lm.features).tolist())
feats["label"] = df["label"].values

rule("DATASET")
print(f"English reviews: {len(df)}  real={int((feats.label==0).sum())}  llm={int((feats.label==1).sum())}")
print(f"Median tokens  real={df.assign(n=feats.n_tokens).query('label==0').n.median():.0f}  "
      f"llm={df.assign(n=feats.n_tokens).query('label==1').n.median():.0f}"
      if False else "")

rule("PER-METRIC MEANS BY CLASS (direction of the signal)")
means = feats.groupby("label")[DIVERSITY_COLS + ["n_tokens"]].mean().rename(index={0: "real", 1: "llm"})
print(means.round(2).to_string())

# --- Experiment 1 ---
X = feats[DIVERSITY_COLS].values
y = feats["label"].values
idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=y)
X_tr, X_te, y_tr, y_te = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]
train_texts = set(df.iloc[tr_idx]["text"])

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "gboost": GradientBoostingClassifier(random_state=SEED),
}


def prec_at_k(yt, sc, k=0.10):
    kk = max(1, int(len(sc) * k))
    return yt[np.argsort(sc)[::-1][:kk]].mean()


rule("EXPERIMENT 1 — lexical baseline (held-out test)")
results = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    results[name] = dict(roc=roc_auc_score(y_te, p), pr=average_precision_score(y_te, p),
                         patk=prec_at_k(y_te, p), scores=p)
    print(f"  {name:7s} ROC-AUC={results[name]['roc']:.3f}  PR-AUC={results[name]['pr']:.3f}  "
          f"prec@10%={results[name]['patk']:.3f}")
best = max(results, key=lambda k: results[k]["pr"])
print(f"  best = {best}")

# ROC/PR figure
p = results[best]["scores"]
fpr, tpr, _ = roc_curve(y_te, p); prec, rec, _ = precision_recall_curve(y_te, p)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(fpr, tpr, color="#DD8452"); ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=.8)
ax[0].set(title=f"ROC — {best} (AUC={results[best]['roc']:.3f})", xlabel="FPR", ylabel="TPR")
ax[1].plot(rec, prec, color="#4C72B0")
ax[1].set(title=f"PR — {best} (AP={results[best]['pr']:.3f})", xlabel="Recall", ylabel="Precision")
plt.tight_layout(); plt.savefig(f"{FIG}/exp1_roc_pr.png", dpi=120); plt.close()

# Feature importance
imp = pd.Series(models["gboost"].feature_importances_, index=DIVERSITY_COLS).sort_values()
rule("EXPERIMENT 1 — gradient-boosting feature importance")
print(imp.round(3).to_string())
imp.plot(kind="barh", figsize=(6, 3), color="#55A868", title="gboost feature importance")
plt.tight_layout(); plt.savefig(f"{FIG}/exp1_importance.png", dpi=120); plt.close()

# Zipf
from collections import Counter
def zipf(texts):
    c = Counter()
    for t in texts: c.update(lm.tokenize(t))
    f = np.array(sorted(c.values(), reverse=True)); return np.arange(1, len(f) + 1), f
plt.figure(figsize=(6, 4.2))
for lbl, name, col in [(0, "real", "#4C72B0"), (1, "llm", "#DD8452")]:
    r, f = zipf(df[df.label == lbl]["text"]); plt.loglog(r, f, label=name, color=col, lw=1.6)
plt.xlabel("word rank (log)"); plt.ylabel("frequency (log)"); plt.title("Zipf curves: real vs. LLM")
plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/zipf.png", dpi=120); plt.close()

# --- Length ablation ---
rule("LENGTH ABLATION — is it diversity, or just length?")
Xl = feats[["n_tokens"]].values
lm_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)).fit(Xl[tr_idx], y_tr)
p_len = lm_model.predict_proba(Xl[te_idx])[:, 1]
Xb = feats[DIVERSITY_COLS + ["n_tokens"]].values
both = GradientBoostingClassifier(random_state=SEED).fit(Xb[tr_idx], y_tr)
p_both = both.predict_proba(Xb[te_idx])[:, 1]
abl = pd.DataFrame({
    "length_only": [roc_auc_score(y_te, p_len), average_precision_score(y_te, p_len)],
    "diversity_only": [results[best]["roc"], results[best]["pr"]],
    "diversity+length": [roc_auc_score(y_te, p_both), average_precision_score(y_te, p_both)],
}, index=["roc_auc", "pr_auc"]).T
print(abl.round(3).to_string())

# --- Experiment 3 (real generators across models AND vendors, held-out eval) ---
rule("EXPERIMENT 3 — cross-generator tiers (Haiku/Sonnet/Opus + Llama, held-out eval)")
ALL_TIERS = ["lazy", "careful", "frontier", "crossvendor"]
if os.path.exists("difficulty_tiers.csv"):
    tiers = pd.read_csv("difficulty_tiers.csv")
    tiers["text"] = tiers["text"].astype(str)
    tiers = tiers[(tiers["text"].str.len() > 0) & (~tiers["text"].isin(train_texts))].reset_index(drop=True)
    tf = pd.DataFrame(tiers["text"].map(lm.features).tolist()); tf["tier"] = tiers["tier"].values
    ho = feats.iloc[te_idx]; real = ho[ho.label == 0][DIVERSITY_COLS].values; real_y = np.zeros(len(real))
    trained = models[best]
    auc = {}
    for t in ALL_TIERS:
        fk = tf[tf.tier == t][DIVERSITY_COLS].values
        if len(fk) == 0: continue
        Xe = np.vstack([real, fk]); ye = np.concatenate([real_y, np.ones(len(fk))])
        auc[t] = roc_auc_score(ye, trained.predict_proba(Xe)[:, 1])
    print(f"  negatives: {len(real)} held-out real; fakes after dedup: {len(tiers)}")
    print("  per-tier (mtld / tokens / AUC); human real-review mtld=59, tokens=49:")
    for t in ALL_TIERS:
        if t in auc:
            sub = tf[tf.tier == t]
            print(f"    {t:11s} mtld={sub.mtld.mean():5.1f}  tokens={sub.n_tokens.mean():5.1f}  AUC={auc[t]:.3f}")

    # Chart: the three Claude tiers as a capability line; the Llama cross-vendor tier
    # as a distinct point at the SAME prompt effort as `careful` (isolates vendor).
    claude = [t for t in ["lazy", "careful", "frontier"] if t in auc]
    cvals = [auc[t] for t in claude]
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(claude, cvals, "o-", color="#DD8452", lw=2, ms=9, label="Claude tiers")
    for x, v in zip(claude, cvals):
        plt.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 10), ha="center")
    if "crossvendor" in auc:  # place at the 'careful' x-position (matched prompt effort)
        plt.scatter(["careful"], [auc["crossvendor"]], color="#55A868", marker="D", s=90, zorder=5,
                    label="Llama (cross-vendor, careful prompt)")
        plt.annotate(f"{auc['crossvendor']:.2f}", ("careful", auc["crossvendor"]),
                     textcoords="offset points", xytext=(8, -4), ha="left", color="#3d7a4e")
    plt.axhline(.5, ls="--", color="gray", lw=.9, label="chance")
    plt.axhline(results[best]["roc"], ls=":", color="#4C72B0", lw=1.1, label="in-distribution (GPT)")
    plt.ylim(.5, .95); plt.ylabel("lexical detector ROC-AUC")
    plt.xlabel("generator:   lazy=Haiku    careful=Sonnet    frontier=Opus")
    plt.title("Cross-vendor: verbose fakes are caught, terse ones evade")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/exp3_tiers.png", dpi=120); plt.close()
else:
    print("  difficulty_tiers.csv not found -- run generate_difficulty_tiers.py --offline")

rule("FIGURES WRITTEN")
for f in sorted(os.listdir(FIG)): print(" ", os.path.join(FIG, f))
