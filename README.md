# Can a 100-Year-Old Statistic Catch ChatGPT?

Lexical-diversity statistics vs. LLM-generated fake hotel reviews — a reproducible benchmark
backing a TDS-standard blog post.

**Thesis.** Lexical-diversity metrics (Yule's K, Herdan's C, MTLD, MATTR) are a cheap, interpretable,
near-zero-cost first-line filter that catches *bulk / lazily-generated* LLM review spam — but degrade
against carefully-prompted output. Knowing *where* the boundary sits beats any single AUC number.

## Data

`all_data.csv` (MAiDE-up) in the parent `stat-analysis/` folder. ~20k hotel reviews.

| column | meaning |
|---|---|
| `source = 0` | real human review → **label 0** |
| `source = 1` | GPT-generated fake → **label 1** |
| `Upside_Review` / `Downside_Review` | pros / cons text (joined into one `text` field) |
| `Review_Language` | 10 languages; we restrict to `English` (see caveat) |
| `Sentiment` | POS / NEG |

**Language caveat.** The lexical metrics assume space-delimited tokenization, so metric-based claims
are valid only on space-delimited languages (English, French, German, …). The Chinese/Korean rows need
a proper word segmenter before the metrics mean anything — don't report them as-is.

**License.** MAiDE-up comes from [github.com/MichiganNLP/multilingual_reviews_deception](https://github.com/MichiganNLP/multilingual_reviews_deception)
([paper](https://arxiv.org/abs/2404.12938)). Verify its license before publishing results built on it.

## Layout

```
lexical-diversity-llm-reviews/
├── README.md
├── requirements.txt
├── lexical_metrics.py        # metric implementations (TTR, Herdan's C, Yule's K, MATTR, MTLD)
├── test_lexical_metrics.py   # sanity tests — run these first
├── generate_difficulty_tiers.py  # Experiment 3: generate lazy/careful/frontier fakes via the Anthropic API
└── notebooks/
    └── lexical_diversity_vs_llm_reviews.ipynb   # the seeded end-to-end study
```

## Run

```bash
cd lexical-diversity-llm-reviews
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python test_lexical_metrics.py          # 8 sanity tests, all pure-stdlib
jupyter lab notebooks/lexical_diversity_vs_llm_reviews.ipynb
```

Everything is seeded (`SEED = 42`: `random`, `numpy`, `PYTHONHASHSEED`). The train/test split is
stratified and seeded, so results are reproducible run-to-run.

## Notebook flow

1. Setup & global seed
2. Load MAiDE-up, build the `text` field and `label`
3. Class balance + review-length sanity check
4. Zipf curves — richness-as-signal intuition
5. Lexical-diversity features + per-metric distributions
6. **Experiment 1** — lexical baseline (logreg + gradient boosting): ROC-AUC, PR-AUC, precision@10%
7. **Length ablation** — is it diversity, or just length? (the credibility check)
8. **Experiment 2 (stub)** — neural detector on the same split + cost/latency
9. **Experiment 3** — cross-generator tiers (Haiku/Sonnet/Opus): who actually evades the detector
10. Takeaway

## What's done vs. what's next

- ✅ Metrics, tests, data loader, Experiment 1, length ablation — runnable now.
- ✅ **Experiment 3** (cross-generator + cross-vendor tiers) — `generate_difficulty_tiers.py` produces
  fakes from Claude Haiku/Sonnet/Opus **and Meta Llama 3.3 (via Groq)**; the notebook scores them. Real
  finding: it's a *verbosity detector* — wordy machine text is caught (Llama, the wordiest, easiest of
  all at 0.88), the terse *lazy* tier evades (0.66); the ranking holds across vendors, the scores don't.
- ⏳ Experiment 2 (neural baseline) is stubbed with the exact next steps.

### Running Experiment 3

```bash
export ANTHROPIC_API_KEY=...   # Claude tiers (lazy/careful/frontier)
export GROQ_API_KEY=...         # cross-vendor tier (Llama 3.3 70B, OpenAI-compatible API)
python generate_difficulty_tiers.py --n 40      # writes difficulty_tiers.csv (a few $ of API)
```

Then run the Experiment 3 cells in the notebook. The tiers span a Claude capability gradient
(Haiku 4.5 → Sonnet 5 → Opus 4.8) plus a cross-**vendor** tier (Meta Llama 3.3 at the `careful` prompt
effort). The idempotent script resumes on rerun and only generates missing `(hotel, tier, sentiment)`
rows; delete the CSV to regenerate. No API key? `--offline` builds a demo stand-in from local data
(different, peaked shape — see the notebook note). The committed `difficulty_tiers.csv` is the real
API output the write-up cites.
