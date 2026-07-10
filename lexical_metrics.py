"""Lexical-diversity metrics for the "old stats vs. LLM fake reviews" study.

All metrics operate on a pre-tokenized list of tokens so tokenization is
decided once, up front (see `tokenize`). Metrics implemented:

    - ttr        : Type-Token Ratio (length-sensitive, kept as a baseline)
    - herdan_c   : Herdan's C / log-TTR (mildly length-robust)
    - yule_k     : Yule's K (the historical repetition/richness signal, 1944)
    - mattr      : Moving-Average TTR (length-robust)
    - mtld       : Measure of Textual Lexical Diversity (length-robust)

Why include the length-robust trio (MTLD/MATTR) at all? Because raw TTR and
Yule's K are confounded by document length, and reviews vary a lot in length.
Reporting only length-sensitive metrics is the classic way to accidentally
build a length detector instead of a diversity detector. The length ablation
in the notebook guards against exactly that.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List

# Unicode-aware word tokenizer. NOTE: this is whitespace/word-boundary based and
# is only meaningful for space-delimited languages. For Chinese/Korean/Japanese
# it does not segment words correctly, so restrict metric-based claims to the
# English (or other space-delimited) subset. See README.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lowercase, Unicode word-boundary tokenization. One place, one decision."""
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


def ttr(tokens: List[str]) -> float:
    """Type-Token Ratio = |types| / |tokens|. Length-sensitive by design."""
    n = len(tokens)
    if n == 0:
        return 0.0
    return len(set(tokens)) / n


def herdan_c(tokens: List[str]) -> float:
    """Herdan's C = log(V) / log(N). Undefined for N<=1; returns 0.0 there."""
    n = len(tokens)
    if n <= 1:
        return 0.0
    v = len(set(tokens))
    return math.log(v) / math.log(n)


def yule_k(tokens: List[str]) -> float:
    """Yule's K (1944).

    K = 10^4 * (sum_i f_i^2 - N) / N^2

    where f_i is the frequency of type i and N the token count. Higher K means
    more repetition / lower diversity -- the templated-text signal. Length bias
    is much milder than TTR but not zero (hence MTLD/MATTR alongside).
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    freqs = Counter(tokens)
    sum_sq = sum(f * f for f in freqs.values())
    return 1e4 * (sum_sq - n) / (n * n)


def mattr(tokens: List[str], window: int = 50) -> float:
    """Moving-Average TTR over a sliding window. Length-robust.

    If the text is shorter than the window, falls back to plain TTR.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return ttr(tokens)
    ttrs = []
    for i in range(n - window + 1):
        window_tokens = tokens[i : i + window]
        ttrs.append(len(set(window_tokens)) / window)
    return sum(ttrs) / len(ttrs)


def _mtld_pass(tokens: List[str], threshold: float) -> float:
    """One directional MTLD pass -> total_tokens / factor_count."""
    factor_count = 0.0
    token_count = 0
    types: set = set()
    for tok in tokens:
        token_count += 1
        types.add(tok)
        cur_ttr = len(types) / token_count
        if cur_ttr <= threshold:
            factor_count += 1.0
            token_count = 0
            types = set()
    if token_count > 0:  # trailing partial factor
        cur_ttr = len(types) / token_count
        factor_count += (1.0 - cur_ttr) / (1.0 - threshold)
    if factor_count == 0:
        return float(len(tokens))
    return len(tokens) / factor_count


def mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """Bidirectional MTLD (McCarthy & Jarvis, 2010). Length-robust.

    Returns the mean of the forward and reverse passes. For very short texts
    the estimate is unstable; the notebook filters on a minimum token count.
    """
    if len(tokens) == 0:
        return 0.0
    forward = _mtld_pass(tokens, threshold)
    reverse = _mtld_pass(list(reversed(tokens)), threshold)
    return (forward + reverse) / 2.0


# Convenience: compute the full feature vector for one text.
FEATURE_NAMES = ["ttr", "herdan_c", "yule_k", "mattr", "mtld", "n_tokens"]


def features(text: str) -> dict:
    """All lexical features for a single text, keyed by FEATURE_NAMES."""
    toks = tokenize(text)
    return {
        "ttr": ttr(toks),
        "herdan_c": herdan_c(toks),
        "yule_k": yule_k(toks),
        "mattr": mattr(toks),
        "mtld": mtld(toks),
        "n_tokens": len(toks),
    }
