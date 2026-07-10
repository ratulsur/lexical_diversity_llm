"""Sanity tests so readers trust the metric implementations.

Run: python -m pytest test_lexical_metrics.py   (or just: python test_lexical_metrics.py)
"""

import math

import lexical_metrics as lm


def test_tokenize_lowercases_and_splits():
    assert lm.tokenize("The cat, the CAT!") == ["the", "cat", "the", "cat"]


def test_ttr_bounds():
    assert lm.ttr([]) == 0.0
    assert lm.ttr(["a", "b", "c"]) == 1.0            # all unique
    assert lm.ttr(["a", "a", "a", "a"]) == 0.25      # one type / four tokens


def test_herdan_c_all_unique_is_one():
    toks = ["a", "b", "c", "d"]
    assert math.isclose(lm.herdan_c(toks), 1.0)      # log V == log N


def test_yule_k_more_repetition_is_higher():
    diverse = ["a", "b", "c", "d", "e", "f", "g", "h"]
    repetitive = ["a", "a", "a", "a", "b", "b", "b", "b"]
    assert lm.yule_k(repetitive) > lm.yule_k(diverse)


def test_yule_k_all_unique_is_zero():
    # sum f^2 == N when every type appears once -> K == 0
    assert math.isclose(lm.yule_k(["a", "b", "c", "d"]), 0.0)


def test_mattr_falls_back_to_ttr_when_short():
    toks = ["a", "b", "c"]
    assert lm.mattr(toks, window=50) == lm.ttr(toks)


def test_mtld_higher_for_diverse_text():
    diverse = ["w%d" % i for i in range(200)]              # every token unique
    repetitive = (["a", "b"] * 100)                        # two types, alternating
    assert lm.mtld(diverse) > lm.mtld(repetitive)


def test_features_keys():
    f = lm.features("the cat sat on the mat")
    assert set(f) == set(lm.FEATURE_NAMES)
    assert f["n_tokens"] == 6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\nAll %d tests passed." % len(fns))
