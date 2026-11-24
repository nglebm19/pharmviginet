"""Experimental Method F (experiments/graph_folds) on synthetic ingredient sets."""
import numpy as np
import pandas as pd
import pytest

from experiments.graph_folds.components import (
    UnionFind, component_folds, graph_summary, ingredient_components,
)
from pharmviginet.data import task_a


def _ingredients():
    # chain a-b, b-c joins {a,b,c}; d|e separate; f..t singletons with varied weight
    rows = ["a|b"] * 5 + ["b|c"] * 3 + ["a"] * 10 + ["c"] * 2 + ["d|e"] * 4 + ["d"] * 6
    rows += [f"s{i:02d}" for i in range(20) for _ in range(i % 7 + 1)]
    return pd.Series(rows)


def test_union_find_is_transitive():
    uf = UnionFind("abcd")
    uf.union("a", "b")
    uf.union("c", "b")
    assert uf.find("a") == uf.find("c") != uf.find("d")


def test_components_and_weights():
    comps = ingredient_components(_ingredients()).set_index("ingredient")
    assert comps.loc["c", "component"] == comps.loc["a", "component"] == "a"
    assert comps.loc["e", "component"] == "d"
    assert comps.loc["a", "pairs"] == 20 and comps.loc["d", "pairs"] == 10
    s = graph_summary(_ingredients())
    assert s["combination_pairs"] == 12 and s["edges"] == 3 and s["largest_component_ingredients"] == 3


def test_no_pair_dropped_and_ingredient_disjoint():
    ing = _ingredients()
    folds = component_folds(ing, n_folds=3)
    assert folds["ingredient"].is_unique
    pairs = pd.DataFrame({"drugname": ing, "pt": "X", "ingredients": ing, "label": 0})
    kept, dropped = task_a.pair_folds(pairs, folds)
    assert dropped == 0 and len(kept) == len(pairs)
    load = kept["fold"].value_counts()
    assert load.max() / load.min() < 1.3


def test_deterministic_and_order_invariant():
    ing = _ingredients()
    a = component_folds(ing)
    b = component_folds(ing.sample(frac=1, random_state=3).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)
    # seeded variants only reorder equal-size components; still valid partitions
    s1, s2 = component_folds(ing, seed=1), component_folds(ing, seed=1)
    pd.testing.assert_frame_equal(s1, s2)


def test_assignment_never_sees_labels():
    # the API takes only ingredient strings, so labels cannot influence folds
    ing = _ingredients()
    rng = np.random.default_rng(0)
    p1 = pd.DataFrame({"ingredients": ing, "label": rng.integers(0, 2, len(ing))})
    p2 = p1.assign(label=1 - p1["label"])
    pd.testing.assert_frame_equal(component_folds(p1["ingredients"]), component_folds(p2["ingredients"]))


@pytest.mark.parametrize("n_folds", [True, False, 1, 0, -3, 5.0, 2.5, "5", None])
def test_invalid_n_folds_rejected(n_folds):
    with pytest.raises(ValueError, match="n_folds must be an integer >= 2"):
        component_folds(_ingredients(), n_folds=n_folds)


@pytest.mark.parametrize("n_folds", [2, 5])
def test_valid_n_folds_accepted(n_folds):
    folds = component_folds(_ingredients(), n_folds=n_folds)
    assert sorted(folds["fold"].unique()) == list(range(n_folds))


def _big_plus_singletons(big_pairs, n_small, small_pairs):
    """One two-ingredient component with big_pairs pairs + n_small singleton components."""
    rows = ["big1|big2"] * big_pairs
    rows += [f"s{i:02d}" for i in range(n_small) for _ in range(small_pairs)]
    return pd.Series(rows)


@pytest.mark.parametrize("ingredients", [pd.Series(["a", "b", "c"]), pd.Series([], dtype=str)])
def test_fewer_components_than_folds_rejected(ingredients):
    with pytest.raises(ValueError, match="connected components cannot fill 5 folds"):
        component_folds(ingredients, n_folds=5)


def test_severe_imbalance_rejected():
    # largest component = 50% of pairs → its fold holds 100 vs 25 elsewhere (ratio 4.0)
    with pytest.raises(ValueError, match="fold balance 4.00 exceeds MAX_PAIR_RATIO"):
        component_folds(_big_plus_singletons(100, 20, 5), n_folds=5)


def test_component_above_one_fifth_accepted_when_balanced():
    # largest component = 25% of pairs (> 1/5) but folds stay within MAX_PAIR_RATIO
    ing = _big_plus_singletons(250, 30, 25)
    folds = component_folds(ing, n_folds=5)
    pairs = pd.DataFrame({"drugname": ing, "pt": "X", "ingredients": ing, "label": 0})
    kept, dropped = task_a.pair_folds(pairs, folds)
    load = kept["fold"].value_counts()
    assert dropped == 0 and len(kept) == len(ing)
    assert load.max() / len(ing) == 0.25
    assert load.max() / load.min() <= task_a.MAX_PAIR_RATIO
    assert folds.set_index("ingredient").loc[["big1", "big2"], "fold"].nunique() == 1
