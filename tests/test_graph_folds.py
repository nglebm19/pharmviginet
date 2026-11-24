"""Experimental Method F (experiments/graph_folds) on synthetic ingredient sets."""
import numpy as np
import pandas as pd

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
