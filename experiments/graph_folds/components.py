"""Experimental Task A fold assignment by ingredient connected components (Method F).

Not used by the production pipeline (`pharmviginet/data/task_a.py`). See FINDINGS.md.

Ingredients are nodes; two ingredients are linked when they occur together in one
FAERS product's ingredient set. Union-Find gives the connected components. Whole
components are packed into folds largest first (LPT: each goes to the fold with the
fewest pairs so far), so no pair can span folds and none are dropped.

Assignment reads only the ingredient sets (one entry per pair), never labels.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from pharmviginet.data.task_a import MAX_PAIR_RATIO


class UnionFind:
    """Disjoint sets with path halving and union by size."""

    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.size = dict.fromkeys(self.parent, 1)

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def ingredient_components(ingredients: pd.Series) -> pd.DataFrame:
    """'|'-joined ingredient sets, one per pair → [ingredient, component, pairs].

    `component` is the smallest ingredient name in the component (a stable id);
    `pairs` is the number of pairs whose ingredient set lies in that component.
    """
    sets = ingredients.value_counts()
    uf = UnionFind(sorted({i for s in sets.index for i in s.split("|")}))
    for s in sets.index:
        first, *rest = s.split("|")
        for other in rest:
            uf.union(first, other)
    members: dict[str, list[str]] = {}
    for ing in uf.parent:
        members.setdefault(uf.find(ing), []).append(ing)
    comp_id = {ing: min(group) for group in members.values() for ing in group}
    weight: Counter[str] = Counter()
    for s, n in sets.items():
        weight[comp_id[str(s).split("|")[0]]] += int(n)
    out = pd.DataFrame({"ingredient": list(comp_id), "component": list(comp_id.values())})
    out["pairs"] = out["component"].map(weight)
    return out.sort_values("ingredient", ignore_index=True)


def component_folds(ingredients: pd.Series, n_folds: int = 5, seed: int | None = None) -> pd.DataFrame:
    """→ [ingredient, fold], the same shape as `task_a.assign_folds`.

    seed=None: deterministic, components ordered by (pairs desc, component id).
    seed=int: equal-size components are ordered randomly instead (tie-breaks only),
    to probe how sensitive results are to the partition.
    """
    # bool is a subclass of int, so it must be excluded explicitly.
    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError(f"n_folds must be an integer >= 2, got {n_folds!r}")
    comps = ingredient_components(ingredients)
    order = comps.drop_duplicates("component")[["component", "pairs"]]
    # With at least n_folds components (all with pairs > 0), LPT's first n_folds picks
    # land on the n_folds empty folds, so every fold is non-empty below.
    if len(order) < n_folds:
        raise ValueError(f"{len(order)} connected components cannot fill {n_folds} folds; "
                         f"need at least {n_folds}")
    if seed is None:
        order = order.sort_values(["pairs", "component"], ascending=[False, True])
    else:
        rng = np.random.default_rng(seed)
        order = order.assign(tie=rng.permutation(len(order))).sort_values(
            ["pairs", "tie"], ascending=[False, True])
    load = np.zeros(n_folds)
    fold_of = {}
    for comp, pairs in zip(order["component"], order["pairs"]):
        k = int(np.argmin(load))
        load[k] += pairs
        fold_of[comp] = k
    # Judge the partition by fold balance, not by component size: a component above
    # 1/n_folds of the pairs is fine as long as the folds stay within the ratio limit.
    ratio = load.max() / load.min()
    if ratio > MAX_PAIR_RATIO:
        top_comp, top_pairs = order["component"].iloc[0], order["pairs"].iloc[0]
        raise ValueError(f"fold balance {ratio:.2f} exceeds MAX_PAIR_RATIO {MAX_PAIR_RATIO} "
                         f"(largest / smallest fold, in pairs); largest component "
                         f"{top_comp!r} holds {top_pairs / load.sum():.1%} of pairs")
    out = comps[["ingredient"]].copy()
    out["fold"] = comps["component"].map(fold_of).astype(int)
    return out


def graph_summary(ingredients: pd.Series) -> dict:
    """Aggregate component statistics (no labels)."""
    comps = ingredient_components(ingredients)
    sets = ingredients.value_counts()
    is_combo = pd.Series(sets.index.str.contains("|", regex=False), index=sets.index)
    by_comp = comps.drop_duplicates("component").sort_values("pairs", ascending=False)
    sizes = comps["component"].value_counts()
    edges = {tuple(sorted((a, b))) for s in sets.index[is_combo]
             for a in s.split("|") for b in s.split("|") if a < b}
    n = int(sets.sum())
    return {
        "pairs": n, "ingredient_sets": len(sets),
        "combination_sets": int(is_combo.sum()), "combination_pairs": int(sets[is_combo].sum()),
        "ingredients": len(comps), "edges": len(edges), "components": int(len(by_comp)),
        "singleton_components": int((sizes == 1).sum()),
        "largest_component_ingredients": int(sizes.max()),
        "largest_components_pairs": by_comp["pairs"].head(5).astype(int).tolist(),
        "largest_component_share": float(by_comp["pairs"].iloc[0] / n),
    }
