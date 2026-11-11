"""Read-only local results verification; never downloads or rewrites pipeline artifacts.

Run from the repository root: python -m scripts.audit_results --output /tmp/results.json
Add --counts to reconstruct Task A counts/folds from master (more memory/time).
The output contains aggregate statistics and fingerprints, never SIDER label rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pharmviginet import config as c
from pharmviginet.data import task_a
from pharmviginet.labels.build_labels import label_pairs
from pharmviginet.models.baseline import evaluate_task_a
from clean_faers import check_split


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    result = {'bytes': stat.st_size, 'sha256': h.hexdigest(),
              'mtime_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}
    if path.suffix == '.parquet':
        result['rows'] = pq.ParquetFile(path).metadata.num_rows
    return result


def compare(actual, expected, path='result'):
    """Compare all replayed fields, accepting only floating point noise."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            compare(actual[key], expected[key], f'{path}.{key}')
    elif isinstance(expected, (int, float)):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12, err_msg=path)
    else:
        assert actual == expected, path


def audit(counts=False):
    pairs = pd.read_parquet(c.TASK_A_PAIRS)
    labels = pd.read_parquet(c.LABELS_SIDER)
    mapping = pd.read_parquet(c.RXNORM_MAP)
    sider = pd.read_parquet(c.SIDER_PAIRS)
    folds = pd.read_parquet(c.TASK_A_FOLDS)
    saved = json.loads(c.TASK_A_RESULTS.read_text())
    assert saved['config']['feature_cutoff_year'] == c.FEATURE_CUTOFF_YEAR == 2022
    assert saved['config']['min_reports'] == c.TASK_A_MIN_REPORTS == 3
    assert saved['config']['n_folds'] == c.TASK_A_N_FOLDS == 5
    assert saved['config']['seed'] == c.TASK_A_SEED == 0
    assert pairs[task_a.PAIR].duplicated().sum() == 0
    assert pairs.a.min() >= 3
    assigned, dropped = task_a.pair_folds(pairs.drop(columns='fold'), folds)
    assert dropped == 0
    np.testing.assert_array_equal(assigned.fold, pairs.fold)
    quality, bad = task_a.fold_quality(pairs)
    assert not bad, bad
    # Reapply SIDER rules to every stored all-years label; completeness is separate.
    relabeled = label_pairs(labels[task_a.PAIR], mapping, sider)
    pd.testing.assert_frame_equal(
        labels[task_a.LABEL_COLS].sort_values(task_a.PAIR).reset_index(drop=True),
        relabeled[task_a.LABEL_COLS].sort_values(task_a.PAIR).reset_index(drop=True))
    assert set(labels.source) == {'sider4.1'}
    assert not check_split({'train': c.TRAIN_PARQUET, 'val': c.VAL_PARQUET,
                            'test': c.TEST_PARQUET}, c.SPLIT_MANIFEST, c.MASTER_PARQUET)
    print('Replaying Task A evaluation from stored pairs (including control)', flush=True)
    replay = json.loads(json.dumps(evaluate_task_a()))
    compare(replay, saved)
    verification = {'stored_labels_reapplied': True, 'fold_quality': True,
                    'all_metrics_replayed': True, 'split_metadata_checked': True,
                    'counts_reconstructed': counts}
    if counts:
        print('Reconstructing cutoff counts and folds from master', flush=True)
        ct, drugs = task_a.feature_counts()
        eligible = task_a.eligible_pairs(ct, labels)
        expected_folds = task_a.assign_folds(eligible.ingredients)
        pd.testing.assert_frame_equal(expected_folds, folds)
        expected, n_dropped = task_a.pair_folds(eligible, folds)
        prior = tuple(saved['config']['mgps_prior'][k] for k in ['a1', 'b1', 'a2', 'b2', 'p'])
        expected = task_a.add_scores(expected, prior).merge(drugs, on='drugname')
        expected['n_ingredients'] = expected.ingredients.str.count(r'\|') + 1
        pd.testing.assert_frame_equal(
            expected[pairs.columns].sort_values(task_a.PAIR).reset_index(drop=True),
            pairs.sort_values(task_a.PAIR).reset_index(drop=True), rtol=1e-7, atol=1e-9)
        verification.update(eligible_pairs=len(eligible), combination_pairs_dropped=n_dropped,
                            scores_recomputed_with_stored_prior=True)
        del ct, expected, eligible

    print('Scanning master coverage and mapping denominator in bounded batches', flush=True)
    years, eras, quarters, names, modern_names, roles = (Counter() for _ in range(6))
    for batch in pq.ParquetFile(c.MASTER_PARQUET).iter_batches(
            batch_size=250000, columns=['year', 'era', 'quarter', 'drugname', 'role_cod']):
        df = batch.to_pandas()
        for counter, col in [(years, 'year'), (eras, 'era'), (quarters, 'quarter'),
                             (names, 'drugname'), (roles, 'role_cod')]:
            counter.update(df[col].value_counts().to_dict())
        modern_names.update(df.loc[df.era == 'faers', 'drugname'].value_counts().to_dict())
    assert set(mapping.drugname) == {n for n, total in names.items() if total >= 10}
    assert roles.keys() == {'PS'}
    split = json.loads(c.SPLIT_MANIFEST.read_text())
    assert dict(years) == {int(y): n for v in split['splits'].values() for y, n in v['rows_by_year'].items()}
    cache = pd.read_parquet(c.EXTERNAL / 'rxnav_cache_faers.parquet')
    joined = mapping.merge(cache, on='clean', suffixes=('', '_cache'), validate='many_to_one')
    assert len(joined) == len(mapping)
    assert joined.ingredients.fillna('').equals(joined.ingredients_cache.fillna(''))
    paths = [c.MASTER_PARQUET, c.RXNORM_MAP, c.SIDER_PAIRS, c.LABELS_SIDER,
             c.TASK_A_PAIRS, c.TASK_A_FOLDS, c.TASK_A_PAIRS.with_suffix('.prior.json'),
             c.TASK_A_RESULTS, c.SPLIT_MANIFEST, c.PROCESSED / 'cases_deduped.parquet',
             c.EXTERNAL / 'rxnav_cache_faers.parquet', c.EXTERNAL / 'rxnav_cache_sider.parquet',
             c.SIDER_DIR / 'meddra_all_se.tsv.gz', c.SIDER_DIR / 'drug_names.tsv']
    paths += sorted(c.LOGS.glob('*.json'))
    paths += [c.LOGS / 'drug_norm.log', c.TRAIN_PARQUET, c.VAL_PARQUET, c.TEST_PARQUET]
    source_paths = [Path('clean_faers.py'), Path('scripts/audit_results.py'),
                    Path('scripts/leakage_check.py')]
    source_paths += sorted(Path('pharmviginet').rglob('*.py'))
    print('Fingerprinting evidence', flush=True)
    return {
        'schema_version': 1,
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'audit_checkout_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'original_artifact_producer_commit': None,
        'provenance_limit': 'Original runs lack source hashes, commit IDs and dependency versions. '
                            'This is a retrospective verification against local artifacts, not a raw-data rebuild. '
                            'mtimes are filesystem evidence, not authenticated run dates. '
                            'MGPS prior was reused, not refitted; all-years label completeness was not rebuilt.',
        'environment': {n: importlib.metadata.version(n) for n in
                        ['numpy', 'pandas', 'pyarrow', 'scipy', 'scikit-learn']},
        'verification': verification,
        'config': saved['config'],
        'metric': {'name': 'auc_strat', 'min_pt_pairs': 20, 'both_classes_required': True,
                   'within_fold_weight': 'PT pair count', 'fold_weight': 'equal', 'sd_ddof': 1},
        'coverage': {'master_rows': sum(years.values()), 'rows_by_year': dict(sorted(years.items())),
                     'rows_by_era': dict(eras), 'quarters': sorted(quarters),
                     'role_counts': dict(roles)},
        'mapping': {'names': len(mapping), 'mapped': int(mapping.ingredients.fillna('').ne('').sum()),
                    'mapped_fraction': float(mapping.ingredients.fillna('').ne('').mean()),
                    'min_master_rows': 10, 'modern_only_names_ge10': sum(n >= 10 for n in modern_names.values()),
                    'cache_matches_mapping': True, 'cache_clean_names': len(cache)},
        'labels': {'all_years_pairs': len(labels), 'positive_pairs': int(labels.label.sum()),
                   'positive_fraction': float(labels.label.mean()), 'drug_names': labels.drugname.nunique(),
                   'sider_pairs': len(sider), 'sider_ingredients': sider.ingredient.nunique()},
        'fold_quality': quality.reset_index().drop(columns='top1_ingredient').to_dict('records'),
        'results': replay,
        'artifacts': {str(p.relative_to(c.ROOT)): fingerprint(p) for p in dict.fromkeys(paths)},
        'source_fingerprints': {str(p): fingerprint(p) for p in source_paths},
        'reproduce_verification': 'python -m scripts.audit_results --counts --output /tmp/results_manifest.json',
        'reproduce_task_a': ['python -m pharmviginet.data.task_a',
                             'python -m pharmviginet.models.baseline --task a'],
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.counts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'All checks passed; wrote {args.output}')
