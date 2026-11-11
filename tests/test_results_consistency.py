"""Keep published claims tied to the audited snapshot, without requiring local data."""
import hashlib
import json
from pathlib import Path

import numpy as np

from pharmviginet import config

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / 'results_manifest.json').read_text())


def test_manifest_config_and_source_identity():
    m = manifest()
    for key, value in [('feature_cutoff_year', config.FEATURE_CUTOFF_YEAR),
                       ('min_reports', config.TASK_A_MIN_REPORTS),
                       ('n_folds', config.TASK_A_N_FOLDS), ('seed', config.TASK_A_SEED)]:
        assert m['config'][key] == value
    # Source changes require a new audit, not silently reusing old provenance.
    for path, info in m['source_fingerprints'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == info['sha256'], path


def test_leaderboards_match_verified_results():
    m = manifest()
    readme = (ROOT / 'README.md').read_text()
    mvp = (ROOT / 'MVP.md').read_text()
    names = {'ror': 'ROR', 'prr': 'PRR', 'ic': 'IC (BCPNN)', 'ic025': 'IC025',
             'ebgm': 'EBGM (MGPS)', 'eb05': 'EB05', 'drug_control': 'Drug-level-only control'}
    for key, name in names.items():
        v = m['results']['summary'][key]
        score = f"{v['auc_strat_mean']:.3f} ± {v['auc_strat_sd']:.3f}"
        diff = f"{v['diff_vs_ref_mean']:+.3f} ± {v['diff_vs_ref_sd']:.3f}".replace('-', '−')
        assert f"| {name} | {score} | {'reference' if key == 'ic' else diff} |" in mvp
        if key in {'ror', 'ic', 'eb05', 'drug_control'}:
            assert f'| {name} | {score} |' in readme
    for text in [readme, mvp]:
        assert f"{m['mapping']['names']:,}" in text
        assert f"{m['mapping']['mapped']:,}" in text
        assert f"{m['labels']['all_years_pairs']:,}" in text
        assert f"{m['config']['n_pairs']:,}" in text
    retained = sum(f['ingredients'] for f in m['fold_quality'])
    assert f'{retained:,} ingredients retained' in mvp


def test_manifest_fold_aggregation_and_population():
    m = manifest()
    folds = m['results']['folds']
    assert len(folds) == m['config']['n_folds']
    assert sum(f['ic']['n'] for f in folds.values()) == m['config']['n_pairs']
    for name, summary in m['results']['summary'].items():
        aucs = np.array([f[name]['auc_strat'] for f in folds.values()])
        diffs = aucs - np.array([f['ic']['auc_strat'] for f in folds.values()])
        np.testing.assert_allclose(
            [summary[k] for k in ['auc_strat_mean', 'auc_strat_sd', 'diff_vs_ref_mean', 'diff_vs_ref_sd']],
            [aucs.mean(), aucs.std(ddof=1), diffs.mean(), diffs.std(ddof=1)], atol=1e-12)
    v = m['verification']
    assert v['counts_reconstructed'] and v['all_metrics_replayed']
    assert v['eligible_pairs'] - v['combination_pairs_dropped'] == m['config']['n_pairs']
