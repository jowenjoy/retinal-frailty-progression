"""Strict IO for rebuilt, baseline-defined first-observed-transition cohorts."""
import hashlib
import json
import os
import pickle
import numpy as np
import pandas as pd

SCHEMA = 'first_observed_transition_v1'
AUDIT_COLUMNS = ['pattern', 'schema_version', 'baseline_score', 'baseline_date',
                 'baseline_visit_instance', 'event_type', 'endpoint_date',
                 'endpoint_visit', 'last_negative_date', 'last_assessment_date']


def load_transition_data(csv_path, pattern=None):
    df = pd.read_csv(csv_path).rename(columns={'Eid': 'eid'})
    required = ['eid', 'Frailty_Score', 'months_between', 'event',
                'score_min_components'] + AUDIT_COLUMNS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Rebuild preparation files; missing transition fields: {sorted(missing)}')
    if df.empty or df['eid'].isna().any() or df['eid'].duplicated().any():
        raise ValueError('Empty cohort or missing/duplicate participant IDs')
    MIN_COMPONENTS = 3
    if (
        not df['schema_version'].eq(SCHEMA).all()
        or not df['score_min_components'].eq(MIN_COMPONENTS).all()
    ):
        raise ValueError(
            f'Incompatible endpoint schema / expected score_min_components={MIN_COMPONENTS}'
        )
    df['eid'] = df['eid'].astype(int)
    df['time'] = pd.to_numeric(df['months_between'], errors='raise')
    df['score'] = pd.to_numeric(df['Frailty_Score'], errors='raise')
    if not (np.isfinite(df['time']) & df['time'].gt(0)).all():
        raise ValueError('Follow-up must be finite and strictly positive')
    if not df['score'].isin(range(6)).all() or not df['event_type'].isin([0, 1, 2]).all():
        raise ValueError('Invalid endpoint score/status')
    if not df['event'].eq(df['event_type'].eq(1).astype(int)).all():
        raise ValueError('event must equal (event_type == 1)')
    early = df['pattern'].eq('prefrailty')
    late = df['pattern'].eq('frailty')
    if not (early | late).all():
        raise ValueError('Unknown prediction task')
    if not df.loc[early, 'baseline_score'].eq(0).all() or not df.loc[late, 'baseline_score'].isin([1, 2]).all():
        raise ValueError('Baseline risk-set violation')
    expected = np.where(early, np.where(df['score'].between(1, 2), 1,
                                      np.where(df['score'].ge(3), 2, 0)),
                        np.where(df['score'].ge(3), 1, 0))
    if not df['event_type'].eq(expected).all():
        raise ValueError('Endpoint score and event_type disagree')
    start = pd.to_datetime(df['baseline_date'], errors='raise')
    end = pd.to_datetime(df['endpoint_date'], errors='raise')
    if start.isna().any() or end.isna().any() or not np.allclose((end-start).dt.days/30.44, df['time']):
        raise ValueError('Endpoint dates and follow-up disagree')
    if pattern is not None:
        df = df[df['pattern'].eq(pattern)].copy()
        if df.empty:
            raise ValueError(f'No eligible participants for {pattern}')
    df['class'] = df['event_type'].astype(int)  # stratify by censor/target/competing
    return df.reset_index(drop=True)


def discover_transition_images(data_dir):
    path = os.path.join(data_dir, 'baseline_image_manifest.csv')
    manifest = pd.read_csv(path).fillna('')
    if manifest['Eid'].duplicated().any():
        raise ValueError('Duplicate baseline image IDs')
    cohort = load_transition_data(os.path.join(data_dir, 'demographicdata.csv'))
    visits = cohort.set_index('eid')['baseline_visit_instance'].to_dict()
    result = {}
    for row in manifest.to_dict('records'):
        eid = int(row['Eid'])
        if eid not in visits or int(row['baseline_visit_instance']) != int(visits[eid]):
            raise ValueError(f'Image/clinical baseline mismatch: {eid}')
        result[eid] = {}
        for side in ['fundus_left', 'fundus_right']:
            if row[side]:
                image_path = os.path.join(data_dir, row[side])
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(image_path)
                result[eid][side] = image_path
        if 'fundus_left' not in result[eid]:
            raise ValueError(f'Missing selected baseline left image: {eid}')
    if set(result) != set(visits):
        raise ValueError('Image manifest and cohort ID sets differ')
    return result


def cohort_signature(df):
    columns = ['eid', 'pattern', 'baseline_date', 'baseline_visit_instance',
               'baseline_score', 'score', 'time', 'event', 'event_type']
    text = df.sort_values('eid')[columns].to_csv(index=False)
    return hashlib.sha256(text.encode()).hexdigest()


def check_fold_signature(path, df, write=False):
    sidecar = path + '.transition.json'
    expected = {'schema': SCHEMA, 'cohort_sha256': cohort_signature(df)}
    if write:
        with open(sidecar, 'w', encoding='utf-8') as f:
            json.dump(expected, f, indent=2)
    else:
        with open(sidecar, encoding='utf-8') as f:
            actual = json.load(f)
        if actual != expected:
            raise ValueError('Fold assignment belongs to a different cohort/endpoint. Regenerate folds.')


def feature_signature(data_dir, side):
    # Bind cache to exact images (content, not just Eid or filename).
    paths = discover_transition_images(data_dir)
    digest = hashlib.sha256()
    for eid in sorted(paths):
        if side not in paths[eid]:
            continue
        digest.update(f'{eid}:{side}:'.encode())
        with open(paths[eid][side], 'rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
    return {'schema': SCHEMA, 'side': side, 'image_sha256': digest.hexdigest()}


def load_verified_features(cache_path, data_dir, side):
    with open(cache_path + '.transition.json', encoding='utf-8') as f:
        metadata = json.load(f)
    if metadata != feature_signature(data_dir, side):
        raise ValueError('Stale feature cache: regenerate from rebuilt baseline images.')
    with open(cache_path, 'rb') as f:
        return pickle.load(f)


def get_or_extract_features(cache_path, data_dir, weights, device, side='fundus_left'):
    if os.path.exists(cache_path):
        return load_verified_features(cache_path, data_dir, side)
    import torch
    from dataset import load_retfound_backbone, extract_all_features
    df = load_transition_data(os.path.join(data_dir, 'demographicdata.csv'))
    paths = discover_transition_images(data_dir)
    signature = feature_signature(data_dir, side)
    backbone = load_retfound_backbone(weights, device)
    backbone.eval()
    with torch.no_grad():
        dim = backbone(torch.zeros(1, 3, 224, 224, device=device)).shape[-1]
    features = extract_all_features(df, paths, backbone, device, feat_dim=dim,
                                    cache_path=cache_path,
                                    image_order=['oct_left' if side.endswith('left') else 'oct_right', side])
    with open(cache_path + '.transition.json', 'w', encoding='utf-8') as f:
        json.dump(signature, f, indent=2)
    return features
