"""Endpoint-class and score-balanced fold assignment."""
import numpy as np
import pandas as pd

def build_fold_assignment(df, n_parts: int = 5, seed: int = 42):
    """
    Split each endpoint class into n_parts score-balanced folds:
      1. Sort each class by (score, eid).
      2. Shuffle within equal-score blocks to avoid systematic participant-order bias.
      3. Distribute records round-robin (ordered[k::n_parts]) to balance scores across folds.

    Return a DataFrame with eid, class, and fold (1-based).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for cls, grp in df.groupby('class', sort=True):
        grp = grp.sort_values(['score', 'eid'])
        ordered = []
        for _, block in grp.groupby('score', sort=True):
            vals = block['eid'].values.copy()
            rng.shuffle(vals)
            ordered.append(vals)
        ordered = np.concatenate(ordered)
        for k in range(n_parts):
            for eid in ordered[k::n_parts]:
                rows.append({'eid': int(eid), 'class': int(cls), 'fold': k + 1})
    return pd.DataFrame(rows)
