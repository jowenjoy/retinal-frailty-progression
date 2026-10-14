"""Endpoint-class and score-balanced fold assignment."""
import numpy as np
import pandas as pd

def build_fold_assignment(df, n_parts: int = 5, seed: int = 42):
    """
    每个 class 内部按 Frailty_Score 均分为 n_parts 份：
      1. 组内按 (score, eid) 排序；
      2. 相同 score 的块内随机打乱（避免 eid 顺序带来的系统性偏差）；
      3. 轮转发牌（ordered[k::n_parts]）—— 每份拿到的分数分布与该 class 一致。

    返回 DataFrame: eid, class, fold (1-based)
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
