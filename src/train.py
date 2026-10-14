"""V1 neural Cox training on rebuilt baseline-defined first-observed transitions. Prefrailty: baseline 0, target 1-2, direct frailty competing. Frailty: baseline 1-2, target >=3; recovery remains under observation."""

import os, sys, argparse, pickle
import numpy as np
import pandas as pd
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from cross_validation import (
    MODALITY_CONFIG, run_single_fold,
)
from folds import build_fold_assignment
from dataset import COVAR_COLS
from transition_data import (load_transition_data, discover_transition_images,
                             get_or_extract_features, check_fold_signature, AUDIT_COLUMNS)
from utils import set_seed, get_logger

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
CACHE_PATH = os.path.join(DATA_DIR, 'img_features_cache_fundus_left_byclass.pkl')
CFP_WEIGHTS = os.path.join(PROJECT_DIR, 'pretrained', 'RETFound_mae_natureCFP.pth')

# (score_lo, score_hi, event_threshold)
PATTERN_CONFIG = {
    'frailty':    dict(score_lo=1, score_hi=2, event_threshold=3),
    'prefrailty': dict(score_lo=0, score_hi=0, event_threshold=1),
}


def log_fold_class_table(logger, train_df, test_df, classes):
    """打印折内 train/test 各 class 人数与事件数。"""
    for part, sub in [('train', train_df), ('test', test_df)]:
        dist = sub.groupby('class').size().reindex(classes, fill_value=0)
        ev = int(sub['event'].sum())
        logger.info(
            f"  {part:5s} n={len(sub):5d} ev={ev:4d} "
            f"class分布={list(dist.values)}")


def write_fold_class_distribution(assign, output_dir, n_parts, classes):
    """写出每折各 class 人数表。"""
    dist_rows = []
    for fold_idx in range(n_parts):
        sub = assign[assign['fold'] == fold_idx + 1]
        row = {'fold': fold_idx + 1}
        for cls in classes:
            row[f'class_{cls}'] = int((sub['class'] == cls).sum())
        dist_rows.append(row)
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(output_dir, 'fold_class_distribution.csv'), index=False)


def parse_args():
    p = argparse.ArgumentParser(description='Per-class 5-fold CV for a frailty transition pattern')
    p.add_argument('--pattern', required=True, choices=list(PATTERN_CONFIG.keys()))
    p.add_argument('--modality', default='demo_fundus', choices=list(MODALITY_CONFIG.keys()))
    p.add_argument('--data_dir', default=DATA_DIR)
    p.add_argument('--cache_path', default=None, help='Defaults to a feature cache within --data_dir')
    p.add_argument('--cfp_weights', default=CFP_WEIGHTS)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--fold_assignment_csv', default=None,
                   help='CSV with (eid, fold); reuse this exact fold assignment')
    p.add_argument('--n_parts', type=int, default=5)
    p.add_argument('--extract_only', action='store_true', help='Build verified baseline-image cache only')

    # Model
    p.add_argument('--architecture', choices=['legacy', 'token_cross'], default='token_cross',
                   help='V1 default: covariate-token cross-attention')
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--num_cross_attn', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.3)

    # Training
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--weight_decay', type=float, default=5e-2)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--warmup_epochs', type=int, default=4)
    p.add_argument('--patience', type=int, default=20)

    # Loss
    p.add_argument('--cox_weight', type=float, default=0.10)
    p.add_argument('--rank_weight', type=float, default=0.70)
    p.add_argument('--focal_gamma', type=float, default=1.5)

    # Other
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--max_pos_weight', type=float, default=10.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.cache_path is None:
        args.cache_path = os.path.join(args.data_dir, 'img_features_cache_fundus_left_byclass.pkl')
    os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)
    if args.epochs < 1 or args.n_parts < 2:
        raise ValueError('epochs >= 1 and n_parts >= 2 required')
    if not args.extract_only and os.path.isdir(args.output_dir):
        import glob
        if glob.glob(os.path.join(args.output_dir, 'fold_*', 'best_model.pth')):
            raise RuntimeError('Archive previous model outputs before rerunning at the same output path.')
    set_seed(args.seed)

    cfg = MODALITY_CONFIG[args.modality]
    args.use_covar = cfg['use_covar']
    args.active_slots = cfg['active_slots']

    pat = PATTERN_CONFIG[args.pattern]
    lo, hi, ev_th = pat['score_lo'], pat['score_hi'], pat['event_threshold']

    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'cv_train.log'),
                        name=f'CVPattern_{args.pattern}_{args.modality}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info(f"Transition-pattern {args.n_parts}-Fold CV | "
                f"pattern={args.pattern} modality={args.modality}")
    logger.info(f"baseline score range=[{lo}, {hi}]; endpoint=first observed target; prefrailty direct frailty=competing")
    logger.info(f"active_slots={args.active_slots} use_covar={args.use_covar}")
    logger.info(f"device={device} seed={args.seed}")
    logger.info("=" * 60)

    # ---- 1. 人口学：加载 → 按模式过滤 → 重定义结局 ----
    df = load_transition_data(os.path.join(args.data_dir, 'demographicdata.csv'), args.pattern)
    logger.info(f"[Data] After pattern filter (score {lo}-{hi}): "
                f"n={len(df)}, events={int(df['event'].sum())} "
                f"({df['event'].mean()*100:.2f}%)")

    img_paths = discover_transition_images(args.data_dir)
    valid_eids = set(img_paths.keys()) & set(df['eid'].values)
    df = df[df['eid'].isin(valid_eids)].reset_index(drop=True)
    logger.info(f"[Data] Matched with fundus images: {len(df)} subjects")

    # ---- 2. 特征缓存（复用，无需重新提取）----
    img_features = get_or_extract_features(args.cache_path, args.data_dir, args.cfp_weights, device)
    eligible_features = {eid for eid, feature in img_features.items()
                         if np.isfinite(feature).all() and np.any(feature[1])}
    df = df[df['eid'].isin(eligible_features)].reset_index(drop=True)
    if args.extract_only:
        logger.info('Verified cache ready; no model training requested.')
        return
    if df.empty or df['event'].value_counts().reindex([0, 1], fill_value=0).min() < args.n_parts:
        raise ValueError('Too few events/non-events for requested folds after image quality filtering.')
    feat_dim = next(iter(img_features.values())).shape[-1]
    logger.info(f"[Feat] Reusing cached features: {args.cache_path}")
    logger.info(f"[Feat] cache subjects={len(img_features)}  feat_dim={feat_dim}")

    # ---- 3. 折划分（与 cross_validation_by_class 完全一致的方法）----
    if args.fold_assignment_csv:
        check_fold_signature(args.fold_assignment_csv, df)
        assign = pd.read_csv(args.fold_assignment_csv)[['eid', 'fold']].drop_duplicates()
        if assign['eid'].duplicated().any() or set(assign['eid']) != set(df['eid']):
            raise ValueError('Fold assignment must contain exactly one row per cohort participant.')
        if not assign['fold'].isin(range(1, args.n_parts + 1)).all():
            raise ValueError('Invalid fold number')
        fold_of = {int(e): int(f) for e, f in assign[['eid', 'fold']].values}
        missing = set(df['eid'].values) - set(fold_of)
        assert not missing, f'{len(missing)} subjects lack a fold'
        df['fold'] = df['eid'].map(fold_of)
        assign = df[['eid', 'class', 'fold']].copy()
        logger.info(f"[Fold] Reusing assignment from {args.fold_assignment_csv}")
    else:
        assign = build_fold_assignment(df, n_parts=args.n_parts, seed=args.seed)
        assign.to_csv(os.path.join(args.output_dir, 'fold_assignment.csv'), index=False)
        check_fold_signature(os.path.join(args.output_dir, 'fold_assignment.csv'), df, write=True)
        logger.info(f"[Fold] Fresh score-balanced per-class split → fold_assignment.csv")

    classes = sorted(df['class'].unique())
    for cls in classes:
        grp = assign[assign['class'] == cls]
        logger.info(f"[Fold] class_{cls}: n={len(grp)}, "
                    f"per-fold={grp.groupby('fold').size().tolist()}")

    covar_cols = [c for c in COVAR_COLS if c in df.columns]

    # ---- 4. 逐折训练 ----
    fold_results, all_val_predictions = [], []
    for fold_idx in range(args.n_parts):
        logger.info(f"\n{'=' * 40}\n  Fold {fold_idx + 1}/{args.n_parts}\n{'=' * 40}")

        val_eids = set(assign.loc[assign['fold'] == fold_idx + 1, 'eid'].values)
        train_data_raw = df[~df['eid'].isin(val_eids)].reset_index(drop=True)
        test_data_raw  = df[df['eid'].isin(val_eids)].reset_index(drop=True)
        log_fold_class_table(logger, train_data_raw, test_data_raw, classes)
        if train_data_raw['event'].nunique() < 2 or test_data_raw['event'].nunique() < 2:
            raise ValueError('Every training/validation fold must contain target events and non-events.')

        result = run_single_fold(
            args, fold_idx, train_data_raw, test_data_raw,
            img_features, covar_cols, feat_dim, device, logger)

        fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx + 1}')
        pred_path = os.path.join(fold_dir, 'val_predictions.csv')
        if os.path.isfile(pred_path):
            fold_preds = pd.read_csv(pred_path)
            pred_eids = fold_preds['eid'].astype(int).values
            true_eids = test_data_raw['eid'].astype(int).values
            if len(pred_eids) != len(true_eids) or not (pred_eids == true_eids).all():
                raise RuntimeError(
                    f"Fold {fold_idx + 1}: val_predictions eid order mismatch")
            fold_preds['class'] = test_data_raw['class'].astype(int).values
            fold_preds['fold']  = fold_idx + 1
            for column in AUDIT_COLUMNS:
                fold_preds[column] = test_data_raw[column].values
            fold_preds.to_csv(pred_path, index=False)
            all_val_predictions.append(fold_preds)

        fold_results.append(result)
        logger.info(f"  Fold {fold_idx + 1} done: C-index = {result['val_cindex']:.4f}")

    # ---- 5. 汇总 ----
    cindices = [r['val_cindex'] for r in fold_results]
    for r in fold_results:
        logger.info(
            f"  Fold {r['fold']}: C-idx={r['val_cindex']:.4f}  "
            f"trainC={r['train_cindex']:.4f}  "
            f"train={r['train_n']}({r['train_events']}ev)  "
            f"test={r['val_n']}({r['val_events']}ev)  best_epoch={r['best_epoch']}")

    mean_c, std_c = np.mean(cindices), np.std(cindices)
    logger.info(f"  Mean C-index: {mean_c:.4f} ± {std_c:.4f}")
    logger.info(f"  Per-fold: {[f'{c:.4f}' for c in cindices]}")

    pd.DataFrame(fold_results).to_csv(
        os.path.join(args.output_dir, 'cv_summary.csv'), index=False)
    assign.to_csv(os.path.join(args.output_dir, 'fold_assignment.csv'), index=False)
    check_fold_signature(os.path.join(args.output_dir, 'fold_assignment.csv'), df, write=True)
    write_fold_class_distribution(assign, args.output_dir, args.n_parts, classes)

    if all_val_predictions:
        all_preds = pd.concat(all_val_predictions, ignore_index=True)
        all_preds.to_csv(os.path.join(args.output_dir, 'oof_predictions.csv'), index=False)
        logger.info(f"  Saved {len(all_preds)} OOF predictions")
        try:
            from lifelines.utils import concordance_index
            overall_c = concordance_index(
                all_preds['time'].values, -all_preds['risk'].values,
                all_preds['event'].values.astype(int))
            logger.info(f"  Overall pooled C-index: {overall_c:.4f}")
        except Exception:
            pass

    with open(os.path.join(args.output_dir, 'cv_final_summary.txt'), 'w') as f:
        f.write(f"Transition-pattern {args.n_parts}-Fold CV Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Pattern: {args.pattern}; baseline score {lo}-{hi}; first observed target; direct frailty competes in prefrailty\n")
        f.write(f"Modality: {args.modality}\n")
        f.write(f"Per-fold Test C-indices: {cindices}\n")
        f.write(f"Mean Test C-index: {mean_c:.4f} +/- {std_c:.4f}\n")

    logger.info(f"\nDone -> {args.output_dir}")
    return fold_results


if __name__ == '__main__':
    main()

