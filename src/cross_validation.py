"""Shared fold-local preprocessing and neural survival training loop."""

import os, sys, math, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import MultiModalSurvivalModel, TokenCrossAttentionSurvivalModel
from losses import total_loss
from training_validation import evaluate_model
from dataset import SurvivalDataset, COVAR_COLS, CAT_COLS, NUM_COLS


MODALITY_CONFIG = {
    'fundus_only': {'active_slots': [1], 'use_covar': False},
    'demo_only':   {'active_slots': [],  'use_covar': True},
    'demo_fundus': {'active_slots': [1], 'use_covar': True},
}


class EarlyStopping:
    def __init__(self, patience=15, path='best_model.pth'):
        self.patience = patience
        self.path = path
        self.best = -float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_cindex, model):
        if val_cindex > self.best:
            self.best = val_cindex
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


# ── Basic data loading without scaling or encoding to prevent leakage across folds ──────────────────────────

def load_demographics_raw(csv_path: str):
    """
    Load demographic data with basic validation only:
      - Validate prepared first-observed event, baseline, and follow-up fields.
      - Preserve event labels and do not refilter by final score.
      - Fit imputation, scaling, and encoding on each training fold only.

    Return a DataFrame with numerical columns on their original scales and categorical columns as original category strings.
    """
    from transition_data import load_transition_data
    return load_transition_data(csv_path)


def fit_transform_covariates(train_df, val_df, covar_cols):
    """
    Fit LabelEncoder and StandardScaler on training data, then transform training and validation data.

    Fit all preprocessing operations on train_df only to prevent validation-data leakage.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        covar_cols: Covariate column names to process.

    Returns:
        train_df, val_df: Copies of the processed DataFrames.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    cat_cols = [c for c in CAT_COLS if c in covar_cols]
    num_cols = [c for c in NUM_COLS if c in covar_cols]

    # 1. Impute missing values using training-set medians and modes.
    for col in num_cols:
        med = train_df[col].median()
        if pd.isna(med):
            raise ValueError(f'Training fold has no observed values for {col}')
        train_df[col] = train_df[col].fillna(med)
        val_df[col]   = val_df[col].fillna(med)

    for col in cat_cols:
        mode_val = train_df[col].mode()
        fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
        train_df[col] = train_df[col].fillna(fill_val)
        val_df[col]   = val_df[col].fillna(fill_val)

    # 2. Fit LabelEncoder on training data only.
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # Handle categories present only in validation data.
        val_classes = set(val_df[col].astype(str).unique())
        known_classes = set(le.classes_)
        unknown = val_classes - known_classes
        if unknown:
            # Map unseen categories to the most frequent known training category.
            most_freq = train_df[col].value_counts().index[0]
            val_df[col] = val_df[col].astype(str).apply(
                lambda x: x if x in known_classes else le.classes_[most_freq])
        val_df[col] = le.transform(val_df[col].astype(str))

    # 3. Fit StandardScaler on training data only.
    scaler = StandardScaler()
    if num_cols:
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        val_df[num_cols]   = scaler.transform(val_df[num_cols])

    return train_df, val_df


# ── Single-fold training ────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, use_covar, args):
    model.train()
    total_loss_sum, cox_sum, rank_sum, n = 0, 0, 0, 0

    for batch in loader:
        img_feats, covar, times, events, _ = [x.to(device) for x in batch]
        if not use_covar:
            covar = torch.zeros_like(covar)
        optimizer.zero_grad()

        risk = model(img_feats, covar)
        loss, lc, lr = total_loss(
            risk, times, events,
            cox_weight=args.cox_weight, rank_weight=args.rank_weight,
            focal_gamma=args.focal_gamma)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss_sum += loss.item()
        cox_sum += lc
        rank_sum += lr
        n += 1

    n = max(n, 1)
    return total_loss_sum / n, cox_sum / n, rank_sum / n


def run_single_fold(args, fold_idx, train_data_raw, val_data_raw,
                    img_features, covar_cols, feat_dim, device, logger):
    """
    Train a single fold:
      1. Fit preprocessing on training data and transform training and validation data.
      2. Build SurvivalDataset instances.
      3. Train the model.
      4. Return fold metrics and save predictions.

    Outputs in the fold directory:
      val_predictions.csv: eid, score, time, event, risk in dataset row order.
      train_predictions.csv: Same columns, predicted using the best checkpoint and noise-free training data (split='val').
    """
    fold_dir = os.path.join(args.output_dir, f'fold_{fold_idx + 1}')
    os.makedirs(fold_dir, exist_ok=True)

    # ---- Fit preprocessing on training data only to prevent leakage ----
    train_data, val_data = fit_transform_covariates(
        train_data_raw, val_data_raw, covar_cols)

    train_ds = SurvivalDataset(
        train_data, img_features, covar_cols,
        split='train',
        active_slots=args.active_slots,
        covar_noise_std=0.0 if getattr(args, 'architecture', 'legacy') == 'token_cross' else 0.01)

    val_ds = SurvivalDataset(
        val_data, img_features, covar_cols,
        split='val',
        active_slots=args.active_slots)

    # Weighted sampler
    n_ev = int(train_data['event'].sum())
    pos_w = min((len(train_data) - n_ev) / max(n_ev, 1), args.max_pos_weight)
    sampler = WeightedRandomSampler(
        torch.DoubleTensor(
            train_data['event'].apply(lambda e: pos_w if e == 1 else 1.0).values),
        len(train_data), replacement=True)

    logger.info(f"Fold {fold_idx + 1}: Train={len(train_ds)} (events={n_ev}) "
                f"Val={len(val_ds)} (events={int(val_data['event'].sum())}) "
                f"pos_w={pos_w:.1f}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True)

    # Model
    covar_dim = len(covar_cols)
    token_cross = getattr(args, 'architecture', 'legacy') == 'token_cross'
    model_class = TokenCrossAttentionSurvivalModel if token_cross else MultiModalSurvivalModel
    extra_kwargs = {}
    if token_cross:
        extra_kwargs = dict(use_covar=args.use_covar, categorical_cardinalities={
            i: int(train_data[c].max()) + 1
            for i, c in enumerate(covar_cols) if c in CAT_COLS})
    model = model_class(
        img_feat_dim=feat_dim, covar_dim=covar_dim,
        d_model=args.d_model, nhead=args.nhead,
        num_cross_attn=args.num_cross_attn,
        dropout=args.dropout,
        active_slots=args.active_slots, **extra_kwargs).to(device)
    # Save construction metadata separately; legacy state_dict format is retained.
    import json
    with open(os.path.join(fold_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(architecture=getattr(args, 'architecture', 'legacy'),
                       img_feat_dim=feat_dim, covar_dim=covar_dim,
                       d_model=args.d_model, nhead=args.nhead,
                       num_cross_attn=args.num_cross_attn, dropout=args.dropout,
                       active_slots=args.active_slots, **extra_kwargs), f, indent=2)
    logger.info(f"  Fold {fold_idx + 1}: model params = "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_schedule(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    ckpt_path = os.path.join(fold_dir, 'best_model.pth')
    early_stop = EarlyStopping(patience=args.patience, path=ckpt_path)
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, lc, lr = train_one_epoch(
            model, train_loader, optimizer, device, args.use_covar, args)
        scheduler.step()

        vm = evaluate_model(model, val_loader, device, use_covar=args.use_covar)

        saved = early_stop(vm['cindex'], model)
        if saved:
            best_epoch = epoch
        mark = ' *' if saved else ''
        if epoch % 10 == 0 or saved:
            logger.info(
                f"  Fold {fold_idx + 1} [{epoch:03d}/{args.epochs}] "
                f"loss={tr_loss:.4f} | Val C-idx={vm['cindex']:.4f}{mark}")

        if early_stop.early_stop:
            logger.info(f"  Fold {fold_idx + 1}: Early stop @ epoch {epoch} "
                        f"best={early_stop.best:.4f}")
            break

    # Load best and evaluate
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    best_vm = evaluate_model(model, val_loader, device, use_covar=args.use_covar)

    def _collect_preds(loader):
        """Collect risk, time, event, and score in loader order.

        The loader does not shuffle, so predictions follow dataset row order.
        Each prediction can therefore be paired with its dataset participant ID.
        """
        model.eval()
        risks, times, events, scores = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                img_feats, covar, b_times, b_events, b_scores = [
                    x.to(device) for x in batch]
                if not args.use_covar:
                    covar = torch.zeros_like(covar)
                risk = model(img_feats, covar)
                risks.extend(risk.squeeze(-1).cpu().numpy().tolist())
                times.extend(b_times.cpu().numpy().tolist())
                events.extend(b_events.cpu().numpy().tolist())
                scores.extend(b_scores.cpu().numpy().tolist())
        return risks, times, events, scores

    # Validation-fold predictions with participant IDs and scores
    all_risks, all_times, all_events, all_scores = _collect_preds(val_loader)
    fold_preds = pd.DataFrame({
        'eid':   val_ds.data['eid'].astype(int).tolist(),
        'score': all_scores,
        'time':  all_times,
        'event': all_events,
        'risk':  all_risks,
    })
    from transition_data import AUDIT_COLUMNS
    for column in AUDIT_COLUMNS:
        fold_preds[column] = val_ds.data[column].values
    fold_preds.to_csv(os.path.join(fold_dir, 'val_predictions.csv'), index=False)

    # Training-fold predictions using the best checkpoint and split='val' (no feature noise or modality dropout), for calibration reuse
    train_eval_ds = SurvivalDataset(
        train_data, img_features, covar_cols,
        split='val', active_slots=args.active_slots)
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True)
    train_vm = evaluate_model(model, train_eval_loader, device,
                              use_covar=args.use_covar)
    logger.info(f"  Fold {fold_idx + 1}: Train C-idx (best ckpt) = "
                f"{train_vm['cindex']:.4f}")

    tr_risks, tr_times, tr_events, tr_scores = _collect_preds(train_eval_loader)
    train_preds = pd.DataFrame({
        'eid':   train_eval_ds.data['eid'].astype(int).tolist(),
        'score': tr_scores,
        'time':  tr_times,
        'event': tr_events,
        'risk':  tr_risks,
    })
    for column in AUDIT_COLUMNS:
        train_preds[column] = train_eval_ds.data[column].values
    train_preds.to_csv(os.path.join(fold_dir, 'train_predictions.csv'), index=False)

    return {
        'fold': fold_idx + 1,
        'train_n': len(train_ds),
        'val_n': len(val_ds),
        'train_events': n_ev,
        'val_events': int(val_data['event'].sum()),
        'best_epoch': best_epoch,
        'val_cindex': best_vm['cindex'],
        'train_cindex': train_vm['cindex'],
    }
