"""
dataset.py — Fundus survival dataset pipeline.
"""
import os, re, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

COVAR_COLS = [
    'Gender', 'Ethnicity', 'Age_Raw_At_Image',
    'BMI_At_Image', 'Smoking_At_Image', 'Alcohol_At_Image',
]
CAT_COLS = ['Gender', 'Ethnicity', 'Smoking_At_Image', 'Alcohol_At_Image']
NUM_COLS = ['Age_Raw_At_Image', 'BMI_At_Image']

IMAGE_MEAN   = [0.485, 0.456, 0.406]
IMAGE_STD    = [0.229, 0.224, 0.225]
# slot 0 = placeholder, slot 1 = Fundus
IMAGE_ORDER_LEFT  = ['oct_left', 'fundus_left']
IMAGE_ORDER_RIGHT = ['oct_right', 'fundus_right']
IMAGE_ORDER       = IMAGE_ORDER_LEFT
FUNDUS_SLOT = 1   # fundus is at slot index 1
N_SLOTS     = 2


def load_retfound_backbone(ckpt_path: str, device: torch.device,
                           img_size: int = 224) -> nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Weights not found: {ckpt_path}")
    import timm
    raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    def unwrap(ckpt, depth=0):
        if depth > 5 or not isinstance(ckpt, dict): return ckpt
        for k in ('model', 'state_dict', 'teacher', 'student',
                  'encoder', 'backbone', 'net'):
            if k in ckpt:
                res = unwrap(ckpt[k], depth + 1)
                if isinstance(res, dict) and any(
                        isinstance(v, torch.Tensor) for v in res.values()):
                    print(f"  [timm] Unwrapped via '{k}'")
                    return res
        return ckpt

    state_dict = unwrap(raw_ckpt)
    proj_key   = next((k for k in state_dict
                       if 'patch_embed.proj.weight' in k), None)
    embed_dim, patch_size = (
        (state_dict[proj_key].shape[0], state_dict[proj_key].shape[-1])
        if proj_key else (768, 16))
    orig_pe_key = next((k for k in state_dict if 'pos_embed' in k), None)
    ckpt_img = (int(round(((state_dict[orig_pe_key].shape[1]-1)**0.5)*patch_size))
                if orig_pe_key else img_size)
    size_str = 'large' if embed_dim >= 1024 else 'base'
    print(f"  [timm] {os.path.basename(ckpt_path)}: "
          f"embed_dim={embed_dim}  patch={patch_size}  ckpt_img={ckpt_img}")

    candidates = [
        f'vit_{size_str}_patch{patch_size}_{ckpt_img}',
        f'vit_{size_str}_patch{patch_size}_224',
        f'vit_{size_str}_patch{patch_size}_reg4_224',
    ]
    probe_name = candidates[-1]
    for mn in candidates:
        try: timm.create_model(mn, pretrained=False, num_classes=0); probe_name=mn; break
        except Exception: continue

    ref_keys   = set(timm.create_model(probe_name, pretrained=False,
                                        num_classes=0).state_dict().keys())
    best_sd, best_overlap = state_dict, len(set(state_dict.keys()) & ref_keys)
    for prefix in ('encoder.','backbone.','module.','base_model.','model.','net.','trunk.'):
        if any(k.startswith(prefix) for k in state_dict):
            cand    = {k[len(prefix):] if k.startswith(prefix) else k: v
                       for k, v in state_dict.items()}
            overlap = len(set(cand.keys()) & ref_keys)
            if overlap > best_overlap:
                best_sd, best_overlap = cand, overlap
                print(f"  [timm] Stripped prefix '{prefix}'")
    cleaned = best_sd

    pe_key = next((k for k in cleaned if 'pos_embed' in k), None)
    if pe_key is not None:
        pe   = cleaned[pe_key].float()
        cls  = pe[:, :1, :]; patch = pe[:, 1:, :]
        h_c  = int(patch.shape[1]**0.5); h_i = img_size // patch_size
        if h_c != h_i:
            patch = patch.reshape(1, h_c, h_c, -1).permute(0,3,1,2)
            patch = torch.nn.functional.interpolate(
                patch, size=(h_i, h_i), mode='bicubic', align_corners=False)
            cleaned[pe_key] = torch.cat(
                [cls, patch.permute(0,2,3,1).reshape(1, h_i*h_i, -1)], dim=1)
            print(f"  [timm] pos_embed: {h_c}×{h_c} → {h_i}×{h_i}")

    final_model = None
    for mn in candidates:
        try:
            final_model = timm.create_model(mn, pretrained=False,
                                             num_classes=0, img_size=img_size)
            print(f"  [timm] Arch: '{mn}'"); break
        except Exception: continue
    if not final_model: raise RuntimeError("Cannot create timm ViT.")

    msg       = final_model.load_state_dict(cleaned, strict=False)
    n_matched = len(set(final_model.state_dict().keys()) - set(msg.missing_keys))
    print(f"  [timm] matched={n_matched}/{len(final_model.state_dict())}  "
          f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    return final_model.to(device)


# ── Demographic preprocessing ──────────────────────────────────────────────────────────────



# ── Image path discovery ──────────────────────────────────────────────────────────────



# ── Feature extraction ───────────────────────────────────────────────────

def extract_all_features(df, img_paths,
                         backbone: nn.Module,
                         device: torch.device,
                         batch_size: int = 32,
                         feat_dim: int = 1024,
                         cache_path: str = './dataset/img_features_cache_fundus.pkl',
                         enable_quality_filter: bool = True,
                         image_order: list = None,
                         horizontal_flip: bool = False):
    """
    Extract fundus features.  Returns {eid: np.ndarray (N_SLOTS, feat_dim)}
    where slot 0 is zeros and slot 1 is fundus features.
    """
    if image_order is None:
        image_order = IMAGE_ORDER
    if os.path.isfile(cache_path):
        print(f"  [Cache] Loading from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"  [Extract] {len(df)} subjects × {len(image_order)} slots (fundus only)"
          f"{' (horizontal_flip)' if horizontal_flip else ''}")
    from torchvision import transforms as T

    _transform_list = [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
    ]
    if horizontal_flip:
        _transform_list.append(T.RandomHorizontalFlip(p=1.0))
    _transform_list += [T.ToTensor(), T.Normalize(IMAGE_MEAN, IMAGE_STD)]
    _transform = T.Compose(_transform_list)

    def _crop_fundus_fov(img, threshold=30, padding=0.01):
        arr = np.array(img.convert('L')); H, W = arr.shape
        col_mask = (arr > threshold).sum(axis=0) > H * 0.10
        row_mask = (arr > threshold).sum(axis=1) > W * 0.10
        if not col_mask.any() or not row_mask.any(): return img
        x0,x1 = np.where(col_mask)[0][[0,-1]]; y0,y1 = np.where(row_mask)[0][[0,-1]]
        px=int((x1-x0)*padding); py=int((y1-y0)*padding)
        x0=max(0,x0-px); x1=min(W,x1+px); y0=max(0,y0-py); y1=min(H,y1+py)
        if (x1-x0)<W*0.3 or (y1-y0)<H*0.3: return img
        return img.crop((x0,y0,x1,y1))

    def _is_bad_fundus(img, max_gr=0.90, min_fov=0.10):
        rgb  = np.array(img.convert('RGB'), dtype=np.float32)
        gray = np.array(img.convert('L'),   dtype=np.float32)
        fov  = gray > 20
        if fov.sum()/gray.size < min_fov: return True
        r = rgb[:,:,0][fov].mean(); g = rgb[:,:,1][fov].mean()
        if r < 1.0: return True
        return (g/r) > max_gr

    # Build records: map IMAGE_ORDER slots to fundus images
    # slot 0 = 'oct_left'/'oct_right' → always None (no OCT)
    # slot 1 = 'fundus_left'/'fundus_right' → actual fundus path
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  [Extract] Building records", unit="subj"):
        eid   = int(row['eid'])
        paths = img_paths.get(eid, {})
        for key in image_order:
            if 'oct' in key:
                # OCT slot: always empty
                records.append((eid, key, None))
            else:
                raw = paths.get(key)
                if raw is None or not os.path.isfile(raw):
                    records.append((eid, key, None))
                else:
                    records.append((eid, key, raw))

    n_fundus = sum(1 for r in records if r[2] is not None and 'fundus' in r[1])
    n_empty  = sum(1 for r in records if r[2] is None)
    print(f"  [Extract] Fundus={n_fundus}  empty={n_empty}")

    class _DS(torch.utils.data.Dataset):
        def __init__(self, recs): self.recs = recs
        def __len__(self): return len(self.recs)
        def __getitem__(self, i):
            eid, key, raw = self.recs[i]
            try:
                if raw is None: return torch.zeros(3,224,224), i, False
                img = Image.open(raw).convert('RGB')
                if enable_quality_filter and _is_bad_fundus(img):
                    return torch.zeros(3,224,224), i, False
                img = _crop_fundus_fov(img)
                return _transform(img), i, True
            except: return torch.zeros(3,224,224), i, False

    def _col(batch):
        return (torch.stack([b[0] for b in batch]),
                [b[1] for b in batch],
                [b[2] for b in batch])

    all_feats = np.zeros((len(records), feat_dim), dtype=np.float32)
    loader = DataLoader(_DS(records), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=True,
                        collate_fn=_col)
    backbone.eval()
    with torch.no_grad():
        for imgs, local_idxs, valids in tqdm(
                loader, desc="  [Extract] Fundus", unit='batch'):
            feats = backbone(imgs.to(device)).cpu().numpy()
            fd = min(feats.shape[-1], feat_dim)
            for j, (li, valid) in enumerate(zip(local_idxs, valids)):
                if valid:
                    all_feats[li, :fd] = feats[j, :fd]

    # Aggregate: per (eid, slot)
    from collections import defaultdict
    slot_accum = defaultdict(list)
    for i, (eid, key, raw) in enumerate(records):
        if raw is not None and all_feats[i].any():
            slot_accum[(eid, key)].append(all_feats[i])

    features = {}
    n_slots = len(image_order)
    for (eid, key), vecs in slot_accum.items():
        if eid not in features:
            features[eid] = np.zeros((n_slots, feat_dim), dtype=np.float32)
        features[eid][image_order.index(key)] = (
            np.median(np.stack(vecs), axis=0).astype(np.float32))

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'wb') as f: pickle.dump(features, f)
    print(f"  [Cache] Saved → {cache_path}  ({len(features)} subjects)")
    return features


# ── SurvivalDataset ───────────────────────────────────────────────────────────

class SurvivalDataset(Dataset):
    def __init__(self, data, img_features, covar_cols,
                 split='train', feat_noise_std=0.02,
                 modality_dropout_p=0.15, covar_noise_std=0.01,
                 active_slots=None):
        self.data               = data.reset_index(drop=True)
        self.img_features       = img_features
        self.covar_cols         = covar_cols
        self.split              = split
        self.feat_noise_std     = feat_noise_std
        self.modality_dropout_p = modality_dropout_p
        self.covar_noise_std    = covar_noise_std
        self.active_slots       = active_slots
        feat_shape = next(iter(img_features.values())).shape
        self.default_feat = np.zeros(feat_shape, dtype=np.float32)

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row      = self.data.iloc[idx]
        eid      = int(row['eid'])
        img_feat = torch.from_numpy(
            self.img_features.get(eid, self.default_feat).astype(np.float32))

        if self.active_slots is not None:
            mask = torch.zeros(img_feat.shape[0], dtype=torch.bool)
            for s in self.active_slots: mask[s] = True
            img_feat[~mask] = 0.0

        mean = img_feat.mean(dim=-1, keepdim=True)
        std  = img_feat.std(dim=-1,  keepdim=True)
        img_feat = (img_feat - mean) / (std + 1e-6)

        covar = torch.from_numpy(
            row[self.covar_cols].values.astype(np.float32))

        if self.split == 'train':
            if self.feat_noise_std > 0:
                img_feat = img_feat + torch.randn_like(img_feat)*self.feat_noise_std
            if (self.modality_dropout_p > 0
                    and torch.rand(1).item() < self.modality_dropout_p):
                img_feat[torch.randint(0, img_feat.shape[0], (1,)).item()] = 0.0
            if self.covar_noise_std > 0:
                covar = covar + torch.randn_like(covar)*self.covar_noise_std

        time  = torch.tensor(float(row['time']),  dtype=torch.float32)
        event = torch.tensor(float(row['event']), dtype=torch.float32)
        score = torch.tensor(int(row['score']),   dtype=torch.long)
        return img_feat, covar, time, event, score


# ── build_datasets ────────────────────────────────────────────────────────────
