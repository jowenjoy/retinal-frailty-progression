"""
model.py — Fundus survival model.

Backward-compatible key names: fundus_proj, fundus_pool, cov_proj,
cross_attn_layers, cox_head.
"""
import torch
import torch.nn as nn


class FeatureTokenCrossAttention(nn.Module):
    """Image queries multiple covariate tokens; no sigmoid gates.

    The cache contains one global image vector, not spatial patches. Keeping
    attention unidirectional avoids a reverse branch with a singleton key.
    """
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))

    def forward(self, image, context):
        context = self.context_norm(context)
        update, _ = self.attention(self.query_norm(image), context, context,
                                   need_weights=False)
        image = image + self.drop(update)
        return image + self.drop(self.ffn(self.ffn_norm(image)))


class TokenCrossAttentionSurvivalModel(nn.Module):
    """Opt-in model for existing (B, 2, F) global-feature caches.

    Each covariate has its own token. Categorical cardinalities are fitted on
    the training fold and supplied as {column_index: number_of_categories}.
    Forward/embedding API matches the legacy model, but weights do not.
    """
    def __init__(self, img_feat_dim=1024, covar_dim=6, d_model=128,
                 nhead=8, num_cross_attn=1, dropout=0.3, active_slots=None,
                 use_covar=True, categorical_cardinalities=None):
        super().__init__()
        if d_model <= 0 or nhead <= 0 or d_model % nhead:
            raise ValueError('d_model must be positive and divisible by nhead')
        if num_cross_attn < 0:
            raise ValueError('num_cross_attn must be nonnegative')
        if active_slots is not None and list(active_slots) not in ([], [1]):
            raise ValueError('Supported active_slots are [] and [1]')
        self.demo_only = active_slots is not None and len(active_slots) == 0
        self.use_covar = use_covar
        self.covar_dim = covar_dim
        if self.demo_only:
            if not use_covar:
                raise ValueError('At least one modality must be enabled')
            self.cox_head = CoxHead(covar_dim)
            return
        self.fundus_proj = nn.Sequential(
            nn.Linear(img_feat_dim, d_model), nn.LayerNorm(d_model))
        self.image_head = nn.Linear(d_model, 1, bias=False)
        if use_covar:
            if covar_dim < 2:
                raise ValueError('Cross-attention requires at least two covariate tokens')
            cards = categorical_cardinalities or {}
            self.category_embeddings = nn.ModuleDict({
                str(i): nn.Embedding(n, d_model) for i, n in cards.items()})
            self.feature_weight = nn.Parameter(torch.empty(covar_dim, d_model))
            self.feature_bias = nn.Parameter(torch.empty(covar_dim, d_model))
            nn.init.normal_(self.feature_weight, std=0.02)
            nn.init.normal_(self.feature_bias, std=0.02)
            self.cross_attn_layers = nn.ModuleList([
                FeatureTokenCrossAttention(d_model, nhead, dropout)
                for _ in range(num_cross_attn)])
            self.covariate_head = nn.Linear(covar_dim * d_model, 1, bias=False)
            self.interaction_head = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2 or 1),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2 or 1, 1))

    def covariate_tokens(self, covar):
        tokens = []
        for i in range(self.covar_dim):
            if str(i) in self.category_embeddings:
                token = self.category_embeddings[str(i)](covar[:, i].long())
            else:
                token = covar[:, i:i + 1] * self.feature_weight[i]
            tokens.append(token + self.feature_bias[i])
        return torch.stack(tokens, dim=1)

    def forward(self, img_feats, covar, return_embeddings=False):
        if self.demo_only:
            risk = self.cox_head(covar)
            return (risk, covar) if return_embeddings else risk
        image = self.fundus_proj(img_feats[:, 1, :])
        risk = self.image_head(image)
        fused = image
        if self.use_covar:
            context = self.covariate_tokens(covar)
            attended = image.unsqueeze(1)
            for layer in self.cross_attn_layers:
                attended = layer(attended, context)
            # Main effects remain accessible without passing through attention.
            risk = (risk + self.covariate_head(context.flatten(1))
                    + self.interaction_head(attended.squeeze(1)))
            fused = torch.cat([attended.squeeze(1), context.mean(1)], dim=-1)
        return (risk, fused) if return_embeddings else risk


class CrossAttentionBlock(nn.Module):
    """Bidirectional cross-attention between image tokens and covariate tokens."""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.img2cov = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cov2img = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_img = nn.Linear(d_model, d_model)
        self.gate_cov = nn.Linear(d_model, d_model)
        self.ffn_img = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.ffn_cov = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.norm_i1 = nn.LayerNorm(d_model)
        self.norm_i2 = nn.LayerNorm(d_model)
        self.norm_c1 = nn.LayerNorm(d_model)
        self.norm_c2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, img_tokens, cov_tokens):
        # Image attends to covariate
        ni = self.norm_i1(img_tokens)
        attn_img, _ = self.img2cov(query=ni, key=cov_tokens, value=cov_tokens)
        img_tokens = img_tokens + torch.sigmoid(self.gate_img(ni)) * self.drop(attn_img)
        img_tokens = img_tokens + self.ffn_img(self.norm_i2(img_tokens))
        # Covariate attends to image
        nc = self.norm_c1(cov_tokens)
        attn_cov, _ = self.cov2img(query=nc, key=img_tokens, value=img_tokens)
        cov_tokens = cov_tokens + torch.sigmoid(self.gate_cov(nc)) * self.drop(attn_cov)
        cov_tokens = cov_tokens + self.ffn_cov(self.norm_c2(cov_tokens))
        return img_tokens, cov_tokens


class CoxHead(nn.Module):
    """Cox regression head: single linear layer → log-hazard risk score."""
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


class MultiModalSurvivalModel(nn.Module):
    """Fundus + Covariate Cox survival model.

    Architecture:
        1. Image: Linear(1024→d_model) — fundus only (slot index 1)
        2. Covariate: Linear(6→d_model)
        3. Image ↔ Covariate cross-attention
        4. Attention pooling → concat(fundus_emb, cov_emb)
        5. Cox head: single Linear layer → risk score

    Backward-compatible key names:
        fundus_proj, fundus_pool, cov_proj, cross_attn_layers, cox_head

    Args:
        active_slots:   []=demo_only, [1]=fundus-based (fundus_only / demo_fundus)
    """
    def __init__(self, img_feat_dim=1024, covar_dim=6, d_model=128,
                 nhead=8, num_cross_attn=1, dropout=0.3,
                 active_slots=None):
        super().__init__()
        self.active_slots = active_slots
        self.d_model = d_model

        # ── demo_only: pure covariate baseline (standard Cox PH) ──
        self.demo_only = (active_slots is not None and len(active_slots) == 0)

        if self.demo_only:
            self.cox_head = CoxHead(covar_dim)
        else:
            # ── Fundus image projection (1024 → d_model) ──
            self.fundus_proj = nn.Sequential(
                nn.Linear(img_feat_dim, d_model),
                nn.LayerNorm(d_model))

            # ── Covariate projection (6 → d_model) ──
            self.cov_proj = nn.Sequential(
                nn.Linear(covar_dim, d_model),
                nn.LayerNorm(d_model))

            # ── Image ↔ Covariate cross-attention ──
            self.cross_attn_layers = nn.ModuleList([
                CrossAttentionBlock(d_model, nhead, dropout)
                for _ in range(num_cross_attn)])

            # ── Attention pooling for fundus tokens ──
            self.fundus_pool = nn.Linear(d_model, 1)

            # ── Cox head: fundus_emb + cov_emb → risk ──
            head_in_dim = d_model * 2
            self.cox_head = CoxHead(head_in_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _pool(self, tokens, pool_layer):
        """Attention-weighted pooling: (B, N, D) → (B, D)."""
        w = pool_layer(tokens).softmax(dim=1)
        return (tokens * w).sum(dim=1)

    def forward(self, img_feats, covar, return_embeddings=False):
        """
        Args:
            img_feats: (B, 2, 1024) — slot 0=zeros, slot 1=fundus
            covar:     (B, 6)      — demographic covariates
        Returns:
            risk: (B, 1) — log-hazard risk score
        """
        # ── demo_only: pure Cox PH ──
        if self.demo_only:
            risk = self.cox_head(covar)
            if return_embeddings:
                return risk, covar
            return risk

        # ── 1. Fundus image projection (slot index 1) ──
        img_tokens = self.fundus_proj(img_feats[:, 1, :]).unsqueeze(1)  # (B, 1, D)

        # ── 2. Covariate projection ──
        cov_tokens = self.cov_proj(covar).unsqueeze(1)  # (B, 1, D)

        # ── 3. Image ↔ Covariate cross-attention ──
        for layer in self.cross_attn_layers:
            img_tokens, cov_tokens = layer(img_tokens, cov_tokens)

        # ── 4. Pool + Concat ──
        cov_emb = cov_tokens.squeeze(1)         # (B, D)
        img_emb = self._pool(img_tokens, self.fundus_pool)  # (B, D)
        fused = torch.cat([img_emb, cov_emb], dim=-1)      # (B, 2*D)

        # ── 5. Cox head ──
        risk = self.cox_head(fused)
        if return_embeddings:
            return risk, fused
        return risk
