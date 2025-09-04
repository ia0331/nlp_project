import torch
import torch.nn.functional as F
from .utils import gather_all
from ..models.encoders import attention_weighted_context  # local_matching_score 안 씀

def contrastive_global(vg, tg, tau=0.07):
    # 양방향 InfoNCE
    tau = max(float(tau), 1e-6)
    vg = F.normalize(vg, dim=-1)
    tg = F.normalize(tg, dim=-1)

    all_vg = gather_all(vg)  # DDP가 아니면 자기 배치만 반환하도록 구현되어 있어야 함
    all_tg = gather_all(tg)

    logits_i2t = (vg @ all_tg.t()) / tau
    logits_t2i = (tg @ all_vg.t()) / tau

    # NaN/Inf 안전 처리
    logits_i2t = torch.nan_to_num(logits_i2t, nan=0.0, posinf=1e4, neginf=-1e4)
    logits_t2i = torch.nan_to_num(logits_t2i, nan=0.0, posinf=1e4, neginf=-1e4)

    labels = torch.arange(vg.size(0), device=vg.device)
    loss_i2t = F.cross_entropy(logits_i2t, labels)
    loss_t2i = F.cross_entropy(logits_t2i, labels)
    return loss_i2t + loss_t2i


def contrastive_local(vl, tl, tau_local=0.1, tau_match=0.1, attn_mask=None):
    """
    vl: [B, M, D]
    tl: [B, T, D]
    attn_mask: [B, T] (1=valid, 0=pad) or None
    """
    B = vl.size(0)
    tau_match = max(float(tau_match), 1e-6)

    # 배치 연산 컨텍스트/어텐션 (encoders.attention_weighted_context는 -1e9 마스킹 + 재정규화로 수정되어 있어야 함)
    C, A = attention_weighted_context(vl, tl, tau_local=tau_local, attn_mask=attn_mask)  # C: [B,T,D]

    # token-level 매칭 점수
    scores = (tl * C).sum(dim=-1)  # [B, T]
    scores = torch.nan_to_num(scores, nan=0.0)  # 혹시 모를 NaN 제거

    if attn_mask is not None:
        # PAD 토큰은 0으로 두고, 유효 토큰 수로 평균
        scores = scores.masked_fill(attn_mask == 0, 0.0)
        valid = attn_mask.sum(dim=1).clamp_min(1)       # [B]
        pos = scores.sum(dim=1) / valid                 # [B]
    else:
        pos = scores.mean(dim=1)                        # [B]

    logits = pos[:, None].repeat(1, B) / tau_match      # [B,B]
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

    labels = torch.arange(B, device=vl.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
    return loss
