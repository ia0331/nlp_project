import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel

class ImageEncoderResNet50(nn.Module):
    """
    Global: avgpool (C=2048)
    Local:  layer3 map (C=1024, H',W')
    """
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3   # local feature map
        self.layer4 = resnet.layer4   # for global
        self.avgpool = resnet.avgpool
        self.out_dim_global = 2048
        self.out_dim_local  = 1024

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        lfeat = self.layer3(x)             # (B,1024,H,W)
        x = self.layer4(lfeat)             # (B,2048,H/2,W/2)
        g = self.avgpool(x).squeeze(-1).squeeze(-1)  # (B,2048)
        return g, lfeat

class TextEncoderBioclinicalBERT(nn.Module):
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok = out.last_hidden_state                         # (B,T,H)
        mask = attention_mask.unsqueeze(-1)
        g = (tok * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)  # (B,H)  # global: masked mean
        return g, tok

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x):
        x = self.proj(x)
        return F.normalize(x, dim=-1)


    
class GLoRIAEncoder(nn.Module):
    def __init__(self, proj_dim=256, text_model="emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()
        self.im  = ImageEncoderResNet50(pretrained=True)
        self.txt = TextEncoderBioclinicalBERT(text_model)
        self.vg_head = ProjectionHead(self.im.out_dim_global, proj_dim)  # Rvg
        self.vl_head = ProjectionHead(self.im.out_dim_local,  proj_dim)  # Rvl (after flatten)
        self.tg_head = ProjectionHead(self.txt.hidden,        proj_dim)  # Rtg
        self.tl_head = ProjectionHead(self.txt.hidden,        proj_dim)  # Rtl

    def forward(self, images, input_ids, attention_mask):
        # Image features
        g_img, l_img = self.im(images)                 # (B,2048), (B,1024,h,w)
        B, C, H, W = l_img.shape
        l_img_flat = l_img.view(B, C, H*W).transpose(1,2)  # (B, M, C)
        vg = self.vg_head(g_img)                        # (B,D)
        vl = self.vl_head(l_img_flat)                   # (B,M,D)

        # Text features
        g_txt, tok = self.txt(input_ids, attention_mask)   # (B,H), (B,T,H)
        tg = self.tg_head(g_txt)                       # (B,D)
        tl = self.tl_head(tok)                         # (B,T,D)
        return vg, vl, tg, tl
    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B,3,H,W] (이미 infer_pipeline의 transform 거친 텐서)
        return: vg [B, D]  (정규화된 projection 공간 임베딩)
        """ 
        g_img, l_img = self.im(images)               # (B,2048), (B,1024,h,w)
        vg = self.vg_head(g_img)                     # (B,D), already normalized in ProjectionHead
        return vg
    
def load_gloria_checkpoint(gloria: GLoRIAEncoder, ckpt_path: str, map_location="cpu"):
    """
    학습해둔 GLoRIA 체크포인트(같은 proj_dim)를 읽어 projection 공간 정렬을 맞춘다.
    ckpt는 state_dict 형태라고 가정.
    """
    sd = torch.load(ckpt_path, map_location=map_location)
    gloria.load_state_dict(sd, strict=False)
    gloria.eval()
    return gloria
# src/models/encoders.py
import torch
import torch.nn.functional as F


def attention_weighted_context(vl, tl, tau_local=0.1, attn_mask=None):
    """
    vl: [B, M, D]  (image patches)
    tl: [B, T, D]  (text tokens)
    attn_mask: [B, T] (1=valid, 0=PAD) or None
    returns:
      C: [B, T, D], A: [B, T, M]
    """
    s = torch.bmm(tl, vl.transpose(1, 2)) / tau_local  # [B,T,M]

    if attn_mask is not None:
        # mask: 1(valid) / 0(PAD) → PAD 토큰의 행은 매우 작은 값으로 밀어넣기
        # (-1e9은 softmax 후 0에 수렴; -inf로 두면 NaN이 발생할 수 있음)
        s = s + (attn_mask.unsqueeze(-1).to(s.dtype) - 1.0) * 1e9

    A = F.softmax(s, dim=-1)              # [B,T,M]
    if attn_mask is not None:
        # PAD 토큰의 attention을 0으로 강제 (NaN 안전)
        A = A * attn_mask.unsqueeze(-1).to(A.dtype)

    # (선택) 정규화: PAD 토큰 행이 전부 0이면 합이 0 → 0으로 나누기 방지
    denom = A.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    A = A / denom

    C = torch.bmm(A, vl)                  # [B,T,D]
    return C, A

def local_matching_score(C, tl, tau_match=0.07):
    """
    Eq.(6): Z = log sum_i exp(<ci, ti>/tau3) * tau3.
    """
    sim = (C * tl).sum(dim=-1) / tau_match   # (B,T)
    z = torch.logsumexp(sim, dim=-1) * tau_match  # (B,)
    return z

