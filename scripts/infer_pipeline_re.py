# scripts/infer_pipeline.py
# -*- coding: utf-8 -*-

import os, json, argparse, torch
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision import models
import re
# >>> 추가 의존성
import chromadb
import numpy as np
import cv2
from transformers import AutoTokenizer

# >>> encoders 임포트 (+ attention 유틸)
from src.models.encoders import GLoRIAEncoder, load_gloria_checkpoint
from src.models.encoders import attention_weighted_context


# -----------------------------
# 0) 상수/키워드
# -----------------------------
FINDING_KEYWORDS = {
    "consolidation": ["consolidation", "consolidative"],
    "opacity": ["opacity", "opacities", "air-space", "airspace", "opacification"],
    "infiltrate": ["infiltrate", "infiltration", "infiltrates"],
    "effusion": ["effusion", "pleural"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "bronchial_wall_thickening": ["bronchial", "peribronchial", "wall", "thickening"],
}
ALL_ANCHOR_TOKENS = sorted({t for toks in FINDING_KEYWORDS.values() for t in toks})

# 문장 수준 부정 패턴 (리트리벌 문장 필터용)
NEG_PATTERNS_SENT = (
    "no ", "not ", "without ", "absence of", "absent ", "negative for",
    "rule out", "ruled out", "unlikely", "less likely", "free of"
)

# 토큰 레벨 부정 판정 파라미터
NEG_WINDOW = 4  # 토큰 기준 n-gram 창
NEG_TOKENS = {
    "no", "not", "without", "absence", "absent", "free", "negative", "deny",
    "denies", "unlikely"
}

# Peakiness(집중도) 임계치 (원하면 조정)
# - global: Grad-CAM 맵에서 상위 p%의 평균 / 전체 평균
# - local(GLoRIA): 토큰-패치 어텐션 맵에서 상위 p%의 평균 / 전체 평균
PEAK_TOP_P = 0.10
PEAKINESS_STRONG = 2.5   # 아주 또렷
PEAKINESS_MODERATE = 1.6 # 적당히 또렷
ACTIVE_TAU_LOCAL = 0.12  # 로컬 어텐션 온도(예시)

# -----------------------------
# 1) 전처리 & 모델 로딩
# -----------------------------
def _norm_ws_lower(s: str) -> str:
    # 공백/개행 통합 + trim + 소문자
    return re.sub(r"\s+", " ", (s or "")).strip().lower()

def dedup_keep_order(strings):
    """공백/대소문자만 다른 중복 제거(순서 보존). 출력은 공백만 정리."""
    seen = set()
    out = []
    for s in strings:
        if not s:
            continue
        key = _norm_ws_lower(s)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(" ".join(s.strip().split()))
    return out

def build_transform(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def load_resnet50(model_path: str, num_classes: int = 3, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, device

def _normalize_space(s: str) -> str:
    return " ".join(s.split())

def filter_support_sentences(sentences: List[str]) -> List[str]:
    """
    문장 수준: 강한 부정/배제 문장 제거 + CXR 비관련 문장 제거
    (단, 'cannot exclude'같은 애매 표현은 유지)
    """
    if not sentences:
        return []
    out = []
    for s in sentences:
        s_norm = _normalize_space(s)
        low = s_norm.lower()

        # CXR 관련 키워드가 전혀 없으면 스킵 (pneumon*만 있으면 유지)
        if all(k not in low for k in (
            "lung", "pulmo", "chest", "lobe", "air", "opacity", "infiltrat",
            "consolidat", "effusion", "pleural", "atelect"
        )):
            if "pneumon" not in low:
                continue

        # 강한 부정/배제 문장 스킵 (단 cannot exclude는 유지)
        if any(pat in low for pat in NEG_PATTERNS_SENT) and "cannot exclude" not in low:
            continue

        out.append(s_norm)
    return out

def _is_negated_token(tokens: List[str], idx: int, window: int = NEG_WINDOW) -> bool:
    """
    BERT wordpiece 토큰 목록에서 특정 키워드 토큰 idx 좌측 window 내에 부정 단서가 있으면 True
    """
    l = max(0, idx - window)
    span = [t.replace("##", "").lower() for t in tokens[l:idx+1]]
    joined = " ".join(span)
    if "rule out" in joined or "less likely" in joined:
        return True
    return any(t in NEG_TOKENS for t in span)

# -----------------------------
# 2) 임베딩 & 벡터DB 검색 어댑터
# -----------------------------
class PneumoRetriever:
    """
    폐렴(pred=1)으로 '판정된 경우에만' 사용.
    query: GLoRIA의 이미지 임베딩(vg, [1,D])  # 정규화된 투영 공간
    return: support sentences (Top-K), 메타데이터
    """
    def __init__(self, chroma_path: str, collection: str, top_k: int = 6, oversample: int = 8):
        self.top_k = top_k
        self.oversample = oversample  # <- 추가: 아래 순위까지 넉넉히 가져오기
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space":"cosine"}
        )

    @torch.no_grad()
    def search(self, img_emb: torch.Tensor) -> Dict[str, Any]:
        q = img_emb.detach().cpu().numpy().astype(np.float32).tolist()

        # 1) 먼저 top_k * oversample 만큼 오버샘플링으로 가져온다.
        n_fetch = max(self.top_k, self.top_k * self.oversample)
        res = self.col.query(query_embeddings=q, n_results=n_fetch)

        all_docs  = res.get("documents", [[]])[0]
        all_metas = res.get("metadatas", [[]])[0]
        all_ids   = res.get("ids", [[]])[0]
        all_dists = (res.get("distances", [[]]) or [[]])[0]

        # 2) 순위(유사도) 유지하며 정규화 중복 제거
        #    (동일/유사 공백만 다른 문장은 한 번만)
        dedup_docs = []
        dedup_metas, dedup_ids, dedup_dists = [], [], []
        seen = set()
        for d, m, i, dist in zip(all_docs, all_metas, all_ids, all_dists):
            key = _norm_ws_lower(d)
            if not key or key in seen:
                continue
            seen.add(key)
            dedup_docs.append(" ".join((d or "").strip().split()))
            dedup_metas.append(m)
            dedup_ids.append(i)
            dedup_dists.append(dist)

        # 3) 필터 적용하면서 위에서부터 통과하는 것만 뽑아 top_k 채움
        filtered_docs = []
        filtered_metas, filtered_ids, filtered_dists = [], [], []

        # filter_support_sentences는 리스트 전체를 입력받아 통과 리스트를 돌려주므로
        # 한 문장씩 검사될 수 있도록 작은 배치로 적용
        for d, m, i, dist in zip(dedup_docs, dedup_metas, dedup_ids, dedup_dists):
            kept = filter_support_sentences([d])
            if kept:  # 통과
                filtered_docs.append(kept[0])
                filtered_metas.append(m)
                filtered_ids.append(i)
                filtered_dists.append(dist)
                if len(filtered_docs) >= self.top_k:
                    break

        # 4) 아직 모자라면(필터가 너무 빡세서) 필터를 통과 못한 문장 중에서
        #    "그다음으로 유사한 애들"을 위에서부터 보충 (요청사항에 맞춰 백필)
        if len(filtered_docs) < self.top_k:
            # 필터 실패한 후보들을 다시 수집 (순위 그대로)
            failed_docs = []
            failed_metas, failed_ids, failed_dists = [], [], []
            for d, m, i, dist in zip(dedup_docs, dedup_metas, dedup_ids, dedup_dists):
                # 이미 채택된 것은 건너뛴다
                if d in filtered_docs:
                    continue
                if not filter_support_sentences([d]):  # 필터 탈락 케이스
                    failed_docs.append(d)
                    failed_metas.append(m)
                    failed_ids.append(i)
                    failed_dists.append(dist)
            # 위에서부터 채워 넣기
            for d, m, i, dist in zip(failed_docs, failed_metas, failed_ids, failed_dists):
                filtered_docs.append(d)
                filtered_metas.append(m)
                filtered_ids.append(i)
                filtered_dists.append(dist)
                if len(filtered_docs) >= self.top_k:
                    break

        # 최종 top_k만 자르기
        out_docs  = filtered_docs[: self.top_k]
        out_metas = filtered_metas[: self.top_k]
        out_ids   = filtered_ids[: self.top_k]
        out_dists = filtered_dists[: self.top_k]

        return {
            "support_sentences": out_docs,     # <- 항상 위에서부터 채워진 top_k
            "metadatas": out_metas,
            "ids": out_ids,
            "distances": out_dists,
            # (옵션) 디버깅용 필드가 필요하면 주석 해제
            # "raw_candidates": all_docs,
            # "filtered_count": sum(1 for d in dedup_docs if filter_support_sentences([d])),
        }

# -----------------------------
# 3) 마스크/사분면 + Grad-CAM + Peakiness
# -----------------------------
def build_lung_mask_from_segmented_image(img_pil: Image.Image, image_size: int) -> torch.Tensor:
    """
    세그된 CXR(배경=검정, 폐=밝음)을 가정하여 마스크 추정.
    return: [H,W] float tensor in {0,1}
    """
    t = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
    x = t(img_pil)  # [C,H,W]
    # 채널 합이 임계 이상이면 '폐'
    mask = (x.sum(dim=0) > 0.05).float()
    return mask

def gradcam_heatmap_resnet(model: torch.nn.Module, x: torch.Tensor, target_idx: int) -> torch.Tensor:
    """
    x: [1,3,H,W] normalized
    return: heatmap [H,W] in [0,1] (cpu)
    """
    feats, grads = [], []

    def f_hook(m, i, o): feats.append(o)
    def b_hook(m, gi, go): grads.append(go[0])

    h1 = model.layer4.register_forward_hook(f_hook)
    h2 = model.layer4.register_full_backward_hook(b_hook)

    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = model(x)
        score = out[0, target_idx]
        score.backward()

    g = grads[-1]        # [1,C,h,w]
    f = feats[-1].detach()  # [1,C,h,w]
    w = g.mean(dim=(2,3), keepdim=True)  # [1,C,1,1]
    cam = F.relu((w * f).sum(dim=1, keepdim=True))  # [1,1,h,w]
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(0).squeeze(0).detach().cpu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)

    h1.remove(); h2.remove()
    return cam

def _bbox_from_mask(mask: torch.Tensor) -> Tuple[int,int,int,int]:
    """ mask: [H,W] bool/float -> 유효 bbox(y0,y1,x0,x1) """
    m = mask > 0.5
    if m.sum() == 0:
        H, W = mask.shape
        return 0, H-1, 0, W-1
    ys = torch.where(m.any(dim=1))[0]
    xs = torch.where(m.any(dim=0))[0]
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

def quadrant_scores_from_mask(heatmap: torch.Tensor, mask: torch.Tensor) -> Tuple[dict, str]:
    """
    heatmap: [H,W] in [0,1], mask: [H,W] in {0,1}
    폐 마스크의 bounding box만 4등분 → 각 분면 평균값 → 최상 분면
    """
    H, W = heatmap.shape
    y0, y1, x0, x1 = _bbox_from_mask(mask)
    ym = (y0 + y1) // 2
    xm = (x0 + x1) // 2

    def mean_in(yb, ye, xb, xe):
        subm = (mask[yb:ye, xb:xe] > 0.5).float()
        subh = heatmap[yb:ye, xb:xe]
        val = (subh * subm).sum() / (subm.sum() + 1e-9)
        return float(val.item())

    scores = {
        "left_upper":  mean_in(y0, ym+1, x0, xm+1),
        "left_lower":  mean_in(ym+1, y1+1, x0, xm+1),
        "right_upper": mean_in(y0, ym+1, xm+1, x1+1),
        "right_lower": mean_in(ym+1, y1+1, xm+1, x1+1),
    }
    top_quad = max(scores, key=scores.get)
    return scores, top_quad

def peakiness_ratio(arr: torch.Tensor, mask: torch.Tensor, top_p: float = PEAK_TOP_P) -> float:
    """
    집중도(peakiness): 상위 p% 값의 평균 / 전체 평균 (마스크 내)
    값이 클수록 '한 곳에 뾰족하게' 집중됨.
    """
    a = arr.clone().float()
    m = (mask > 0.5)
    vals = a[m].view(-1)
    if vals.numel() == 0:
        return 0.0
    k = max(1, int(vals.numel() * top_p))
    topk, _ = torch.topk(vals, k)
    return (topk.mean() / (vals.mean() + 1e-9)).item()

# -----------------------------
# 4) GLoRIA 국소화(+ 부정어 토큰 필터) & Peakiness
# -----------------------------
def _token_indices_for(tokens: List[str], keywords: List[str]) -> List[int]:
    idxs = []
    needles = set([k.lower() for k in keywords])
    for i, t in enumerate(tokens):
        c = t.replace("##", "").lower()
        if c in needles:
            idxs.append(i)
    return idxs

@torch.no_grad()
def localize_findings_with_gloria(
    gloria: GLoRIAEncoder,
    tokenizer,
    x: torch.Tensor,              # [1,3,H,W] normalized
    lung_mask: torch.Tensor,      # [H,W] in {0,1}
    sentences: List[str],
    tau_local: float = ACTIVE_TAU_LOCAL,
    max_len: int = 64,
) -> List[Dict[str, Any]]:
    """
    support_sentences를 토큰화 → finding 키워드 토큰의 어텐션을 이미지 패치로 투영
    → 폐 마스크 기준 4분면 점수/최상 분면 + peakiness 산출
    - 토큰 레벨 부정어(_is_negated_token) 감지 시 해당 인스턴스는 제외
    """
    if not sentences:
        return []

    device = next(gloria.parameters()).device
    H, W = x.shape[-2:]

    # 이미지 local feature → [1, M, D]
    _, l_img = gloria.im(x)  # [1,1024,h,w]
    _, C, h, w = l_img.shape
    vl = gloria.vl_head(l_img.view(1, C, h*w).transpose(1, 2))  # [1,M,D]

    lung_mask = lung_mask.to(vl.device)

    localized = []
    for sent in sentences:
        tok = tokenizer(
            [sent],
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        out = gloria.txt.model(
            input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]
        )
        tl_h = out.last_hidden_state          # [1,T,Htxt]
        tl = gloria.tl_head(tl_h)             # [1,T,D] (norm)

        # 로컬 어텐션
        C_ctx, A = attention_weighted_context(
            vl, tl, tau_local=tau_local, attn_mask=tok["attention_mask"]
        )  # A: [1,T,M]
        A = A[0]                               # [T,M]
        tokens = tokenizer.convert_ids_to_tokens(tok["input_ids"][0].tolist())

        for finding, kws in FINDING_KEYWORDS.items():
            idxs_all = _token_indices_for(tokens, kws)
            if not idxs_all:
                continue

            # 부정어 토큰 근처에 있는 키워드 인스턴스 제거
            idxs = [i for i in idxs_all if not _is_negated_token(tokens, i)]

            if not idxs:
                continue

            # (부정어 제외) 해당 토큰들의 어텐션 평균
            amap = A[idxs, :].mean(dim=0)  # [M]
            amap = amap.view(h, w).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
            amap = F.interpolate(amap, size=(H, W), mode="bilinear", align_corners=False).squeeze()
            amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-9)

            # 마스크 적용
            amap = (amap * lung_mask).cpu()
            q_scores, q_top = quadrant_scores_from_mask(amap, lung_mask.cpu())
            pk = peakiness_ratio(amap, lung_mask.cpu(), top_p=PEAK_TOP_P)

            localized.append({
                "finding": finding,
                "sentence": sent,
                "quadrant_scores": q_scores,
                "top_quadrant": q_top,
                "peakiness": pk,
                # 해석 보조 태그(선택)
                "confidence": (
                    "high" if pk >= PEAKINESS_STRONG else
                    "moderate" if pk >= PEAKINESS_MODERATE else
                    "low"
                )
            })

    return localized

# -----------------------------
# 5) 분류 추론
# -----------------------------
def predict(model, device, img: Image.Image, tfm) -> Dict[str, Any]:
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred = int(torch.argmax(logits, dim=1).item())
    return {"pred": pred, "probs": probs, "tensor": x}

# -----------------------------
# 6) 리포트 입력 패키징 (폐렴일 때만 리트리벌)
# -----------------------------
def build_generation_payload(
    pred: int,
    probs: List[float],
    x: torch.Tensor,
    model: torch.nn.Module,
    retriever: "PneumoRetriever | None",
    gloria: "GLoRIAEncoder | None",
    tokenizer: "AutoTokenizer | None",
    org_img: Image.Image,
    image_size: int,
) -> Dict[str, Any]:

    # 공통: 세그된 입력으로 폐 마스크 구성
    lung_mask = build_lung_mask_from_segmented_image(org_img, image_size)

    if pred == 1:  # PNEUMONIA
        # 1) Grad-CAM(global) → 분면 + peakiness
        heatmap = gradcam_heatmap_resnet(model, x, target_idx=1)  # class index=1 == pneumonia
        quad_scores, top_quad = quadrant_scores_from_mask(heatmap, lung_mask)
        peak_global = peakiness_ratio(heatmap, lung_mask, top_p=PEAK_TOP_P)

        # 2) 이미지 임베딩 기반 support sentences 검색
        support = {"support_sentences": [], "metadatas": [], "ids": []}
        if retriever is not None and gloria is not None:
            img_emb = gloria.encode_image(x)  # normalized [1,D]
            support = retriever.search(img_emb)
        sentences = support.get("support_sentences", [])

        # 3) GLoRIA 로컬 어텐션으로 국소화 (부정어 토큰 제외)
        localized = []
        if gloria is not None and tokenizer is not None and sentences:
            localized = localize_findings_with_gloria(
                gloria, tokenizer, x, lung_mask, sentences, tau_local=ACTIVE_TAU_LOCAL
            )

        return {
            "class_name": "pneumonia",
            "probs": probs,
            "findings": {
                # >>> '중증도' 없음. 위치와 집중도만 제공
                "global": {
                    "global_quadrant_scores": quad_scores,
                    "global_top_quadrant": top_quad,
                    "peakiness": peak_global,
                    "confidence": (
                        "high" if peak_global >= PEAKINESS_STRONG else
                        "moderate" if peak_global >= PEAKINESS_MODERATE else
                        "low"
                    )
                },
                "support_sentences": sentences,
                "localized_findings": localized
            }
        }

    elif pred == 0:  # NORMAL
        return {
            "class_name": "normal",
            "probs": probs,
            "evidence": {},
            "findings": {
                # 템플릿: 경미한 표현 변주만 남김
                "base": (
                    "Clear lungs without focal consolidation. "
                    "No pneumothorax or pleural effusion is identified. "
                    "No free air is seen beneath the diaphragm."
                ),
                "variants": [
                    "The lungs are clear with no focal air-space opacity.",
                    "No evidence of pneumothorax, pleural effusion, or lobar consolidation.",
                    "No acute cardiopulmonary abnormality is apparent."
                ]
            }
        }

    else:  # ABNORMAL(2) - 폐렴은 아님
        return {
            "class_name": "abnormal",
            "probs": probs,
            "evidence": {
                "note": "Non-pneumonia abnormality suspected; physician review recommended."
            },
            "findings": {
                "base": (
                    "Abnormal findings are suspected, but do not suggest pneumonia. "
                    "Clinical correlation and physician review are recommended."
                ),
                "variants": [
                    "Non-pneumonia abnormality likely; further evaluation is advised.",
                    "Findings may be outside the scope of pneumonia; please correlate clinically."
                ]
            }
        }

# -----------------------------
# 7) CLI & main
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True)  # 3-class model.pth
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--image_size", type=int, default=224)

    # >>> 추가 인자 (옵션)
    ap.add_argument("--gloria_ckpt", type=str, default=None, help="trained GLoRIA state_dict path")
    ap.add_argument("--chroma_path", type=str, default=None, help="ChromaDB persistent dir")
    ap.add_argument("--chroma_collection", type=str, default="pneumo_reports")
    ap.add_argument("--top_k_support", type=int, default=6)
    ap.add_argument("--tokenizer_name", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    return ap.parse_args()

def main():
    a = parse_args()

    tfm = build_transform(a.image_size)
    model, device = load_resnet50(a.model, num_classes=3)
    img = Image.open(a.image).convert("RGB")

    out = predict(model, device, img, tfm)

    # >>> GLoRIA(optional) & Retriever(optional)
    gloria = None
    retriever = None
    tokenizer = None
    if a.gloria_ckpt and a.chroma_path:
        gloria = GLoRIAEncoder(proj_dim=256)  # proj_dim은 ckpt와 동일해야 함
        gloria = load_gloria_checkpoint(gloria, a.gloria_ckpt, map_location=device)
        gloria.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(a.tokenizer_name)
        retriever = PneumoRetriever(
            chroma_path=a.chroma_path,
            collection=a.chroma_collection,
            top_k=a.top_k_support
        )

    payload = build_generation_payload(
        pred=out["pred"],
        probs=out["probs"],
        x=out["tensor"],
        model=model,
        retriever=retriever,
        gloria=gloria,
        tokenizer=tokenizer,
        org_img=img,
        image_size=a.image_size,
    )

    out_dict = {
        "prediction": payload["class_name"],
        "probs": payload["probs"],
        "gen_payload": payload,
    }

    Path(os.path.dirname(a.out_json)).mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(out_dict, f, indent=2)

if __name__ == "__main__":
    main()
