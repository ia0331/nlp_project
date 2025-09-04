# src/utils/common.py

import torch, re

# -----------------------------
# 1. 키워드 패턴
# -----------------------------
PNEUMONIA_POS = re.compile(
    r"\b("
    r"pneumonia|"
    r"consolidation|"
    r"opacity|opacities|"
    r"infiltrat(e|ion|es)|"
    r"air[-\s]?space opacity|air[-\s]?space opacities|airspace|"
    r"parenchymal opacity|parenchymal opacities|parenchymal|"
    r"alveolar opacities|alveolar|"
    r"lobar opacity|lobar|"
    r"bronchopneumonia|bronchopneumonic|"
    r"patchy opacity|patchy opacities|patchy|"
    r"multifocal|reticular|nodular|interstitial|"
    r"ggo|ground\s?glass opacity|ground[-\s]?glass opacity|ground\s?glass opacities|"
    r"aspiration pneumonia|aspiration pneumonitis|aspiration|"
    r"infection|infectious"
    r")\b",
    re.I
)

NEGATION = re.compile(
    r"\b(no|without|absence of|negative for|not seen|no evidence of|free of)\b",
    re.I
)

NORMAL_PHRASES = re.compile(
    r"\b("
    r"clear|unremarkable|clear lungs|"
    r"no acute cardiopulmonary (abnormality|process)|"
    r"no focal consolidation|no consolidation|"
    r"within normal limits|"
    r"no focal opacity|no focal opacities"
    r")\b",
    re.I
)

LATERALITY = {
    "L": re.compile(
        r"\b(left|lt\.?|l\s?lobe|l lung|l hemithorax|l\.?\s?lower|l\.?\s?upper)\b", re.I
    ),
    "R": re.compile(
        r"\b(right|rt\.?|r\s?lobe|r lung|r hemithorax|r\.?\s?lower|r\.?\s?upper)\b", re.I
    ),
    "bilateral": re.compile(
        r"\b(bilateral|both lungs|diffuse(ly)?|multifocal|scattered)\b", re.I
    ),
}

CERTAINTY = {
    "definite": re.compile(
        r"\b(consistent with|in keeping with|diagnostic of|definite)\b", re.I
    ),
    "likely": re.compile(
        r"\b(likely|probable|suggestive of)\b", re.I
    ),
    "possible": re.compile(
        r"\b(possible|cannot exclude|may represent|suspicious for)\b", re.I
    ),
}

SEVERITY = re.compile(r"\b(mild|moderate|severe)\b", re.I)

LOBES = re.compile(
    r"\b(RUL|RML|RLL|LUL|LLL|"
    r"right (upper|middle|lower) lobe|"
    r"left (upper|lower) lobe)\b",
    re.I
)

EXCLUDE_POS = re.compile(
    r"\b(pleural effusion|effusion|pneumothorax|osseous|bone)\b", re.I
)

# -----------------------------
# 2. 기존 함수 + 텍스트 분석 함수 추가
# -----------------------------

QUADRANTS = ["left_upper", "left_lower", "right_upper", "right_lower"]

def quadrant_from_heatmap(heatmap: torch.Tensor):
    """heatmap: (H,W) tensor in [0,1] -> mean score per quadrant."""
    H, W = heatmap.shape
    h2, w2 = H//2, W//2
    return {
        "left_upper":  heatmap[:h2, :w2].mean().item(),
        "left_lower":  heatmap[h2:, :w2].mean().item(),
        "right_upper": heatmap[:h2, w2:].mean().item(),
        "right_lower": heatmap[h2:, w2:].mean().item(),
    }

def analyze_text_findings(text: str):
    """
    보고서 텍스트에서 폐렴 관련 소견, 부정, 위치, 확신도, 중증도 등을 추출
    """
    t = text.lower()

    # 정상 패턴 체크
    if NORMAL_PHRASES.search(t):
        return {"status": "normal", "symptoms": []}

    # 폐렴 관련 여부
    has_pneu = bool(PNEUMONIA_POS.search(t))
    negated = bool(NEGATION.search(t))
    if not has_pneu or negated:
        return {"status": "no_pneumonia", "symptoms": []}

    # 위치
    loc = []
    for key, pat in LATERALITY.items():
        if pat.search(t):
            loc.append(key)

    # 확신도
    certainty = None
    for level, pat in CERTAINTY.items():
        if pat.search(t):
            certainty = level
            break

    # 중증도
    sev_match = SEVERITY.search(t)
    severity = sev_match.group(1) if sev_match else None

    # 로브 단위
    lobe_match = LOBES.findall(t)
    lobes = [m[0] if isinstance(m, tuple) else m for m in lobe_match] if lobe_match else []

    return {
        "status": "pneumonia",
        "negated": negated,
        "locations": loc or None,
        "certainty": certainty,
        "severity": severity,
        "lobes": lobes or None
    }

