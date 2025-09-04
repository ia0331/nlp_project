# scripts/generate_report.py
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------
# I/O utils
# -----------------------------
def load_json(p: str) -> Dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)

def save_text(p: str, text: str):
    Path(os.path.dirname(p) or ".").mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(text)

# -----------------------------
# Helpers
# -----------------------------
def map_quadrant_to_location_en(q: Optional[str]) -> str:
    mapping = {
        "left_upper": "the left upper lung field",
        "left_lower": "the left lower lung field",
        "right_upper": "the right upper lung field",
        "right_lower": "the right lower lung field",
    }
    return mapping.get(q, "the lungs")

def extract_global(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    f = payload.get("findings", {}) if "findings" in payload else payload
    g = f.get("global") or {}
    top = g.get("global_top_quadrant") or f.get("global_top_quadrant")
    peak = g.get("peakiness") or f.get("peakiness")
    conf = g.get("confidence") or f.get("confidence")
    try:
        peak = float(peak) if peak is not None else None
    except Exception:
        peak = None
    return top, peak, conf

def verbal_severity(peak: Optional[float], conf: Optional[str]) -> Optional[str]:
    # return only words (no numbers)
    if conf:
        c = conf.lower()
        return {"low":"low", "moderate":"moderate", "high":"high"}.get(c, None)
    if peak is None:
        return None
    if peak < 1.2: return "low"
    if peak < 2.1: return "moderate"
    return "high"

def support_snippets(payload: Dict[str, Any], max_n: int = 2) -> List[str]:
    sents = (payload.get("findings", {}) or {}).get("support_sentences", []) or []
    out = []
    for s in sents:
        s_clean = " ".join(s.strip().split())
        if s_clean:
            out.append(s_clean)
        if len(out) >= max_n:
            break
    return out

# -----------------------------
# LLM providers (optional)
# -----------------------------
def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2, system: Optional[str] = None) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    msgs = []
    if system:
        msgs.append({"role":"system","content":system})
    msgs.append({"role":"user","content":prompt})
    resp = client.chat.completions.create(model=model, temperature=temperature, messages=msgs)
    return resp.choices[0].message.content.strip()

def call_bedrock(prompt: str, model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0", temperature: float = 0.2, system: Optional[str] = None) -> str:
    import boto3, json as _json
    bedrock = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION","us-east-1"))
    sys_prompt = (system or "") + "\nUser:\n" + prompt + "\n\nAssistant:"
    body = {
        "anthropic_version":"bedrock-2023-05-31",
        "max_tokens": 220,
        "temperature": temperature,
        "messages":[{"role":"user","content":sys_prompt}]
    }
    out = bedrock.invoke_model(modelId=model_id, body=_json.dumps(body))
    payload = _json.loads(out["body"].read())
    parts = payload.get("content", [])
    txts = []
    for p in parts:
        if p.get("type") == "text":
            txts.append(p.get("text",""))
    return "\n".join(txts).strip() or payload.get("output_text","").strip()

def build_pneumonia_prompt(payload: Dict[str, Any]) -> Tuple[str, str]:
    f = payload.get("findings", {}) or {}

    # 위치/심도(말 단어) 추출
    top_quad, peak, conf = extract_global(payload)
    loc_text = map_quadrant_to_location_en(top_quad)
    sev_word = verbal_severity(peak, conf)  # "low|moderate|high" or None

    # JSON 안 문장들
    base = " ".join((f.get("base") or "").split())
    sents = [" ".join(s.split()) for s in (f.get("support_sentences") or []) if s and s.strip()]
    variants = [" ".join(s.split()) for s in (f.get("variants") or []) if s and s.strip()]

    # 너무 길지 않게 잘라줌(안전)
    support_txt = " ".join(sents)[:2000]
    variants_txt = " ".join(variants)[:1000]

    system = (
        "You generate FINDINGS for chest X-ray reports.\n"
        "Use only the provided facts from JSON. Do NOT add any pathology or recommendation not present.\n"
        "Output 2–4 sentences in professional radiology English.\n"
        "Do NOT include any numbers, percentages, or scores.\n"
        "Do NOT describe temporal changes (e.g., 'increased', 'decreased', 'compared to').\n"
        "Minimize paraphrasing and keep wording close to the provided phrases.\n"
        "Output only the FINDINGS text."
    )

    sev_line = f"{sev_word} severity" if sev_word else "unspecified"
    user = (
        "Confirmed class: pneumonia.\n"
        f"Global location (verbal): {loc_text}.\n"
        f"Severity (verbal only): {sev_line}.\n"
        f"Base text (if any): {base}\n"
        f"Support sentences (verbatim pool; keep wording close; no numbers): {support_txt}\n"
        f"Variants (optional pool): {variants_txt}\n"
        "Compose 2–4 concise sentences describing parenchymal opacification/infiltration consistent with pneumonia, "
        "mentioning the location and (if present) the verbal severity only. "
        "Avoid temporal comparisons or probabilistic wording. Use phrases already present when possible."
    )
    return system, user
    

# -----------------------------
# Templates (English)
# -----------------------------
def normal_findings_en() -> str:
    # 2–3 sentences, include required phrases
    sentences = [
        "The lungs are clear without focal consolidation.",
        "There is no pneumothorax or pleural effusion identified.",
        "No free air is seen beneath the diaphragm."
    ]
    return " ".join(sentences)

def abnormal_findings_en() -> str:
    # 2–3 sentences, neutral and conservative
    sentences = [
        "Abnormal findings are suspected that do not specifically suggest pneumonia.",
        "Clinical correlation and physician review are recommended."
    ]
    return " ".join(sentences)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, help="infer_pipeline output JSON")
    ap.add_argument("--out_report", required=True, help="path to save FINDINGS .txt")
    ap.add_argument("--provider", choices=["none","openai","bedrock"], default="none")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--bedrock_model", default="anthropic.claude-3-5-sonnet-20240620-v1:0")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    data = load_json(args.in_json)
    payload = data.get("gen_payload", {})
    class_name = (payload.get("class_name") or data.get("prediction") or "").lower()

    if class_name == "pneumonia":
        # JSON 기반 최소 변형 프롬프트로만 생성
        if args.provider in ("openai", "bedrock"):
            sys_p, usr_p = build_pneumonia_prompt(payload)
            try:
                if args.provider == "openai":
                    findings = call_openai(usr_p, model=args.openai_model, temperature=args.temperature, system=sys_p)
                else:
                    findings = call_bedrock(usr_p, model_id=args.bedrock_model, temperature=args.temperature, system=sys_p)
            except Exception as e:
                # LLM 에러 시: 템플릿 없이 JSON 문장으로 최소 조립
                f = payload.get("findings", {}) or {}
                base = " ".join((f.get("base") or "").split())
                sents = [" ".join(s.split()) for s in (f.get("support_sentences") or []) if s.strip()]
                if base:
                    findings = base if not sents else f"{base} {sents[0]}"
                elif sents:
                    findings = " ".join(sents[:2])
                else:
                    # 아주 마지막 안전망
                    top, peak, conf = extract_global(payload)
                    loc_text = map_quadrant_to_location_en(top)
                    sev = verbal_severity(peak, conf)
                    findings = f"Findings in {loc_text} suggest pneumonia" + (f" with overall {sev} severity." if sev else ".")
                findings += f"\n[Note: LLM error; minimal fallback used: {e}]"
        else:
            # provider == none: LLM 없이 JSON 문장만으로 최소 조립
            f = payload.get("findings", {}) or {}
            base = " ".join((f.get("base") or "").split())
            sents = [" ".join(s.split()) for s in (f.get("support_sentences") or []) if s.strip()]
            if base:
                findings = base if not sents else f"{base} {sents[0]}"
            elif sents:
                findings = " ".join(sents[:2])
            else:
                top, peak, conf = extract_global(payload)
                loc_text = map_quadrant_to_location_en(top)
                sev = verbal_severity(peak, conf)
                findings = f"Findings in {loc_text} suggest pneumonia" + (f" with overall {sev} severity." if sev else ".")

    elif class_name == "normal" or class_name == "0":
        findings = normal_findings_en()

    else:  # abnormal / non-pneumonia
        findings = abnormal_findings_en()

    # Always output only FINDINGS text
    findings = " ".join(findings.strip().split())
    save_text(args.out_report, findings)
    print(f"[OK] FINDINGS saved to: {args.out_report}")

if __name__ == "__main__":
    main()


