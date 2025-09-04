# -*- coding: utf-8 -*-
"""
Batch runner for:
1) scripts/infer_pipeline_re.py  -> JSON
2) scripts/generate_report.py    -> FINDINGS text
3) Append 'report' and 'label' to CSV (keeping 'file_name' intact)

Usage example:
python scripts/batch_infer_and_report.py \
  --csv data/test.csv \
  --model checkpoints/model.pth \
  --infer_script scripts/infer_pipeline_re.py \
  --report_script scripts/generate_report.py \
  --out_csv data/test_with_reports.csv \
  --work_dir outputs \
  --image_size 224 \
  --provider none
  # (옵션) --gloria_ckpt ckpts/gloria.pth --chroma_path ./chroma --chroma_collection pneumo_reports --top_k_support 6 --tokenizer_name emilyalsentzer/Bio_ClinicalBERT
"""

import os
import re
import sys
import json
import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def run_cmd(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}")
    return r

def strip_findings_prefix(text: str) -> str:
    # 앞뒤 공백 정리 + 'FINDINGS:' / 'Findings:' 같은 prefix 제거
    t = (text or "").strip()
    t = re.sub(r'^\s*findings\s*:\s*', '', t, flags=re.IGNORECASE)
    return t.strip()

def map_prediction_to_label(pred: str) -> int:
    """
    3-class 매핑: normal=0, pneumonia=1, abnormal=2
    infer_pipeline_re.py의 out_json["prediction"]가 문자열로 들어온다고 가정.
    """
    p = (pred or "").strip().lower()
    if p in ("0", "normal"): return 0
    if p in ("1", "pneumonia"): return 1
    if p in ("2", "abnormal"): return 2
    # 알 수 없으면 2(비정상)로 처리하거나 -1로 남겨도 됨. 여기선 -1로 명시.
    return -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV (must contain 'file_name' column)")
    ap.add_argument("--model", required=True, help="3-class model.pth for infer script")
    ap.add_argument("--infer_script", default="scripts/infer_pipeline_re.py")
    ap.add_argument("--report_script", default="scripts/generate_report.py")
    ap.add_argument("--out_csv", required=True, help="Output CSV with added 'report','label' columns")
    ap.add_argument("--work_dir", default="outputs", help="Where to write intermediate json/txt")
    ap.add_argument("--image_size", type=int, default=224)

    # generate_report options
    ap.add_argument("--provider", choices=["none","openai","bedrock"], default="none")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--bedrock_model", default="anthropic.claude-3-5-sonnet-20240620-v1:0")
    ap.add_argument("--temperature", type=float, default=0.2)

    # optional GLoRIA / Chroma for infer
    ap.add_argument("--gloria_ckpt", default=None)
    ap.add_argument("--chroma_path", default=None)
    ap.add_argument("--chroma_collection", default="pneumo_reports")
    ap.add_argument("--top_k_support", type=int, default=6)
    ap.add_argument("--tokenizer_name", default="emilyalsentzer/Bio_ClinicalBERT")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "file_name" not in df.columns:
        raise ValueError("Input CSV must have a 'file_name' column.")

    work_dir = Path(args.work_dir)
    json_dir = work_dir / "json"
    rep_dir  = work_dir / "reports"
    json_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    labels  = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="[Batch] infer + report"):
        img_path = str(row["file_name"])
        stem = Path(img_path).stem

        out_json = json_dir / f"{stem}.json"
        out_txt  = rep_dir  / f"{stem}.txt"

        # 1) infer -> JSON
        infer_cmd = [
            sys.executable, args.infer_script,
            "--image", img_path,
            "--model", args.model,
            "--out_json", str(out_json),
            "--image_size", str(args.image_size)
        ]
        # optional GLoRIA + Chroma
        if args.gloria_ckpt and args.chroma_path:
            infer_cmd += [
                "--gloria_ckpt", args.gloria_ckpt,
                "--chroma_path", args.chroma_path,
                "--chroma_collection", args.chroma_collection,
                "--top_k_support", str(args.top_k_support),
                "--tokenizer_name", args.tokenizer_name
            ]

        try:
            run_cmd(infer_cmd)
        except Exception as e:
            # 실패 시 빈 값 처리
            reports.append("")
            labels.append(-1)
            print(f"[WARN] infer failed for {img_path}: {e}")
            continue

        # prediction 읽기
        try:
            with open(out_json, "r") as f:
                data = json.load(f)
            pred_str = data.get("prediction", "")
        except Exception as e:
            pred_str = ""
            print(f"[WARN] failed to read prediction from {out_json}: {e}")

        # 2) generate_report -> txt
        rep_cmd = [
            sys.executable, args.report_script,
            "--in_json", str(out_json),
            "--out_report", str(out_txt),
            "--provider", args.provider,
            "--openai_model", args.openai_model,
            "--bedrock_model", args.bedrock_model,
            "--temperature", str(args.temperature),
        ]
        try:
            run_cmd(rep_cmd)
        except Exception as e:
            print(f"[WARN] report generation failed for {img_path}: {e}")

        # 텍스트 읽고 'FINDINGS:' prefix 제거(혹시 존재하면)
        rep_text = ""
        try:
            if out_txt.exists():
                rep_text = Path(out_txt).read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WARN] cannot read report txt for {img_path}: {e}")

        rep_text = strip_findings_prefix(rep_text)
        label_id = map_prediction_to_label(pred_str)

        reports.append(rep_text)
        labels.append(label_id)

    # 3) CSV에 붙이기 (file_name은 그대로 유지)
    df = df.copy()
    df["report"] = reports
    df["label"]  = labels

    out_csv_path = Path(args.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"[OK] Saved: {out_csv_path}")

if __name__ == "__main__":
    main()
