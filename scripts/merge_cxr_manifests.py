#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import sys

def add_prefix_to_filename(val: str, prefix: str) -> str:
    """file_name에 prefix를 붙임. 이미 붙어있으면 그대로 둠.
       file_name에 경로가 들어있어도 basename만 남기고 붙입니다."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    prefix = prefix.strip().rstrip("/")
    if s.startswith(prefix + "/"):
        return s
    # OS 구분자 혼용 대비: 마지막 파일명만 추출
    base = s.replace("\\", "/").split("/")[-1]
    return f"{prefix}/{base}"

def main():
    ap = argparse.ArgumentParser(description="Merge two CSVs and prefix 'file_name' with a given path.")
    ap.add_argument("--normal", required=True, help="Path to final_normal.csv")
    ap.add_argument("--pneumo", required=True, help="Path to final_pneumo.csv")
    ap.add_argument("--out", required=True, help="Output merged.csv path")
    ap.add_argument("--prefix", default="data/imdat", help="Prefix to add before file_name (default: data/imdat)")
    ap.add_argument("--drop-dups", choices=["true","false"], default="false",
                    help="Drop duplicate rows based on file_name after prefixing (default: false)")
    args = ap.parse_args()

    normal_path = Path(args.normal).expanduser()
    pneumo_path = Path(args.pneumo).expanduser()
    out_path = Path(args.out).expanduser()
    drop_dups = (args.drop_dups.lower() == "true")

    if not normal_path.exists():
        print(f"[ERR] normal CSV not found: {normal_path}", file=sys.stderr); sys.exit(1)
    if not pneumo_path.exists():
        print(f"[ERR] pneumo CSV not found: {pneumo_path}", file=sys.stderr); sys.exit(1)

    df_n = pd.read_csv(normal_path)
    df_p = pd.read_csv(pneumo_path)

    if "file_name" not in df_n.columns or "file_name" not in df_p.columns:
        print("[ERR] Both CSVs must contain a 'file_name' column.", file=sys.stderr); sys.exit(1)

    merged = pd.concat([df_n, df_p], ignore_index=True, sort=False)

    # file_name 앞에 prefix 붙이기 (이미 있으면 중복 방지)
    merged["file_name"] = merged["file_name"].apply(lambda v: add_prefix_to_filename(v, args.prefix))

    # (선택) file_name 기준 중복 제거
    if drop_dups:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["file_name"]).reset_index(drop=True)
        print(f"[INFO] Dropped duplicates by file_name: {before - len(merged)} rows")

    # 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    print("==== SUMMARY ====")
    print(f"normal rows : {len(df_n)}")
    print(f"pneumo rows : {len(df_p)}")
    print(f"merged rows : {len(merged)}")
    print(f"output      : {out_path}")

if __name__ == "__main__":
    main()


