# -*- coding: utf-8 -*-
"""
Compare SBERT similarity between 'output.csv' and 'test.csv' report texts,
and additionally run with PubMedBERT to save a second histogram.

Usage:
python scripts/compare_sbert.py \
  --gen_csv data/test_with_reports.csv \
  --ref_csv data/test.csv \
  --out_csv outputs/report_similarity.csv \
  --hist_png outputs/similarity_hist.png \
  --gen_col report \
  --ref_col report \
  --pub_hist_png outputs/similarity_pub.png \
  --pubmed_model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
"""

import os
import re
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def norm_ws(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()

def load_and_merge(gen_csv: str, ref_csv: str, gen_col: str, ref_col: str) -> pd.DataFrame:
    gen = pd.read_csv(gen_csv)
    ref = pd.read_csv(ref_csv)
    gen.columns = gen.columns.str.strip()
    ref.columns = ref.columns.str.strip()

    if "file_name" not in gen.columns:
        raise ValueError(f"{gen_csv} must contain 'file_name' column.")
    if "file_name" not in ref.columns:
        raise ValueError(f"{ref_csv} must contain 'file_name' column.")
    if gen_col not in gen.columns:
        raise ValueError(f"{gen_csv} must contain '{gen_col}' column. Available: {list(gen.columns)}")
    if ref_col not in ref.columns:
        raise ValueError(f"{ref_csv} must contain '{ref_col}' column. Available: {list(ref.columns)}")

    gen = gen.drop_duplicates(subset=["file_name"])
    ref = ref.drop_duplicates(subset=["file_name"])

    ref_ren = ref[["file_name", ref_col]].rename(columns={ref_col: "ref_report"})
    gen_ren = gen[["file_name", gen_col]].rename(columns={gen_col: "gen_report"})
    df = pd.merge(ref_ren, gen_ren, on="file_name", how="inner")

    df["ref_report"] = df["ref_report"].astype(str).map(norm_ws)
    df["gen_report"] = df["gen_report"].astype(str).map(norm_ws)
    return df

def load_st_model(model_name: str, device: str):
    """
    SentenceTransformer로 로딩을 시도하고, 실패하면 HF Transformer + MeanPooling으로 fallback.
    → PubMedBERT 같은 일반 HF 모델도 문장 임베딩 가능.
    """
    from sentence_transformers import SentenceTransformer, models
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception:
        # HF backbone + mean pooling
        we = models.Transformer(model_name)
        pool = models.Pooling(we.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        model = SentenceTransformer(modules=[we, pool])
        model.to(device)
        return model

def embed_sbert(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None, batch_size=64):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_st_model(model_name, device=device)
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"[Embed] {model_name.split('/')[-1]}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(
            batch,
            batch_size=len(batch),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embs.append(emb)
    return np.vstack(embs)

def compute_similarity(df: pd.DataFrame, model_name: str) -> np.ndarray:
    ref_emb = embed_sbert(df["ref_report"].tolist(), model_name=model_name)
    gen_emb = embed_sbert(df["gen_report"].tolist(), model_name=model_name)
    sims = (ref_emb * gen_emb).sum(axis=1)  # normalized → cosine
    return sims

# --- 기존 save_histogram 대체 ---
def save_histogram_styled(values: np.ndarray, png_path: str, title: str):
    Path(os.path.dirname(png_path) or ".").mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlim(0.0, 1.0)                               # x축 범위 고정
    plt.xticks([i/10 for i in range(0, 11)])         # 0.0 ~ 1.0
    plt.yticks(range(0, 35, 5))                      # 0,5,...,30
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Reports")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_csv", required=True, help="CSV with generated reports (must have file_name + report)")
    ap.add_argument("--ref_csv", required=True, help="Reference CSV (must have file_name + report)")
    ap.add_argument("--out_csv", default="outputs/report_similarity.csv")
    ap.add_argument("--hist_png", default="outputs/similarity_hist.png")
    ap.add_argument("--gen_col", default="report")
    ap.add_argument("--ref_col", default="report")
    ap.add_argument("--bins", type=int, default=40)
    # 추가: PubMedBERT 설정
    ap.add_argument("--pubmed_model", default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    ap.add_argument("--pub_hist_png", default="outputs/similarity_pub.png")
    # 기본 SBERT 모델
    ap.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    df = load_and_merge(args.gen_csv, args.ref_csv, args.gen_col, args.ref_col)
    if len(df) == 0:
        raise ValueError("No overlapping file_name rows found between the two CSVs.")

    # 1) SBERT
    sims_sbert = compute_similarity(df, model_name=args.sbert_model)
    print("[Summary] SBERT cosine similarity")
    print(f"  N = {len(sims_sbert)}")
    print(f"  mean = {np.mean(sims_sbert):.4f}")
    print(f"  median = {np.median(sims_sbert):.4f}")
    print(f"  p10 = {np.percentile(sims_sbert,10):.4f} | p25 = {np.percentile(sims_sbert,25):.4f} | "
      f"p75 = {np.percentile(sims_sbert,75):.4f} | p90 = {np.percentile(sims_sbert,90):.4f}")
    print(pd.Series(sims_sbert, name="SBERT").describe())
    save_histogram_styled(sims_sbert, args.hist_png, title="Distribution of SBERT Cosine Similarity")

    # 2) PubMedBERT
    sims_pub = compute_similarity(df, model_name=args.pubmed_model)
    print("[Summary] PubMedBERT cosine similarity (mean-pooled)")
    print(f"  mean = {np.mean(sims_pub):.4f} | median = {np.median(sims_pub):.4f}")
    print(pd.Series(sims_pub, name="PubMedBERT").describe())
    save_histogram_styled(sims_pub, args.pub_hist_png, title="Distribution of PubMedBERT Cosine Similarity")

    # CSV (두 모델 점수 함께 저장)
    out = df.copy()
    out["sbert_cosine"] = sims_sbert
    out["sbert_cosine_pub"] = sims_pub
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] Saved per-row similarity CSV: {out_csv}")
    print(f"[OK] Saved histograms: {args.hist_png}, {args.pub_hist_png}")


if __name__ == "__main__":
    main()
