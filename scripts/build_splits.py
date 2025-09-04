import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with image_path,label,findings_text")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2025)
    return ap.parse_args()

def main():
    a = parse()
    df = pd.read_csv(a.manifest)
    train_df, temp = train_test_split(df, test_size=a.val_ratio + a.test_ratio,
                                      stratify=df["label"], random_state=a.seed)
    rel_test = a.test_ratio / (a.val_ratio + a.test_ratio)
    val_df, test_df = train_test_split(temp, test_size=rel_test,
                                       stratify=temp["label"], random_state=a.seed)

    os.makedirs(a.out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(a.out_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(a.out_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(a.out_dir, "test.csv"), index=False)
    print("Saved:", a.out_dir, len(train_df), len(val_df), len(test_df))

if __name__ == "__main__":
    main()


#실행 명령:
#  python scripts/build_splits.py \
#   --manifest data/merged.csv \
#   --out_dir data \
#   --val_ratio 0.15 --test_ratio 0.15 --seed 2025
