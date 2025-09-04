import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

class PairedCXRDataset(Dataset):
    def __init__(self, csv_path, image_size=512, max_report_tokens=256, tokenizer=None):
        self.df = pd.read_csv(csv_path)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.max_report_tokens = max_report_tokens

        self.tx = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        
        if "label" in self.df.columns:
            def _to_int_label(v):
                if pd.isna(v):
                    return -1
                # 이미 숫자일 수 있음
                if isinstance(v, (int, np.integer)):
                    return int(v)
                # 문자열/기타 → 정규화
                s = str(v).strip().lower()
                if s in {"normal"}:
                    return 0
                if s in {"pneumonia"}:
                    return 1
                return -1  # 알 수 없는 값은 -1 처리
            self.df["label"] = self.df["label"].apply(_to_int_label)
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["file_name"]).convert("RGB")
        img = self.tx(img)

        findings = str(row["report"]) if pd.notna(row["report"]) else ""
        if self.tokenizer is not None:
            tok = self.tokenizer(findings, padding="max_length", truncation=True,
                                 max_length=self.max_report_tokens, return_tensors="pt")
            tok = {k: v.squeeze(0) for k, v in tok.items()}
        else:
            tok = None

        label = int(row["label"]) if "label" in row else -1
        return {"image": img, "text": tok, "label": label, "raw_text": findings}
