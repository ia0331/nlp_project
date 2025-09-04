# src/rag/index.py
import argparse, os, pandas as pd, torch
from tqdm import tqdm
from transformers import AutoTokenizer
import chromadb
import uuid

# 네가 정의한 인코더
from src.models.encoders import GLoRIAEncoder

@torch.no_grad()
def encode_texts_with_gloria(encoder: GLoRIAEncoder, tokenizer, texts, device="cpu", max_len=128, batch_size=128):
    """
    GLoRIA 텍스트 경로: txt -> (g_txt, tok) -> tg_head(g_txt) => [B, D], 이미 normalize됨
    """
    embs = []
    encoder.eval().to(device)
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}

        # Text only: encoder.txt -> g_txt, tok_hidden
        g_txt, _tok_hidden = encoder.txt(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        # Projection to shared space (normalized inside)
        tg = encoder.tg_head(g_txt)   # [B, D], F.normalize() 적용됨

        embs.append(tg.detach().cpu())
    return torch.cat(embs, dim=0).contiguous()  # [N, D]

def load_gloria_encoder(ckpt_path: str, text_model_name: str, proj_dim: int = 256) -> GLoRIAEncoder:
    """
    네가 학습한 구조 그대로 초기화한 뒤, rep_best.pt를 유연하게 로드.
    """
    enc = GLoRIAEncoder(proj_dim=proj_dim, text_model=text_model_name)
    state = torch.load(ckpt_path, map_location="cpu")

    # 다양한 저장 포맷 대응
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            sd = state["model"]
        else:
            sd = state
    else:
        sd = state

    # DataParallel('module.') prefix 제거 대응
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "")
        new_sd[nk] = v

    missing, unexpected = enc.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)} (예: {missing[:5]})")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)} (예: {unexpected[:5]})")
    return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)                     # ex) data/train.csv
    ap.add_argument("--text_col", default="report")      # 문장 소스 컬럼
    ap.add_argument("--ckpt", required=True)                    # rep_best.pt 경로
    ap.add_argument("--tokenizer_name", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--persist_dir", required=True)             # ex) experiments/chroma_store
    ap.add_argument("--collection", default="cxr_findings")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--proj_dim", type=int, default=256)        # 네 학습 설정과 반드시 일치
    a = ap.parse_args()

    os.makedirs(a.persist_dir, exist_ok=True)

    # 1) 텍스트 로드/문장 분할
    df = pd.read_csv(a.csv)
    sentences, metas = [], []
    for i, t in enumerate(df[a.text_col].fillna("")):
        for s in str(t).split("."):
            s = s.strip()
            if len(s) > 5:
                sentences.append(s)
                metas.append({"row_id": i})
    if not sentences:
        print("No sentences extracted. Check --text_col and CSV.")
        return

    # 2) 토크나이저 & GLoRIA 로드
    tok = AutoTokenizer.from_pretrained(a.tokenizer_name)
    encoder = load_gloria_encoder(a.ckpt, text_model_name=a.tokenizer_name, proj_dim=a.proj_dim)

    # 3) 임베딩
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embs = encode_texts_with_gloria(
        encoder, tok, sentences, device=device,
        max_len=a.max_len, batch_size=a.batch_size
    )  # [N, D], float32, normalized

    # 4) ChromaDB 영속 컬렉션에 upsert
    client = chromadb.PersistentClient(path=a.persist_dir)
    col = client.get_or_create_collection(
        name=a.collection,
        metadata={"hnsw:space": "cosine"}   # cosine 일관성
)

    # 안전한 고정 배치 크기(여유 있게)
    ADD_BATCH = 2000

    # 재실행 안전: uuid 기반 ID (기존 count 기반은 충돌 소지)
    all_ids = [f"sent_{uuid.uuid4().hex}" for _ in range(len(sentences))]

    docs = sentences
    metas_list = metas
    vecs = embs.numpy().tolist()  # [[D], [D], ...]

    assert len(docs) == len(metas_list) == len(vecs) == len(all_ids), "length mismatch"

    for i in tqdm(range(0, len(docs), ADD_BATCH), desc="Chroma add(batched)"):
        j = i + ADD_BATCH
        col.add(
            ids=all_ids[i:j],
            documents=docs[i:j],
            embeddings=vecs[i:j],
            metadatas=metas_list[i:j],
        )

    print(f"✅ Indexed {len(docs)} sentences → collection='{a.collection}', total={col.count()}, dim={embs.shape[1]}")

if __name__ == "__main__":
    main()
    