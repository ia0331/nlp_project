# rag/retrieve_gloria_chroma.py
import chromadb, torch
from typing import List, Tuple
from transformers import AutoTokenizer
from src.models.encoders import GLoRIAEncoder

@torch.no_grad()
def embed_query_with_gloria(encoder: GLoRIAEncoder, tokzr, text, device="cpu", max_len=128):
    t = tokzr([text], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    t = {k: v.to(device) for k, v in t.items()}
    g_txt, _ = encoder.txt(input_ids=t["input_ids"], attention_mask=t["attention_mask"])
    q = encoder.tg_head(g_txt)  # [1, D], normalized
    return q.cpu().numpy().tolist()

def load_gloria_encoder(ckpt_path: str, tokenizer_name: str, proj_dim=256):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = GLoRIAEncoder(proj_dim=proj_dim, text_model=tokenizer_name)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = state.get("model", state)
    new_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    enc.load_state_dict(new_sd, strict=False)
    return enc, tok

def retrieve_topk_from_chroma(persist_dir: str, collection: str, query: str, k: int = 5,
                              ckpt_path="checkpoints/rep_best.pt",
                              tokenizer_name="emilyalsentzer/Bio_ClinicalBERT",
                              proj_dim=256, max_len=128,
                              where: dict | None = None) -> List[Tuple[str, dict, str]]:
    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_collection(collection)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, tok = load_gloria_encoder(ckpt_path, tokenizer_name, proj_dim)
    encoder.eval().to(device)

    q = embed_query_with_gloria(encoder, tok, query, device=device, max_len=max_len)
    res = col.query(query_embeddings=q, n_results=k, where=where or None)
    # 반환: (문장, 메타데이터, id)
    return list(zip(res["documents"][0], res["metadatas"][0], res["ids"][0]))
