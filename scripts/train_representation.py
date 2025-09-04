import argparse, json, os, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import math
import sys, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # HF 경고 억제
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.paired_dataset import PairedCXRDataset
from src.models.encoders import GLoRIAEncoder
from src.losses.gloria_losses import contrastive_global, contrastive_local
import torch.nn.functional as F

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--cfg", required=True)
    return ap.parse_args()


def freeze_backbone_and_enable_projections(model: GLoRIAEncoder):
    """
    - 백본(ResNet, BioClinicalBERT)은 freeze + eval
    - projection head들(vg_head, vl_head, tg_head, tl_head)만 학습
    """
    # 0) 전체 우선 freeze
    for p in model.parameters():
        p.requires_grad = False

    # 1) 이미지 백본 freeze + eval
    #    model.im = ImageEncoderResNet50(stem, layer1~4, avgpool)
    for name in ["stem", "layer1", "layer2", "layer3", "layer4", "avgpool"]:
        m = getattr(model.im, name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad = False
            # BN 러닝스탯 고정
            try:
                m.eval()
            except Exception:
                pass

    # 2) 텍스트 백본 freeze + eval
    #    model.txt = TextEncoderBioclinicalBERT(model=AutoModel)
    if hasattr(model.txt, "model") and isinstance(model.txt.model, nn.Module):
        for p in model.txt.model.parameters():
            p.requires_grad = False
        model.txt.model.eval()

    # 3) 프로젝션 헤드만 학습 허용
    #    model.vg_head, model.vl_head, model.tg_head, model.tl_head = ProjectionHead(...)
    for head_name in ["vg_head", "vl_head", "tg_head", "tl_head"]:
        head = getattr(model, head_name, None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True

    # 4) (선택) learnable temperature/logit_scale가 있다면 풀기
    for attr in ["temp_g", "logit_scale", "logit_scale_g"]:
        if hasattr(model, attr):
            obj = getattr(model, attr)
            if isinstance(obj, nn.Parameter):
                obj.requires_grad = True
            elif hasattr(obj, "parameters"):
                for p in obj.parameters():
                    p.requires_grad = True

    # 5) 디버그 출력: 실제 학습 파라미터 목록
    total = 0
    print("=== Trainable parameters ===")
    for n, p in model.named_parameters():
        if p.requires_grad:
            print("  ", n, tuple(p.shape))
            total += p.numel()
    print("Total trainable params:", total)


def build_scheduler(optimizer, total_steps, warmup_ratio=0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        # cosine decay to 0
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

def evaluate(model, data_loader, device, cfg, alpha_loc=1.0):
    model.eval()
    total_loss_g = 0.0
    total_loss_l = 0.0
    n_batches = 0

    # Retrieval 통계
    it_r1 = it_r5 = ti_r1 = ti_r5 = 0
    it_total = ti_total = 0

    for b in data_loader:
        img = b["image"].to(device)
        t   = b["text"]
        ids = t["input_ids"].to(device)
        att = t["attention_mask"].to(device)

        vg, vl, tg, tl = model(img, ids, att)

        # --- Loss (학습과 동일) ---
        loss_g = contrastive_global(vg, tg, cfg["tau_global"])
        loss_l = contrastive_local(vl, tl, cfg["tau_local"], cfg["tau_match"], attn_mask=att)
        total_loss_g += loss_g.item()
        total_loss_l += loss_l.item()
        n_batches += 1

        # --- Retrieval (global embedding 사용) ---
        # 보수적으로 cosine 유사도 사용
        vg_n = F.normalize(vg, dim=1)
        tg_n = F.normalize(tg, dim=1)
        sim  = vg_n @ tg_n.t()            # [B, B]

        # image->text
        it_total += sim.size(0)
        it_rank = sim.argsort(dim=1, descending=True)  # 각 이미지 기준 텍스트 랭킹
        gt = torch.arange(sim.size(0), device=sim.device)
        it_r1 += (it_rank[:, :1] == gt.unsqueeze(1)).any(dim=1).sum().item()
        it_r5 += (it_rank[:, :5] == gt.unsqueeze(1)).any(dim=1).sum().item()

        # text->image
        ti_total += sim.size(0)
        ti_rank = sim.t().argsort(dim=1, descending=True)  # 각 텍스트 기준 이미지 랭킹
        ti_r1 += (ti_rank[:, :1] == gt.unsqueeze(1)).any(dim=1).sum().item()
        ti_r5 += (ti_rank[:, :5] == gt.unsqueeze(1)).any(dim=1).sum().item()

    avg_g = total_loss_g / max(1, n_batches)
    avg_l = total_loss_l / max(1, n_batches)
    total = avg_g + alpha_loc * avg_l

    # R@1/5 (양방향과 평균)
    it_r1 = it_r1 / max(1, it_total)
    it_r5 = it_r5 / max(1, it_total)
    ti_r1 = ti_r1 / max(1, ti_total)
    ti_r5 = ti_r5 / max(1, ti_total)

    metrics = {
        "loss_total": total,
        "loss_global": avg_g,
        "loss_local": avg_l,
        "R1_image_to_text": it_r1,
        "R5_image_to_text": it_r5,
        "R1_text_to_image": ti_r1,
        "R5_text_to_image": ti_r5,
        "R1_mean": 0.5 * (it_r1 + ti_r1),
        "R5_mean": 0.5 * (it_r5 + ti_r5),
    }
    return metrics

def main():
    args = parse()
    cfg = json.load(open(args.cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Early Stopping (기존 로직 유지)
    patience   = cfg.get("early_stopping_patience", 5)
    min_delta  = cfg.get("early_stopping_min_delta", 0.0)
    save_path  = cfg["save_path"]
    alpha_loc  = cfg.get("alpha_local", 1.0)  # 로컬 손실 가중치 (기본=1.0)
    warmup_ratio = cfg.get("warmup_ratio", 0.05)

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_model"])

    pin_mem = bool(torch.cuda.is_available())
    train_ds = PairedCXRDataset(args.train_csv, cfg["image_size"], cfg["max_report_tokens"], tokenizer)
    val_ds   = PairedCXRDataset(args.val_csv,   cfg["image_size"], cfg["max_report_tokens"], tokenizer)
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=4, pin_memory=pin_mem)
    val_ld   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                          num_workers=2, pin_memory=pin_mem)

    model = GLoRIAEncoder(proj_dim=cfg["proj_dim"], text_model=cfg["text_model"]).to(device)

    # ★ 백본 freeze + eval, 프로젝션만 학습
    freeze_backbone_and_enable_projections(model)

    # 학습되는 파라미터만 Optimizer에 등록
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check projection layers naming.")
    base_lr = cfg.get("lr", 5e-5)  # 과대업데이트 방지: 1e-4 -> 5e-5 권장
    opt = torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=cfg["weight_decay"])

    # ★ 스케줄러: 워밍업 + 코사인
    total_steps = len(train_ld) * cfg["epochs"]
    scheduler = build_scheduler(opt, total_steps, warmup_ratio=warmup_ratio)

    best_val = float("inf")
    no_improve = 0
    global_step = 0

    for epoch in range(cfg["epochs"]):
        # -------- Train --------
        model.train()
        # 백본 모듈은 계속 eval 유지 (BN 드리프트 방지)
        if hasattr(model, "im"):
            for name in ["stem", "layer1", "layer2", "layer3", "layer4"]:
                if hasattr(model.im, name):
                    getattr(model.im, name).eval()
        if hasattr(model, "txt") and hasattr(model.txt, "bert"):
            model.txt.bert.eval()

        running = 0.0
        pbar = tqdm(train_ld, desc=f"Epoch {epoch+1}/{cfg['epochs']}", unit="batch")
        for b in pbar:
            img = b["image"].to(device)
            t   = b["text"]
            ids = t["input_ids"].to(device)
            att = t["attention_mask"].to(device)

            vg, vl, tg, tl = model(img, ids, att)
            loss_g = contrastive_global(vg, tg, cfg["tau_global"])
            loss_l = contrastive_local(vl, tl, cfg["tau_local"], cfg["tau_match"], attn_mask=att)
            loss = loss_g + alpha_loc * loss_l  # ★ 로컬 가중치

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("grad_clip", 0) and cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, cfg["grad_clip"])
            opt.step()
            scheduler.step()  # ★ 스케줄러 스텝

            running += float(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "g": f"{loss_g.item():.4f}",
                "l": f"{loss_l.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            global_step += 1

        train_loss = running / max(1, len(train_ld))

        # -------- Validate: global + local 모니터링 --------
        model.eval()
        with torch.no_grad():
            vloss_g = vloss_l = 0.0
            n = 0
            for b in val_ld:
                img = b["image"].to(device)
                t   = b["text"]
                ids = t["input_ids"].to(device)
                att = t["attention_mask"].to(device)
                vg, vl, tg, tl = model(img, ids, att)
                vloss_g += contrastive_global(vg, tg, cfg["tau_global"]).item()
                vloss_l += contrastive_local(vl, tl, cfg["tau_local"], cfg["tau_match"], attn_mask=att).item()
                n += 1
            vloss_g /= max(1, n)
            vloss_l /= max(1, n)
            vloss = vloss_g + alpha_loc * vloss_l

        print(f"[Epoch {epoch+1}] train={train_loss:.4f}  val_g={vloss_g:.4f}  val_l={vloss_l:.4f}  "
              f"val_total={vloss:.4f}  best={best_val:.4f}")

        # -------- Checkpoint & Early Stopping (기존 로직) --------
        improved = (best_val - vloss) > min_delta
        if improved:
            best_val = vloss
            no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, save_path)
            print(f"✅ Improved. Saved: {save_path}")
        else:
            no_improve += 1
            print(f"no_improve={no_improve}/{patience}")
            if no_improve >= patience:
                print("⏹ Early stopping triggered.")
                break
        
    if args.test_csv:
        print("\n===== Loading best checkpoint and evaluating on TEST =====")
            # 1) 체크포인트 로드
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()

            # 2) 테스트 데이터로더
        test_ds = PairedCXRDataset(args.test_csv, cfg["image_size"], cfg["max_report_tokens"], tokenizer)
        test_ld = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=pin_mem)

        # 3) 평가 실행
        test_metrics = evaluate(
            model, test_ld, device, cfg, alpha_loc=alpha_loc
        )

            # 4) 로그 출력
        print(
            "[TEST] loss_total={:.4f}  loss_g={:.4f}  loss_l={:.4f}  "
            "R@1(i→t)={:.3f}  R@1(t→i)={:.3f}  R@1(mean)={:.3f}  "
            "R@5(i→t)={:.3f}  R@5(t→i)={:.3f}  R@5(mean)={:.3f}".format(
                test_metrics["loss_total"],
                test_metrics["loss_global"],
                test_metrics["loss_local"],
                test_metrics["R1_image_to_text"],
                test_metrics["R1_text_to_image"],
                test_metrics["R1_mean"],
                test_metrics["R5_image_to_text"],
                test_metrics["R5_text_to_image"],
                test_metrics["R5_mean"],
            )
        )        

if __name__ == "__main__":
    main()
