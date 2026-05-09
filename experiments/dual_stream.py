import hydra
import torch
import torch.nn.functional as F
from loguru import logger
import wandb
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from nbr.models.dual_stream import DualStreamNBR
from nbr.train.data_module import BasketDataModule
from nbr.utils.seed import seed_everything
from nbr.utils.device import get_device
from nbr.metrics.ranking import recall_at_k, ndcg_at_k

PROJECT_ROOT = Path(__file__).parent.parent

@hydra.main(version_base=None, config_path="../configs/train", config_name="dual_stream_cfg")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    wandb.init(project=cfg.wandb.project, name="dual_stream_run", config=OmegaConf.to_container(cfg, resolve=True))

    # 1. Data Setup
    processed_dir = str(PROJECT_ROOT / "data" / "processed" / cfg.dataset.name)
    datamodule = BasketDataModule(
        processed_dir=processed_dir,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        num_workers=cfg.num_workers
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    vocab_size = datamodule.vocab_size

    # 2. Model Setup
    model = DualStreamNBR(
        vocab_size=vocab_size + 2,
        dim=cfg.model.dim,
        num_heads=cfg.model.num_heads,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_gpt_layers=cfg.model.num_gpt_layers,
        dropout=cfg.model.dropout
    ).to(device)

    # 3. Load Pre-trained Weights (Skip Phase 1 & 2)
    processed_dir = PROJECT_ROOT / "data" / "processed" / cfg.dataset.name
    
    # - Word2Vec
    kv_path = processed_dir / f"word2vec_dim{cfg.model.dim}.kv"
    if kv_path.exists():
        logger.info(f"Loading Word2Vec embeddings from {kv_path}")
        model.item_embedding.from_word2vec(str(kv_path), vocab_size, cfg.model.dim)
        
    # - BERT Bundle (IntraBasketEncoder + MLM)
    # ------------------------------------------------------------------ #
# Load Phase 1 — pre-trained BERT encoder + MLM head                  #
# ------------------------------------------------------------------ #
    bert_path = PROJECT_ROOT / "data/processed" / cfg.dataset.name / f"bert_encoder_bundle_{cfg.dataset.name}.pt"
    if bert_path.exists():
        logger.info(f"Loading pre-trained BERT bundle from {bert_path}")
        checkpoint = torch.load(bert_path, map_location=device)
        sd = checkpoint["state_dict"]

        model.encoder.load_state_dict(sd["encoder"])

        saved_vocab = sd["embedding.weight"].shape[0]
        model_vocab = model.item_embedding.embedding.weight.shape[0]
        if saved_vocab == model_vocab:
            model.item_embedding.embedding.weight.data.copy_(sd["embedding.weight"])
            logger.info("Loaded encoder + item embeddings from BERT bundle.")
        else:
            logger.warning(
                f"Vocab size mismatch: bundle has {saved_vocab}, model has {model_vocab}. "
                f"Bundle metadata: num_items={checkpoint['num_items']}, "
                f"pad_token_id={checkpoint['pad_token_id']}, "
                f"mask_token_id={checkpoint['mask_token_id']}. "
                f"Skipping embedding copy — check your vocab_size / datamodule config."
            )
    else:
        logger.warning(f"BERT bundle not found at {bert_path}. Random init used.")
    
    # - Importance Head
    head_path = PROJECT_ROOT / "data/processed" / cfg.dataset.name /"importance_head"/ "importance_head_best.pt"
    if head_path.exists():
        logger.info("Loading pre-trained Importance Head...")
        checkpoint = torch.load(head_path, map_location=device)
        model.importance_head.load_state_dict(checkpoint["head_state_dict"])
    else:
        logger.warning(f"Importance head not found at {head_path}. Random init used.")

    # 4. Joint Training Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    
    # Tracking for analysis
    gate_stats = []

    logger.info("Starting Dual-Stream Joint Training (Standard BCE)...")
    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            items, item_mask, basket_mask, basket_mask_target, targets = (
                batch["items"].to(device),
                batch["item_mask"].to(device),
                batch["basket_mask"].to(device),
                batch["basket_mask_target"].to(device),
                batch["targets"].to(device), # Expected multi-hot (B, T, V)
            )
            if targets.shape[-1] < vocab_size:
                targets = F.pad(targets, (0, vocab_size - targets.shape[-1]), value=0)
            
            optimizer.zero_grad()
            outputs = model(items, item_mask, basket_mask)
            
            # Phase 3: Standard BCE next-item prediction loss
            logits = outputs["logits"] # (B, T, V)
            
            # Mask out padding baskets in the sequence
            active_baskets = basket_mask_target.flatten()
            flat_logits = logits.view(-1, logits.shape[-1])[active_baskets]
            flat_targets = targets.view(-1, targets.shape[-1])[active_baskets]

            if flat_targets.shape[-1] < flat_logits.shape[-1]:
                flat_targets = F.pad(flat_targets, (0, flat_logits.shape[-1] - flat_targets.shape[-1]), value=0)

            loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets.float())
                        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_norm)
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        wandb.log({"train/bce_loss": avg_loss, "epoch": epoch})
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if (epoch + 1) % cfg.trainer.eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_recalls, val_ndcgs, val_hit1 = [], [], []
            with torch.no_grad():
                for batch in val_loader:
                    items, item_mask, basket_mask, basket_mask_target, targets = (
                        batch["items"].to(device),
                        batch["item_mask"].to(device),
                        batch["basket_mask"].to(device),
                        batch["basket_mask_target"].to(device),
                        batch["targets"].to(device),
                    )

                    outputs = model(items, item_mask, basket_mask)
                    logits = outputs["logits"]

                    active_baskets = basket_mask_target.flatten()
                    flat_logits = logits.view(-1, logits.shape[-1])[active_baskets]
                    flat_targets = targets.view(-1, targets.shape[-1])[active_baskets]

                    if flat_targets.shape[-1] < flat_logits.shape[-1]:
                        flat_targets = F.pad(
                            flat_targets,
                            (0, flat_logits.shape[-1] - flat_targets.shape[-1]),
                            value=0,
                        )

                    batch_loss = F.binary_cross_entropy_with_logits(flat_logits, flat_targets.float())
                    val_loss += batch_loss.item()

                    seq_lengths = basket_mask_target.sum(dim=1).long() - 1
                    last_logits = logits[torch.arange(len(items)), seq_lengths, :]
                    target_last = targets[torch.arange(len(items)), seq_lengths, :]

                    val_recalls.append(recall_at_k(last_logits, target_last, k=10).mean().item())
                    val_ndcgs.append(ndcg_at_k(last_logits, target_last, k=10).mean().item())
                    top1 = torch.topk(last_logits, k=1, dim=-1).indices
                    hits1 = torch.gather(target_last, 1, top1).sum(dim=-1)
                    val_hit1.append((hits1 > 0).float().mean().item())

            val_loss = val_loss / max(len(val_loader), 1)
            wandb.log(
                {
                    "val/bce_loss": val_loss,
                    "val/recall@10": sum(val_recalls) / max(len(val_recalls), 1),
                    "val/ndcg@10": sum(val_ndcgs) / max(len(val_ndcgs), 1),
                    "val/hit@1": sum(val_hit1) / max(len(val_hit1), 1),
                    "epoch": epoch,
                }
            )
            logger.info(
                f"Epoch {epoch} | Val loss: {val_loss:.4f} | "
                f"Recall@10: {sum(val_recalls)/max(len(val_recalls),1):.4f} | "
                f"NDCG@10: {sum(val_ndcgs)/max(len(val_ndcgs),1):.4f} | "
                f"Hit@1: {sum(val_hit1)/max(len(val_hit1),1):.4f}"
            )
            model.train()

    # 5. Validation and Gate Analysis
    model.eval()
    all_recalls, all_ndcgs = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            items, item_mask, basket_mask, basket_mask_target = (
                batch["items"].to(device),
                batch["item_mask"].to(device),
                batch["basket_mask"].to(device),
                batch["basket_mask_target"].to(device),
            )
            targets = batch["targets"].to(device)
            
            outputs = model(items, item_mask, basket_mask)
            
            # Extract last valid basket step for predictions
            seq_lengths = basket_mask_target.sum(dim=1).long() - 1
            last_logits = outputs["logits"][torch.arange(len(items)), seq_lengths, :] # (B, V)
            
            # Gate analysis
            # gate_values shape: (B, T). Average over active baskets for this subset
            gates = outputs["gate_values"]
            for b_idx in range(len(items)):
                b_gates = gates[b_idx, :seq_lengths[b_idx]+1]
                gate_stats.extend(b_gates.cpu().tolist())
            
            # Metrics
            target_last = targets[torch.arange(len(items)), seq_lengths, :]
            
            # Compute for the whole batch at once and average
            batch_recall = recall_at_k(last_logits, target_last, k=10).mean().item()
            batch_ndcg = ndcg_at_k(last_logits, target_last, k=10).mean().item()
            
            all_recalls.append(batch_recall)
            all_ndcgs.append(batch_ndcg)
            
    final_recall = sum(all_recalls) / len(all_recalls)
    final_ndcg = sum(all_ndcgs) / len(all_ndcgs)
    wandb.log({"val/recall@10": final_recall, "val/ndcg@10": final_ndcg})
    
    # 6. Detailed Analysis Printout
    logger.info("=== Final Evaluation Metrics ===")
    logger.info(f"Recall@10: : {final_recall:.4f}")
    logger.info(f"NDCG@10   : {final_ndcg:.4f}")
    
    logger.info("\n=== Dual-Stream Gate Component Analysis ===")
    avg_g = sum(gate_stats) / len(gate_stats)
    logger.info(f"Average Gate Value (g) across valid training baskets: {avg_g:.4f}")
    logger.info("Interpretation:")
    logger.info("- Gate 'g' bounds: [0.0, 1.0]")
    logger.info("- fused_repr = g * [CLS (Full Basket)] + (1 - g) * [Importance-Weighted Basket Core]")
    if avg_g > 0.5:
        logger.info(f"-> The model leans heavily ({avg_g:.1%}) towards the FULL basket context (BERT [CLS]).")
    else:
        logger.info(f"-> The model leans heavily ({1.0 - avg_g:.1%}) towards the STRICT importance-weighted items.")

     # 7. Save Model Checkpoint
    save_dir = PROJECT_ROOT / "data" / "processed" / cfg.dataset.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "dual_stream_best.pt"
    logger.info(f"Saving final model weights to {save_path}")
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()
