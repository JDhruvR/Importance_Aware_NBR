"""
full_train.py — Intent-Aware NBR: Phase 3 Training & Evaluation
================================================================
Phases 0, 1, and 2 have already been run by separate scripts and their
outputs saved to disk:

    Phase 0 — word2vec_dim{d}.kv          (item embedding init)
    Phase 1 — bert_encoder_bundle.pt      (BERT encoder + MLM head)
    Phase 2 — importance_head_best.pt     (importance head pre-trained to alpha_IDF)
              importance_scores.npy       (precomputed alpha_IDF scores)

This script loads those checkpoints and runs:

    Phase 3 — Joint training with full loss L (Eq. 25):
                  L = L_intent + lambda*L_fill + gamma*L_orth + eta*L_MLM
              Gram-Schmidt re-orthonormalization of projection P every N steps.

    Inference (Section VI Steps 1-6) — two-stage residual decode:
        Step 1  Encode history: BERT + importance + fusion -> basket vectors b_j
        Step 2  GPT forward -> predicted next-basket representation h_{T+1}
        Step 3  Decompose h_{T+1} into h^intent and h^fill via projection P
        Step 4  Core item residual loop in intent subspace (K1 items)
        Step 5  Condition fill query on discovered core (hard mean embedding)
        Step 6  Fill item residual loop in fill subspace (K2 items)
"""

import hydra
import torch
from loguru import logger
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from nbr.models.full_model import IntentAwareNBR
from nbr.models.decoder import residual_decode
from nbr.losses import total_loss
from nbr.utils.seed import seed_everything
from nbr.utils.device import get_device
from nbr.metrics.ranking import recall_at_k, ndcg_at_k, repeat_explore_masks, build_history_multihot, mrr_at_k
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent


def _log_partition_stats(alpha_idf: torch.Tensor, tau_alpha: float) -> None:
    """Log alpha_idf partition diagnostics before training."""
    core_frac = (alpha_idf > tau_alpha).float().mean().item()
    logger.info(f"Partition diagnostics: tau_alpha={tau_alpha}")
    logger.info(f"  Items with alpha_idf > tau: {core_frac:.2%} (core), {1-core_frac:.2%} (fill)")
    logger.info(f"  alpha_idf stats: mean={alpha_idf.mean():.4f}, std={alpha_idf.std():.4f}, "
                f"min={alpha_idf.min():.4f}, max={alpha_idf.max():.4f}")
    wandb.run.summary["partition/core_fraction"] = core_frac
    wandb.run.summary["partition/alpha_mean"] = alpha_idf.mean().item()
    wandb.run.summary["partition/alpha_std"] = alpha_idf.std().item()


# ============================================================================ #
#  Helper: build MLM targets from original items + mask                        #
# ============================================================================ #

def _build_mlm_targets(
    original_items: torch.Tensor,
    mlm_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """
    Build mlm_targets for mlm_loss() from the data pipeline's masking output.

    The data pipeline replaces some item positions in `items` with a [MASK]
    token before passing to the model. `mlm_mask` records which positions were
    replaced; `original_items` holds the true ids at those positions.

    At masked positions  (mlm_mask == 1): target = original item id
    At all other positions               : target = mask_token_id  (ignored by
                                           mlm_loss() via ignore_index)

    Args:
        original_items : (B, T, S) unmasked item ids from data pipeline
        mlm_mask       : (B, T, S) 1 at masked positions, 0 elsewhere
        mask_token_id  : int       value used as ignore_index in cross-entropy

    Returns:
        mlm_targets : (B, T, S)
    """
    mlm_targets = torch.full_like(original_items, fill_value=mask_token_id)
    mlm_targets[mlm_mask.bool()] = original_items[mlm_mask.bool()]
    return mlm_targets


# ============================================================================ #
#  Phase 3 — Joint training with full loss L (Section V-I, Eq. 25)            #
# ============================================================================ #

def phase3_joint_training(
    model: IntentAwareNBR,
    train_loader,
    val_loader,
    alpha_idf: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    save_dir: Path,
) -> None:
    """
    All components unfrozen. Trains with the full loss from Eq. 25:

        L = L_intent + lambda*L_fill + gamma*L_orth + eta*L_MLM

    Includes:
    - Validation every `eval_every` epochs via two-stage residual decode
    - Best-model checkpointing based on val/recall@10
    - Full W&B logging of per-step losses, epoch summaries, and val metrics
    """
    logger.info("Phase 3 — joint training with full loss (all components unfrozen)")

    loss_weights = {
        "intent": cfg.loss.weights.intent,
        "fill":   cfg.loss.weights.fill,
        "orth":   cfg.loss.weights.orth,
        "mlm":    cfg.loss.weights.mlm,
    }
    tau_alpha     = cfg.model.tau_alpha
    mask_token_id = cfg.data.mask_token_id
    reorth_every  = cfg.trainer.get("reorth_every", 100)
    mlm_mask_prob = cfg.trainer.get("mlm_mask_prob", 0.15)
    eval_every    = cfg.trainer.get("eval_every", 3)

    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / "full_model_best.pt"
    last_ckpt_path = save_dir / "full_model_last.pt"

    best_recall = -1.0
    best_epoch  = -1
    step_count  = 0
    freeze_epochs = 3  # Freeze backbone for first few epochs to let random Decoder converge

    total_params = sum(p.numel() for p in model.parameters())
    
    # Initially freeze everything except decoder
    logger.info(f"Freezing backbone for the first {freeze_epochs} epochs to prevent catastrophic forgetting.")
    for name, param in model.named_parameters():
        if "decoder" not in name:
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.run.summary["total_params"] = total_params
    wandb.run.summary["trainable_params_phase1"] = trainable_params
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable (Phase 3.1)")

    for epoch in range(cfg.trainer.max_epochs):
        if epoch == freeze_epochs:
            logger.info("Unfreezing backbone for joint fine-tuning (Phase 3.2).")
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.run.summary["trainable_params_phase2"] = trainable_params
            logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

        model.train()
        epoch_loss = 0.0
        epoch_losses = {}
        num_batches = 0

        for batch in train_loader:
            items       = batch["items"].to(device)           # (B, T, S)
            item_mask   = batch["item_mask"].to(device)       # (B, T, S)
            basket_mask = batch["basket_mask"].to(device)     # (B, T)
            targets     = batch["targets"].to(device)         # (B, T, V) binary

            # Pad targets to match model vocab (num_items + 2 special tokens)
            model_vocab = model.mlm_head.out_features
            if targets.shape[-1] < model_vocab:
                pad_size = model_vocab - targets.shape[-1]
                targets = F.pad(targets, (0, pad_size), value=0)

            # --- Inline MLM masking (Eq. 3 auxiliary signal) ---
            if mlm_mask_prob > 0.0:
                original_items = items.clone()
                rand_vals = torch.rand_like(items, dtype=torch.float32)
                mlm_mask = item_mask.bool() & (rand_vals < mlm_mask_prob)
                items = items.clone()
                items[mlm_mask] = mask_token_id
                mlm_targets = _build_mlm_targets(original_items, mlm_mask, mask_token_id)
            else:
                mlm_targets = None

            optimizer.zero_grad()

            out = model(items, item_mask, basket_mask)

            loss, loss_dict = total_loss(
                intent_logits = out["intent_logits"],
                fill_logits   = out["fill_logits"],
                targets       = targets,
                alpha_idf     = alpha_idf,
                tau_alpha     = tau_alpha,
                intent_repr   = out["intent_repr"],
                fill_repr     = out["fill_repr"],
                mlm_logits    = out["mlm_logits"],
                mlm_targets   = mlm_targets,
                weights       = loss_weights,
                mask_token_id = mask_token_id,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.clip_norm)
            optimizer.step()

            step_count += 1
            if step_count % reorth_every == 0:
                model.orthogonalize_projection()

            # Per-step W&B logging
            wandb.log({
                **loss_dict,
                "phase3/step": step_count,
                "phase3/lr": optimizer.param_groups[0]["lr"],
            })

            epoch_loss += loss_dict["loss/total"]
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            num_batches += 1

        # ---- Epoch-level logging ----
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_log = {
            "epoch": epoch + 1,
            "train/epoch_loss": avg_epoch_loss,
        }
        for k, v in epoch_losses.items():
            epoch_log[f"train/{k}_epoch"] = v / max(num_batches, 1)

        logger.info(
            f"  Phase 3 epoch {epoch+1}/{cfg.trainer.max_epochs}  "
            f"loss={avg_epoch_loss:.4f}  "
            f"intent={epoch_losses.get('loss/intent',0)/max(num_batches,1):.4f}  "
            f"fill={epoch_losses.get('loss/fill',0)/max(num_batches,1):.4f}  "
            f"orth={epoch_losses.get('loss/orth',0)/max(num_batches,1):.4f}  "
            f"mlm={epoch_losses.get('loss/mlm',0)/max(num_batches,1):.4f}"
        )

        # ---- Save last checkpoint every epoch ----
        torch.save({
            "epoch": epoch + 1,
            "step": step_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "train_loss": avg_epoch_loss,
        }, last_ckpt_path)

        # ---- Validation every eval_every epochs ----
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == cfg.trainer.max_epochs:
            val_metrics = run_inference(model, val_loader, device, cfg)
            epoch_log.update(val_metrics)

            current_recall = val_metrics.get("val/recall@10", 0.0)

            # Save best checkpoint based on val/recall@10
            if current_recall > best_recall:
                best_recall = current_recall
                best_epoch = epoch + 1
                torch.save({
                    "epoch": best_epoch,
                    "step": step_count,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "val_metrics": val_metrics,
                    "train_loss": avg_epoch_loss,
                }, best_ckpt_path)
                logger.info(
                    f"  ★ New best model saved at epoch {best_epoch} "
                    f"(val/recall@10={best_recall:.4f})"
                )

            model.train()  # switch back after eval

        wandb.log(epoch_log)

    # ---- Final summary ----
    wandb.run.summary["best_epoch"] = best_epoch
    wandb.run.summary["best_val_recall@10"] = best_recall
    wandb.run.summary["best_checkpoint"] = str(best_ckpt_path)
    wandb.run.summary["last_checkpoint"] = str(last_ckpt_path)
    wandb.run.summary["total_steps"] = step_count

    logger.info(
        f"Phase 3 complete — best epoch={best_epoch}, "
        f"best val/recall@10={best_recall:.4f}"
    )


# ============================================================================ #
#  Inference — two paths: residual decode (Section VI) + flat top-K diagnostic #
# ============================================================================ #

@torch.no_grad()
def run_inference(
    model: IntentAwareNBR,
    val_loader,
    device: torch.device,
    cfg: DictConfig,
) -> dict:
    """
    Evaluate using both the paper's residual decode and flat top-K diagnostic.

    Logs metrics to diagnose which head is learning:
      - combined (intent + fill logits)  — what matters for overall performance
      - intent-only                      — is the projection learning core items?
      - fill-only                        — is the fill head learning peripheral items?
      - residual decode                  — the paper's actual 6-step inference

    Also logs the average cosine similarity between intent and fill representations
    as a live orthogonality diagnostic.
    """
    logger.info("Running inference (residual decode + flat top-K diagnostic)...")
    model.eval()

    k1 = cfg.model.k1
    k2 = cfg.model.k2

    # Flat top-K metrics
    combined_r10, combined_n10 = [], []
    combined_r20, combined_n20 = [], []
    intent_r10, fill_r10 = [], []
    orth_sims = []
    
    # Advanced Diagnostics (Explore/Repeat, MRR, Coverage)
    exp_r10, rep_r10 = [], []
    combined_mrr, intent_mrr = [], []
    all_predicted_items = set()
    
    # Residual decode metrics
    res_recalls_10, res_ndcgs_10 = [], []
    res_recalls_20, res_ndcgs_20 = [], []

    for batch in val_loader:
        items       = batch["items"].to(device)
        item_mask   = batch["item_mask"].to(device)
        basket_mask = batch["basket_mask"].to(device)
        bm_target   = batch["basket_mask_target"].to(device)
        targets     = batch["targets"].to(device)

        out = model(items, item_mask, basket_mask)

        # Use the last valid target position per user
        seq_lengths = bm_target.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0)
        B = items.size(0)
        idx = torch.arange(B, device=device)

        h_next = out["next_basket_repr"][idx, seq_lengths, :]    # (B, D)
        vocab_embeddings = model.item_embedding.embedding.weight # (V, D)

        intent_logits = out["intent_logits"][idx, seq_lengths, :]  # (B, V)
        fill_logits   = out["fill_logits"][idx, seq_lengths, :]    # (B, V)
        combined      = intent_logits + fill_logits                # (B, V)
        target_last   = targets[idx, seq_lengths, :]               # (B, V)

        # Pad targets if needed
        if target_last.shape[-1] < combined.shape[-1]:
            target_last = F.pad(target_last, (0, combined.shape[-1] - target_last.shape[-1]))

        # --- Flat Metrics ---
        combined_r10.append(recall_at_k(combined, target_last, 10).mean().item())
        combined_n10.append(ndcg_at_k(combined, target_last, 10).mean().item())
        combined_r20.append(recall_at_k(combined, target_last, 20).mean().item())
        combined_n20.append(ndcg_at_k(combined, target_last, 20).mean().item())

        intent_r10.append(recall_at_k(intent_logits, target_last, 10).mean().item())
        fill_r10.append(recall_at_k(fill_logits, target_last, 10).mean().item())

        # --- Advanced Diagnostics ---
        # MRR
        combined_mrr.append(mrr_at_k(combined, target_last, 10).mean().item())
        intent_mrr.append(mrr_at_k(intent_logits, target_last, 10).mean().item())

        # Coverage
        top10_idx = torch.topk(combined, k=min(10, combined.shape[-1]), dim=-1).indices
        all_predicted_items.update(top10_idx.flatten().tolist())

        # Repeat vs Explore
        model_vocab = vocab_embeddings.shape[0]
        history = build_history_multihot(items, item_mask, model_vocab)
        rep_target, exp_target = repeat_explore_masks(target_last, history)

        rep_r10_batch = recall_at_k(combined, rep_target, 10)
        exp_r10_batch = recall_at_k(combined, exp_target, 10)
        
        valid_rep = (rep_target.sum(dim=-1) > 0)
        valid_exp = (exp_target.sum(dim=-1) > 0)
        
        if valid_rep.any():
            rep_r10.append(rep_r10_batch[valid_rep].mean().item())
        if valid_exp.any():
            exp_r10.append(exp_r10_batch[valid_exp].mean().item())

        # Orthogonality diagnostic
        h_intent = out["intent_repr"][idx, seq_lengths, :]
        h_fill   = out["fill_repr"][idx, seq_lengths, :]
        cos = F.cosine_similarity(h_intent, h_fill, dim=-1).abs().mean().item()
        orth_sims.append(cos)

        # --- Residual decode (Section VI) ---
        for i in range(items.size(0)):
            target = target_last[i]
            predicted_basket = residual_decode(
                repr_vec=h_next[i], item_embeddings=vocab_embeddings,
                decoder=model.decoder, k1=k1, k2=k2, excluded=set(),
            )
            recs_10 = predicted_basket[:10]
            recs_20 = predicted_basket[:20]

            preds_10 = torch.zeros_like(target)
            preds_20 = torch.zeros_like(target)
            
            # Clamp the indices to valid vocab size just in case
            valid_recs_10 = [r for r in recs_10 if 0 <= r < target.size(0)]
            valid_recs_20 = [r for r in recs_20 if 0 <= r < target.size(0)]
            
            if valid_recs_10:
                preds_10[torch.tensor(valid_recs_10, device=target.device)] = 1.0
            if valid_recs_20:
                preds_20[torch.tensor(valid_recs_20, device=target.device)] = 1.0

            res_recalls_10.append(recall_at_k(preds_10.unsqueeze(0), target.unsqueeze(0), 10).item())
            res_ndcgs_10.append(ndcg_at_k(preds_10.unsqueeze(0), target.unsqueeze(0), 10).item())
            res_recalls_20.append(recall_at_k(preds_20.unsqueeze(0), target.unsqueeze(0), 20).item())
            res_ndcgs_20.append(ndcg_at_k(preds_20.unsqueeze(0), target.unsqueeze(0), 20).item())

    n = max(len(combined_r10), 1)
    n_res = max(len(res_recalls_10), 1)
    n_rep = max(len(rep_r10), 1)
    n_exp = max(len(exp_r10), 1)
    avg_orth = sum(orth_sims) / n
    coverage = len(all_predicted_items) / float(vocab_embeddings.shape[0])

    metrics = {
        "val/recall@10":       sum(res_recalls_10) / n_res,
        "val/ndcg@10":         sum(res_ndcgs_10) / n_res,
        "val/recall@20":       sum(res_recalls_20) / n_res,
        "val/ndcg@20":         sum(res_ndcgs_20) / n_res,
        "val/recall@10_flat":     sum(combined_r10) / n,
        "val/ndcg@10_flat":       sum(combined_n10) / n,
        "val/recall@20_flat":     sum(combined_r20) / n,
        "val/ndcg@20_flat":       sum(combined_n20) / n,
        "val/recall@10_intent": sum(intent_r10) / n,
        "val/recall@10_fill":   sum(fill_r10) / n,
        "val/orth_cosine":   avg_orth,
        "val/mrr@10_flat":        sum(combined_mrr) / n,
        "val/mrr@10_intent":      sum(intent_mrr) / n,
        "val/repeat_recall@10":   sum(rep_r10) / n_rep if rep_r10 else 0.0,
        "val/explore_recall@10":  sum(exp_r10) / n_exp if exp_r10 else 0.0,
        "val/coverage@10":        coverage,
    }

    logger.info("=== Inference Metrics ===")
    logger.info(f"  Residual   Recall@10={metrics['val/recall@10']:.4f}  NDCG@10={metrics['val/ndcg@10']:.4f}")
    logger.info(f"  Residual   Recall@20={metrics['val/recall@20']:.4f}  NDCG@20={metrics['val/ndcg@20']:.4f}")
    logger.info(f"  Flat Comb. Recall@10={metrics['val/recall@10_flat']:.4f}  NDCG@10={metrics['val/ndcg@10_flat']:.4f}")
    logger.info(f"  Intent-only R@10={metrics['val/recall@10_intent']:.4f}  |  Fill-only R@10={metrics['val/recall@10_fill']:.4f}")
    logger.info(f"  Orthogonality |cos(h_intent, h_fill)|={avg_orth:.4f}")
    logger.info(f"  [ADV] Explore Recall={metrics['val/explore_recall@10']:.4f}  |  Repeat Recall={metrics['val/repeat_recall@10']:.4f}")
    logger.info(f"  [ADV] Catalog Coverage={metrics['val/coverage@10']:.2%}  |  Intent MRR={metrics['val/mrr@10_intent']:.4f}")

    return metrics


# ============================================================================ #
#  Main                                                                        #
# ============================================================================ #

@hydra.main(version_base=None, config_path="../configs/train", config_name="full_model_cfg")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.get("name", "full_model"),
        config=OmegaConf.to_container(cfg),
    )

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    processed_dir = PROJECT_ROOT / "data" / "processed" / cfg.dataset.name
    datamodule = BasketDataModule(
        processed_dir=processed_dir,
        batch_size=cfg.batch_size,
        max_seq_len=cfg.max_seq_len,
        num_workers=cfg.num_workers
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader   = datamodule.val_dataloader()


    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = IntentAwareNBR(
        vocab_size         = datamodule.vocab_size + 2,  # +2 for pad_token_id=0, mask_token_id=1
        dim                = cfg.model.dim,
        intent_dim         = cfg.model.intent_dim,
        num_heads          = cfg.model.num_heads,
        num_encoder_layers = cfg.model.num_encoder_layers,
        num_gpt_layers     = cfg.model.num_gpt_layers,
        dropout            = cfg.model.get("dropout", 0.1),
        temperature        = cfg.model.get("temperature", 1.0),
    ).to(device)

    # ------------------------------------------------------------------ #
    # Load Phase 0 — word2vec item embeddings                             #
    # ------------------------------------------------------------------ #
    kv_path = processed_dir / f"word2vec_dim{cfg.model.dim}.kv"
    if kv_path.exists():
        logger.info(f"Loading Word2Vec embeddings from {kv_path}")
        model.item_embedding.from_word2vec(str(kv_path), datamodule.vocab_size, cfg.model.dim)
    else:
        logger.warning(f"Word2Vec checkpoint not found at {kv_path}. Random init used.")

    # ------------------------------------------------------------------ #
    # Load Phase 1 — pre-trained BERT encoder + MLM head                  #
    # ------------------------------------------------------------------ #
    bert_path = h = PROJECT_ROOT / "data/processed" / cfg.dataset.name / f"bert_encoder_bundle_{cfg.dataset.name}.pt"
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

    # ------------------------------------------------------------------ #
    # Load dual-stream weights — warm-start GPT + fusion from trained     #
    # dual-stream checkpoint so Phase 3 doesn't start from scratch        #
    # ------------------------------------------------------------------ #
    ds_path = processed_dir / "dual_stream_best.pt"
    if ds_path.exists():
        logger.info(f"Warm-starting from dual-stream checkpoint: {ds_path}")
        ds_sd = torch.load(ds_path, map_location=device)
        loaded, skipped = [], []
        for name, param in model.named_parameters():
            if name in ds_sd and ds_sd[name].shape == param.shape:
                param.data.copy_(ds_sd[name])
                loaded.append(name.split('.')[0])
            else:
                skipped.append(name.split('.')[0])
        loaded_modules = sorted(set(loaded))
        skipped_modules = sorted(set(skipped))
        logger.info(f"  Loaded modules: {loaded_modules}")
        logger.info(f"  Skipped (new/resized): {skipped_modules}")
    else:
        logger.warning(f"Dual-stream checkpoint not found at {ds_path}. GPT/fusion start from random init.")

    # ------------------------------------------------------------------ #
    # Load Phase 2 — pre-trained importance head + alpha_IDF scores       #
    # ------------------------------------------------------------------ #
    head_path = processed_dir / "importance_head" / "importance_head_best.pt"
    if head_path.exists():
        logger.info(f"Loading pre-trained importance head from {head_path}")
        checkpoint = torch.load(head_path, map_location=device)
        model.importance_head.load_state_dict(checkpoint["head_state_dict"])
    else:
        logger.warning(f"Importance head not found at {head_path}. Random init used.")

    # alpha_IDF scores are needed for the Phase 3 loss partition (Eq. 22)
    # and importance-weighted intent BCE (Eq. 23). Must exist before training.
    alpha_idf_path = processed_dir / "importance_scores.npz"
    if not alpha_idf_path.exists():
        raise FileNotFoundError(
            f"alpha_IDF scores not found at {alpha_idf_path}. "
            "Run the Phase 2 script first to compute and save them."
        )
    
    # .npz files contain multiple arrays; we extract the first one
    loaded_npz = np.load(alpha_idf_path)
    array_key = loaded_npz.files[0]
    alpha_idf = torch.from_numpy(loaded_npz[array_key]).float().to(device)

    # Pad alpha_idf to match model vocab (num_items + 2 special tokens)
    model_vocab = datamodule.vocab_size + 2
    if alpha_idf.shape[0] < model_vocab:
        pad_size = model_vocab - alpha_idf.shape[0]
        alpha_idf = F.pad(alpha_idf, (0, pad_size), value=0.0)

    logger.info(f"Loaded alpha_IDF from {alpha_idf_path}  shape={tuple(alpha_idf.shape)}")
    
    # Compute dynamic tau_alpha if requested
    tau_config = str(cfg.model.tau_alpha)
    if tau_config.lower() == "median":
        tau_alpha = alpha_idf.median().item()
        cfg.model.tau_alpha = tau_alpha
    else:
        tau_alpha = float(tau_config)
        
    _log_partition_stats(alpha_idf, tau_alpha)

    # ------------------------------------------------------------------ #
    # Phase 3 — Joint training with full loss L (Eq. 25)                  #
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
    )
    phase3_joint_training(
        model, train_loader, val_loader, alpha_idf, optimizer, device, cfg,
        save_dir=processed_dir,
    )

    # ------------------------------------------------------------------ #
    # Final inference on val set (uses best checkpoint from training)     #
    # ------------------------------------------------------------------ #
    best_ckpt = processed_dir / "full_model_best.pt"
    if best_ckpt.exists():
        logger.info(f"Loading best checkpoint from {best_ckpt}")
        best_state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

    metrics = run_inference(model, val_loader, device, cfg)
    wandb.log({"final/" + k.split("/")[1]: v for k, v in metrics.items()})
    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
