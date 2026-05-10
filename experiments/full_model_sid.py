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
from nbr.train.data_module import BasketDataModule
from nbr.utils.seed import seed_everything
from nbr.utils.device import get_device
from nbr.metrics.ranking import recall_at_k, ndcg_at_k
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent


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
    cooc,
) -> None:
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.run.summary["total_params"] = total_params
    wandb.run.summary["trainable_params"] = trainable_params
    logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    for epoch in range(cfg.trainer.max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {}
        num_batches = 0

        # B4 + B8 — reset every epoch so values don't accumulate across epochs
        gate_vals_epoch, basket_sizes_epoch = [], []

        for batch in train_loader:
            items       = batch["items"].to(device)
            item_mask   = batch["item_mask"].to(device)
            basket_mask = batch["basket_mask"].to(device)
            basket_mask_target = batch["basket_mask_target"].to(device)
            targets     = batch["targets"].to(device)

            model_vocab = model.mlm_head.out_features
            if targets.shape[-1] < model_vocab:
                pad_size = model_vocab - targets.shape[-1]
                targets = F.pad(targets, (0, pad_size), value=0)

            original_items = items.clone()
            rand_vals = torch.rand_like(items, dtype=torch.float32)
            mlm_mask = item_mask.bool() & (rand_vals < mlm_mask_prob)
            items = items.clone()
            items[mlm_mask] = mask_token_id
            mlm_targets = _build_mlm_targets(original_items, mlm_mask, mask_token_id)

            optimizer.zero_grad()

            out = model(items, item_mask, basket_mask)

            # B4 + B8 — collect gate values and basket sizes this batch
            if "gate_values" in out:
                gates = out["gate_values"]            # (B, T)
                sizes = item_mask.sum(dim=-1).float() # (B, T)
                bmask = basket_mask.bool()
                gate_vals_epoch.extend(gates[bmask].tolist())
                basket_sizes_epoch.extend(sizes[bmask].tolist())

            binary_targets = (targets > 0).float()
            mask = basket_mask_target.bool().unsqueeze(-1)
            intent_logits = out["intent_logits"] * mask
            fill_logits   = out["fill_logits"] * mask
            masked_targets = binary_targets * mask

            loss, loss_dict = total_loss(
                intent_logits = intent_logits,
                fill_logits   = fill_logits,
                targets       = masked_targets,
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

        # B4 — log mean gate value this epoch (plot across epochs in W&B)
        if gate_vals_epoch:
            mean_gate = sum(gate_vals_epoch) / len(gate_vals_epoch)
            epoch_log["train/mean_gate"] = mean_gate
            logger.info(f"  mean gate={mean_gate:.4f}")

        # B8 — gate vs basket size correlation, logged every epoch
        if gate_vals_epoch:
            g = torch.tensor(gate_vals_epoch)
            s = torch.tensor(basket_sizes_epoch)
            corr = torch.corrcoef(torch.stack([g, s]))[0, 1].item()
            epoch_log["train/gate_basket_size_corr"] = corr
            if (epoch + 1) == cfg.trainer.max_epochs:
                logger.info(f"  Gate vs basket size correlation (final): {corr:.4f}")

        logger.info(
            f"  Phase 3 epoch {epoch+1}/{cfg.trainer.max_epochs}  "
            f"loss={avg_epoch_loss:.4f}  "
            f"intent={epoch_losses.get('loss/intent',0)/max(num_batches,1):.4f}  "
            f"fill={epoch_losses.get('loss/fill',0)/max(num_batches,1):.4f}  "
            f"orth={epoch_losses.get('loss/orth',0)/max(num_batches,1):.4f}  "
            f"mlm={epoch_losses.get('loss/mlm',0)/max(num_batches,1):.4f}"
        )

        torch.save({
            "epoch": epoch + 1,
            "step": step_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "train_loss": avg_epoch_loss,
        }, last_ckpt_path)

        if (epoch + 1) % eval_every == 0 or (epoch + 1) == cfg.trainer.max_epochs:
            val_metrics = run_inference(model, val_loader, device, cfg, alpha_idf, cooc)
            epoch_log.update(val_metrics)

            current_recall = val_metrics.get("val/recall@10", 0.0)

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

            model.train()

        wandb.log(epoch_log)

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
#  Inference — Section VI six-step two-stage residual decode                   #
# ============================================================================ #

@torch.no_grad()
def run_inference(
    model: IntentAwareNBR,
    val_loader,
    device: torch.device,
    cfg: DictConfig,
    alpha_idf,
    cooc,
) -> dict:
    logger.info("Running Section VI two-stage residual-decode inference...")
    model.eval()

    k1 = cfg.model.k1
    k2 = cfg.model.k2

    recalls_10, ndcgs_10 = [], []
    recalls_20, ndcgs_20 = [], []
    repeat_recalls, explore_recalls = [], []
    core_counts_twostage, core_counts_flat = [], []
    core_precisions = []
    diversities_twostage, diversities_flat = [], []
    fill_coherences_ts, fill_coherences_flat = [], []
    user_segment_records = []
    per_user_records = []

    for batch in val_loader:
        items       = batch["items"].to(device)
        item_mask   = batch["item_mask"].to(device)
        basket_mask = batch["basket_mask"].to(device)

        out = model(items, item_mask, basket_mask)

        idx = torch.arange(items.size(0), device=device)
        h_next           = out["next_basket_repr"][idx, seq_lengths, :]
        vocab_embeddings = model.item_embedding.embedding.weight

        basket_mask_target = batch["basket_mask_target"].to(device)
        targets = batch["targets"].to(device)

        seq_lengths = basket_mask_target.sum(dim=1).long() - 1
        seq_lengths = seq_lengths.clamp(min=0)

        for i in range(items.size(0)):
            target = targets[i, seq_lengths[i]]
            vocab_size = vocab_embeddings.size(0)
            if target.shape[-1] < vocab_size:
                target = F.pad(target, (0, vocab_size - target.shape[-1]), value=0)

            predicted_basket = residual_decode(
                repr_vec        = h_next[i],
                item_embeddings = vocab_embeddings,
                decoder         = model.decoder,
                k1              = k1,
                k2              = k2,
                excluded        = set(),
            )

            # B1 — Repeat vs Explore
            history_items = set(items[i][batch["basket_mask"][i].bool()].flatten().tolist()) - {0}
            gt_set   = set(target.nonzero(as_tuple=True)[0].tolist())
            recs_set = set(predicted_basket[:10])
            repeat_gt  = gt_set & history_items
            explore_gt = gt_set - history_items
            if repeat_gt:
                repeat_recalls.append(len(recs_set & repeat_gt) / len(repeat_gt))
            if explore_gt:
                explore_recalls.append(len(recs_set & explore_gt) / len(explore_gt))

            # B2 — Slot saturation
            tau = cfg.model.tau_alpha
            pred_alphas = alpha_idf[torch.tensor(predicted_basket[:10], device=device)]
            core_counts_twostage.append((pred_alphas > tau).sum().item())

            flat_scores = vocab_embeddings @ h_next[i]
            flat_top10  = torch.topk(flat_scores, 10).indices.tolist()
            flat_alphas = alpha_idf[torch.tensor(flat_top10, device=device)]
            core_counts_flat.append((flat_alphas > tau).sum().item())

            # B5 — Core precision
            core_preds = set(predicted_basket[:k1])
            core_precisions.append(len(core_preds & gt_set) / max(k1, 1))

            # B6 — Intra-list diversity
            pred_embs  = F.normalize(vocab_embeddings[torch.tensor(predicted_basket[:10])], dim=-1)
            sim_matrix = pred_embs @ pred_embs.T
            upper_tri  = sim_matrix[torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)]
            diversities_twostage.append(1 - upper_tri.mean().item())

            flat_embs  = F.normalize(vocab_embeddings[torch.tensor(flat_top10)], dim=-1)
            sim_flat   = flat_embs @ flat_embs.T
            upper_flat = sim_flat[torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)]
            diversities_flat.append(1 - upper_flat.mean().item())

            # B9 — User segmentation (planner vs opportunist)
            user_basket_alphas = []
            for t in range(basket_mask[i].sum().item()):
                ids = items[i, t][item_mask[i, t].bool()].tolist()
                if ids:
                    user_basket_alphas.append(alpha_idf[torch.tensor(ids, device=device)].mean().item())
            if len(user_basket_alphas) > 1:
                user_alpha_std = torch.tensor(user_basket_alphas).std().item()
                user_type = "planner" if user_alpha_std < 0.15 else "opportunist"
            else:
                user_type = "unknown"

            # B10 — Fill coherence
            core_ids = predicted_basket[:k1]
            fill_ids = predicted_basket[k1:k1 + k2]
            flat_fill = flat_top10[k1:]
            if core_ids and fill_ids:
                coherence_ts   = cooc[torch.tensor(core_ids)][:, torch.tensor(fill_ids)].mean().item()
                coherence_flat = cooc[torch.tensor(core_ids)][:, torch.tensor(flat_fill)].mean().item()
                fill_coherences_ts.append(coherence_ts)
                fill_coherences_flat.append(coherence_flat)

            recs_10 = predicted_basket[:10]
            recs_20 = predicted_basket[:20]

            preds_10 = torch.zeros_like(target)
            preds_20 = torch.zeros_like(target)
            valid_10 = [idx for idx in recs_10 if 0 <= idx < vocab_size]
            valid_20 = [idx for idx in recs_20 if 0 <= idx < vocab_size]
            if valid_10:
                preds_10[torch.tensor(valid_10, device=target.device)] = 1.0
            if valid_20:
                preds_20[torch.tensor(valid_20, device=target.device)] = 1.0

            r10 = recall_at_k(preds_10.unsqueeze(0), target.unsqueeze(0), 10).item()
            recalls_10.append(r10)
            ndcgs_10.append(ndcg_at_k(preds_10.unsqueeze(0), target.unsqueeze(0), 10).item())
            recalls_20.append(recall_at_k(preds_20.unsqueeze(0), target.unsqueeze(0), 20).item())
            ndcgs_20.append(ndcg_at_k(preds_20.unsqueeze(0), target.unsqueeze(0), 20).item())

            # B7 — per-user record for history-length stratification
            per_user_records.append({
                "history_len": basket_mask[i].sum().item(),
                "recall@10":   r10,
                "user_type":   user_type,   # B9 also stored here
            })
            user_segment_records.append({"type": user_type, "recall@10": r10})

    # B7 — save CSV
    import pandas as pd
    processed_dir = PROJECT_ROOT / "data" / "processed" / cfg.dataset.name
    pd.DataFrame(per_user_records).to_csv(processed_dir / "val_predictions.csv", index=False)

    # B9 — log segmentation summary
    seg_df = pd.DataFrame(user_segment_records)
    if not seg_df.empty:
        seg_summary = seg_df.groupby("type")["recall@10"].mean().to_dict()
        logger.info(f"  User segmentation recall@10: {seg_summary}")

    n = max(len(recalls_10), 1)
    metrics = {
        "val/recall@10":               sum(recalls_10)  / n,
        "val/ndcg@10":                 sum(ndcgs_10)    / n,
        "val/recall@20":               sum(recalls_20)  / n,
        "val/ndcg@20":                 sum(ndcgs_20)    / n,
        "val/repeat_recall@10":        sum(repeat_recalls)  / max(len(repeat_recalls), 1),
        "val/explore_recall@10":       sum(explore_recalls) / max(len(explore_recalls), 1),
        "val/core_count_twostage":     sum(core_counts_twostage) / max(len(core_counts_twostage), 1),
        "val/core_count_flat":         sum(core_counts_flat)     / max(len(core_counts_flat), 1),
        "val/core_precision@k1":       sum(core_precisions)      / max(len(core_precisions), 1),
        "val/diversity_twostage":      sum(diversities_twostage) / max(len(diversities_twostage), 1),
        "val/diversity_flat":          sum(diversities_flat)     / max(len(diversities_flat), 1),
        "val/fill_coherence_twostage": sum(fill_coherences_ts)   / max(len(fill_coherences_ts), 1),
        "val/fill_coherence_flat":     sum(fill_coherences_flat) / max(len(fill_coherences_flat), 1),
    }

    logger.info("=== Inference Metrics ===")
    for k, v in metrics.items():
        logger.info(f"  {k:<20} {v:.4f}")

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

    # Build co-occurrence matrix for B9/B10
    V = datamodule.vocab_size + 2
    cooc = torch.zeros(V, V, dtype=torch.float32)
    for batch in train_loader:
        items_b = batch["items"]        # (B, T, S)
        for b in range(items_b.size(0)):
            for t in range(items_b.size(1)):
                ids = items_b[b, t][batch["item_mask"][b, t].bool()].tolist()
                for x in ids:
                    for y in ids:
                        if x != y:
                            cooc[x, y] += 1
    cooc = cooc.to(device)


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
        cooc=cooc,
    )

    # ------------------------------------------------------------------ #
    # Final inference on val set (uses best checkpoint from training)     #
    # ------------------------------------------------------------------ #
    best_ckpt = processed_dir / "full_model_best.pt"
    if best_ckpt.exists():
        logger.info(f"Loading best checkpoint from {best_ckpt}")
        best_state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(best_state["model_state_dict"])

    metrics = run_inference(model, val_loader, device, cfg, alpha_idf, cooc)
    wandb.log({"final/" + k.split("/")[1]: v for k, v in metrics.items()})
    wandb.finish()
    logger.info("Done.")


if __name__ == "__main__":
    main()
