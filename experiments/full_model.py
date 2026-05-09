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
            original_items = items.clone()
            rand_vals = torch.rand_like(items, dtype=torch.float32)
            mlm_mask = item_mask.bool() & (rand_vals < mlm_mask_prob)
            items = items.clone()
            items[mlm_mask] = mask_token_id
            mlm_targets = _build_mlm_targets(original_items, mlm_mask, mask_token_id)

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
#  Inference — Section VI six-step two-stage residual decode                   #
# ============================================================================ #

@torch.no_grad()
def run_inference(
    model: IntentAwareNBR,
    val_loader,
    device: torch.device,
    cfg: DictConfig,
) -> dict:
    """
    Section VI inference: encode each user's history, then run the two-stage
    residual decode to produce the predicted next basket.

    Steps 1-3 are performed by model.forward() which returns next_basket_repr.
    Steps 4-6 are performed by residual_decode() using next_basket_repr[:, -1, :].

    The decoded basket contains K1 core items (Stage 1) + K2 fill items (Stage 2).
    Both @10 and @20 metrics are evaluated from a single K1+K2=20 decode pass.
    """
    logger.info("Running Section VI two-stage residual-decode inference...")
    model.eval()

    k1 = cfg.model.k1   # core items  (Stage 1, intent subspace)
    k2 = cfg.model.k2   # fill items  (Stage 2, conditioned on discovered core)

    recalls_10, ndcgs_10 = [], []
    recalls_20, ndcgs_20 = [], []

    for batch in val_loader:
        items       = batch["items"].to(device)
        item_mask   = batch["item_mask"].to(device)
        basket_mask = batch["basket_mask"].to(device)

        out = model(items, item_mask, basket_mask)

        # next_basket_repr[:, -1, :] is h_{T+1} — the GPT output at the final
        # position, i.e. the prediction of the next basket given all history.
        # This is NOT cls_repr, which encodes each observed past basket.
        h_next           = out["next_basket_repr"][:, -1, :]    # (B, D)
        vocab_embeddings = model.item_embedding.embedding.weight # (V, D)

        for i in range(items.size(0)):
            target = batch["target_basket"][i]

            # Section VI Steps 4-6: two-stage residual decode
            predicted_basket = residual_decode(
                repr_vec        = h_next[i],
                item_embeddings = vocab_embeddings,
                decoder         = model.decoder,
                k1              = k1,
                k2              = k2,
                excluded        = set(),
            )

            recs_10 = predicted_basket[:10]
            recs_20 = predicted_basket[:20]

            recalls_10.append(recall_at_k(recs_10, target, 10))
            ndcgs_10.append(ndcg_at_k(recs_10, target, 10))
            recalls_20.append(recall_at_k(recs_20, target, 20))
            ndcgs_20.append(ndcg_at_k(recs_20, target, 20))

    n = max(len(recalls_10), 1)
    metrics = {
        "val/recall@10": sum(recalls_10) / n,
        "val/ndcg@10":   sum(ndcgs_10)   / n,
        "val/recall@20": sum(recalls_20) / n,
        "val/ndcg@20":   sum(ndcgs_20)   / n,
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
        name="full_model",
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