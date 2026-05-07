"""Pre-train the ImportanceHead MLP to reproduce alpha_idf targets.

Loads a frozen BERT encoder, feeds training baskets through it, and
optimizes the ImportanceHead against normalized alpha_idf targets using
masked MSE loss.  Only the MLP params (~8K) are trained.

Usage:
    PYTHONPATH=. uv run python scripts/train_importance_head.py data=instacart
    PYTHONPATH=. uv run python scripts/train_importance_head.py data=tafeng
"""

from __future__ import annotations

import math
from pathlib import Path
from time import perf_counter

import hydra
import numpy as np
import polars as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from nbr.data.split import split_user_baskets
from nbr.models.encoder import IntraBasketEncoder
from nbr.models.importance import ImportanceHead, importance_init_loss
from nbr.utils.device import get_device
from nbr.utils.logger import setup_logging, wandb_env
from nbr.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BasketImportanceDataset(Dataset):
    """Yields (input_ids, attention_mask, target_importance) per basket."""

    def __init__(
        self,
        baskets: list[list[int]],
        alpha_idf: np.ndarray,
        item_id_offset: int,
        max_items: int,
    ) -> None:
        self.baskets = baskets
        self.alpha_idf = alpha_idf
        self.item_id_offset = item_id_offset
        self.max_items = max_items

    def __len__(self) -> int:
        return len(self.baskets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        items = self.baskets[idx][: self.max_items]
        S = len(items)
        input_ids = torch.tensor(
            [iid + self.item_id_offset for iid in items], dtype=torch.long,
        )
        attention_mask = torch.ones(S, dtype=torch.bool)
        targets = torch.tensor(
            [self.alpha_idf[iid] for iid in items], dtype=torch.float32,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "targets": targets,
        }


def _collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length baskets to the same length."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    input_ids_list = []
    mask_list = []
    target_list = []

    for b in batch:
        S = b["input_ids"].shape[0]
        pad = max_len - S
        input_ids_list.append(
            torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)])
        )
        mask_list.append(
            torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.bool)])
        )
        target_list.append(
            torch.cat([b["targets"], torch.zeros(pad, dtype=torch.float32)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),       # (B, S)
        "attention_mask": torch.stack(mask_list),         # (B, S)
        "targets": torch.stack(target_list),              # (B, S)
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        remain = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / remain)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _run_eval(
    embedding: torch.nn.Embedding,
    encoder: IntraBasketEncoder,
    head: ImportanceHead,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    head.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            token_emb = embedding(input_ids)
            _, item_reprs = encoder(token_emb, mask)
            predicted = head(item_reprs)

            loss = importance_init_loss(predicted, targets, mask)
            total_loss += float(loss.item())
            total_batches += 1

    return {"val/mse_loss": total_loss / max(1, total_batches)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="train_importance_head",
)
def main(cfg: DictConfig) -> None:
    wandb_env()
    seed_everything(int(cfg.seed))
    setup_logging(cfg.output_dir)
    device = get_device() if str(cfg.device) == "auto" else torch.device(cfg.device)

    processed_dir = Path(str(cfg.data.processed_dir))
    dataset_name = processed_dir.name

    # ------------------------------------------------------------------
    # 1. Load frozen BERT encoder
    # ------------------------------------------------------------------
    bundle_path = processed_dir / f"bert_encoder_bundle_{dataset_name}.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Encoder bundle not found: {bundle_path}")

    print(f"[imp-head] Loading encoder bundle from {bundle_path}", flush=True)
    bundle = torch.load(bundle_path, map_location="cpu")

    dim = bundle["dim"]
    num_items = bundle["num_items"]
    item_id_offset = bundle["item_id_offset"]
    vocab_size = num_items + item_id_offset

    embedding = torch.nn.Embedding(vocab_size, dim, padding_idx=bundle["pad_token_id"])
    embedding.weight.data.copy_(bundle["state_dict"]["embedding.weight"])
    embedding = embedding.to(device)
    embedding.eval()
    for p in embedding.parameters():
        p.requires_grad = False

    encoder = IntraBasketEncoder(
        dim=dim,
        num_heads=int(cfg.model.num_heads),
        num_layers=int(cfg.model.L1),
        dropout=0.0,
    )
    encoder.load_state_dict(bundle["state_dict"]["encoder"])
    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # ------------------------------------------------------------------
    # 2. Load and normalize alpha_idf targets
    # ------------------------------------------------------------------
    scores_path = processed_dir / "importance_scores.npz"
    if not scores_path.exists():
        raise FileNotFoundError(f"Importance scores not found: {scores_path}")

    data = np.load(scores_path)
    alpha_idf_raw = data["alpha_idf"]  # (num_items,)
    alpha_max = alpha_idf_raw.max()
    if alpha_max > 0:
        alpha_idf_norm = alpha_idf_raw / alpha_max  # normalize to [0, 1]
    else:
        alpha_idf_norm = alpha_idf_raw.copy()

    print(
        f"[imp-head] alpha_idf: raw_max={alpha_max:.4f}, "
        f"normalized range=[{alpha_idf_norm.min():.4f}, {alpha_idf_norm.max():.4f}]",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 3. Build datasets
    # ------------------------------------------------------------------
    df = pl.read_parquet(processed_dir / "baskets.parquet")
    train_df, val_df, _ = split_user_baskets(df)

    def _group_baskets(split_df: pl.DataFrame) -> list[list[int]]:
        return (
            split_df.group_by(["user_id", "order_idx"])
            .agg(pl.col("item_id"))
            .select("item_id")
            .to_series()
            .to_list()
        )

    train_baskets = _group_baskets(train_df)
    val_baskets = _group_baskets(val_df)

    max_items = int(cfg.train.max_items_per_basket)
    if cfg.train.max_train_baskets is not None:
        train_baskets = train_baskets[: int(cfg.train.max_train_baskets)]
    if cfg.train.max_val_baskets is not None:
        val_baskets = val_baskets[: int(cfg.train.max_val_baskets)]

    train_ds = BasketImportanceDataset(
        train_baskets, alpha_idf_norm, item_id_offset, max_items,
    )
    val_ds = BasketImportanceDataset(
        val_baskets, alpha_idf_norm, item_id_offset, max_items,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # 4. Build ImportanceHead + optimizer
    # ------------------------------------------------------------------
    head = ImportanceHead(dim=dim).to(device)
    trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[imp-head] ImportanceHead params: {trainable_params:,}", flush=True)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    total_steps = int(cfg.train.epochs) * max(1, len(train_loader))
    scheduler = _build_warmup_cosine_scheduler(
        optimizer, total_steps,
        warmup_steps=int(cfg.train.warmup_steps),
        min_lr_ratio=float(cfg.train.min_lr_ratio),
    )

    # ------------------------------------------------------------------
    # 5. W&B init
    # ------------------------------------------------------------------
    run_name = (
        str(cfg.train.run_name)
        if cfg.train.run_name
        else f"imp-head-{dataset_name}"
    )
    run = wandb.init(
        project=str(cfg.train.wandb_project),
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(cfg.output_dir),
    )
    run.summary["trainable_params"] = trainable_params
    run.summary["alpha_idf_max"] = float(alpha_max)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    output_dir = Path(str(cfg.output_dir))
    best_ckpt = output_dir / "importance_head_best.pt"
    last_ckpt = output_dir / "importance_head_last.pt"
    best_val = float("inf")
    best_epoch = 0
    global_step = 0
    no_improve_epochs = 0
    start_time = perf_counter()

    print(
        f"[imp-head] start dataset={dataset_name} "
        f"epochs={int(cfg.train.epochs)} batch_size={int(cfg.train.batch_size)} "
        f"train_baskets={len(train_ds)} val_baskets={len(val_ds)} "
        f"device={device.type}",
        flush=True,
    )

    for epoch in range(1, int(cfg.train.epochs) + 1):
        head.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = perf_counter()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            # Frozen forward pass through BERT
            with torch.no_grad():
                token_emb = embedding(input_ids)
                _, item_reprs = encoder(token_emb, mask)

            # Trainable forward pass through head
            predicted = head(item_reprs)
            loss = importance_init_loss(predicted, targets, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(cfg.train.grad_clip))
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_steps += 1
            epoch_loss += float(loss.item())

            if global_step % int(cfg.train.log_every_n_steps) == 0:
                run.log(
                    {
                        "train/mse_loss_step": float(loss.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )

        train_loss = epoch_loss / max(1, epoch_steps)
        val_metrics = _run_eval(embedding, encoder, head, val_loader, device)
        epoch_time = perf_counter() - t0
        elapsed = perf_counter() - start_time
        avg_epoch = elapsed / epoch
        eta = avg_epoch * (int(cfg.train.epochs) - epoch)

        print(
            f"[imp-head] epoch={epoch}/{int(cfg.train.epochs)} "
            f"train_mse={train_loss:.6f} val_mse={val_metrics['val/mse_loss']:.6f} "
            f"epoch_s={epoch_time:.1f} elapsed_s={elapsed:.1f} eta_s={eta:.1f}",
            flush=True,
        )

        run.log(
            {
                "epoch": epoch,
                "train/mse_loss": train_loss,
                "train/lr_epoch": float(optimizer.param_groups[0]["lr"]),
                "train/epoch_seconds": epoch_time,
                **val_metrics,
            },
            step=global_step,
        )

        # Save last checkpoint
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_mse_loss": val_metrics["val/mse_loss"],
            "alpha_idf_max": float(alpha_max),
            "dim": dim,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        torch.save(ckpt, last_ckpt)

        # Best checkpoint
        if val_metrics["val/mse_loss"] < (best_val - float(cfg.train.early_stop_min_delta)):
            best_val = val_metrics["val/mse_loss"]
            best_epoch = epoch
            no_improve_epochs = 0
            ckpt["val_mse_loss"] = best_val
            torch.save(ckpt, best_ckpt)
        else:
            no_improve_epochs += 1

        if int(cfg.train.early_stop_patience) > 0:
            if no_improve_epochs >= int(cfg.train.early_stop_patience):
                print(
                    f"[imp-head] early_stop epoch={epoch} "
                    f"best_epoch={best_epoch} best_val={best_val:.6f} "
                    f"patience={int(cfg.train.early_stop_patience)}",
                    flush=True,
                )
                break

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    run.summary["best_val_mse"] = best_val
    run.summary["best_epoch"] = best_epoch
    run.summary["best_checkpoint"] = str(best_ckpt)
    run.summary["last_checkpoint"] = str(last_ckpt)
    run.finish()

    print(
        f"[imp-head] done best_epoch={best_epoch} best_val_mse={best_val:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
