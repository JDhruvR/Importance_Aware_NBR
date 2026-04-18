"""Train plain basket-BERT warmup with MLM objective."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import hydra
import polars as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nbr.models.bert import BasketBERT
from nbr.train.bert_data_module import BasketBERTDataModule
from nbr.utils.logger import setup_logging, wandb_env
from nbr.utils.seed import seed_everything


def _default_word2vec_path(cfg: DictConfig) -> Path:
    return Path(str(cfg.data.processed_dir)) / f"word2vec_dim{int(cfg.model.D)}.kv"


def _save_encoder_bundle(model: BasketBERT, output_dir: Path, dataset_name: str) -> Path:
    bundle = {
        "dataset": dataset_name,
        "num_items": model.num_items,
        "dim": model.dim,
        "pad_token_id": model.pad_token_id,
        "mask_token_id": model.mask_token_id,
        "item_id_offset": model.item_id_offset,
        "state_dict": {
            "embedding.weight": model.embedding.weight.detach().cpu(),
            "encoder": {k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
        },
    }
    out_path = output_dir / f"bert_encoder_bundle_{dataset_name}.pt"
    torch.save(bundle, out_path)
    return out_path


def _masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != -100
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)
    pred = logits.argmax(dim=-1)
    return (pred[valid] == labels[valid]).float().mean()


def _run_eval(model: BasketBERT, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attention_mask)
            logits = out["mlm_logits"]
            loss = model.mlm_loss(logits, labels)
            acc = _masked_accuracy(logits, labels)

            total_loss += float(loss.item())
            total_acc += float(acc.item())
            total_batches += 1

    mean_loss = total_loss / max(1, total_batches)
    mean_acc = total_acc / max(1, total_batches)
    return {
        "val/mlm_loss": mean_loss,
        "val/masked_acc": mean_acc,
        "val/perplexity": float(min(1e6, torch.exp(torch.tensor(mean_loss)).item())),
    }


def _compute_item_neighbor_drift(model: BasketBERT, sample_items: int = 256) -> dict[str, float]:
    item_emb = model.embedding.weight.detach().cpu()[model.item_id_offset :]
    total_items = item_emb.shape[0]
    n = min(sample_items, total_items)
    query = F.normalize(item_emb[:n], dim=-1)
    base = F.normalize(item_emb, dim=-1)
    sim = query @ base.T
    top2 = torch.topk(sim, k=2, dim=1).indices
    nn_ids = top2[:, 1]
    overlap = (nn_ids < n).float().mean().item()
    return {
        "nn_overlap_prefix": overlap,
    }


def _compute_cls_sanity(
    model: BasketBERT,
    processed_dir: str | Path,
    num_items: int,
    max_baskets: int,
) -> dict[str, float]:
    from nbr.data.basket_mlm_dataset import BasketMLMCollator, BasketSample

    grouped = (
        pl.read_parquet(Path(processed_dir) / "baskets.parquet")
        .group_by(["user_id", "order_idx"])
        .agg(pl.col("item_id").alias("items"))
        .sort(["user_id", "order_idx"])
    )

    samples: list[BasketSample] = []
    for row in grouped.iter_rows(named=True):
        samples.append(
            BasketSample(
                user_id=int(row["user_id"]),
                order_idx=int(row["order_idx"]),
                items=[int(x) for x in row["items"]],
            )
        )
        if len(samples) >= max_baskets:
            break

    if len(samples) < 4:
        return {"adjacent_cos": 0.0, "random_cos": 0.0, "adj_minus_rand": 0.0}

    collator = BasketMLMCollator(num_items=num_items, apply_mlm=False)
    cls_all: list[torch.Tensor] = []
    users: list[int] = []
    orders: list[int] = []
    model.eval()
    with torch.no_grad():
        step = 256
        for i in range(0, len(samples), step):
            chunk = samples[i : i + step]
            batch = collator(chunk)
            out = model(batch["input_ids"], batch["attention_mask"])
            cls_all.append(out["cls_repr"].cpu())
            users.extend(batch["user_ids"].tolist())
            orders.extend(batch["order_idxs"].tolist())

    cls = F.normalize(torch.cat(cls_all, dim=0), dim=-1)
    sims = cls @ cls.T

    user_order_to_idx = {(u, o): i for i, (u, o) in enumerate(zip(users, orders, strict=True))}
    adj_scores: list[float] = []
    rand_scores: list[float] = []
    for i, (u, o) in enumerate(zip(users, orders, strict=True)):
        nxt = user_order_to_idx.get((u, o + 1))
        if nxt is not None:
            adj_scores.append(float(sims[i, nxt]))
        j = (i * 9973 + 17) % len(users)
        if j == i:
            j = (j + 1) % len(users)
        if users[j] == u:
            j = (j + 101) % len(users)
        rand_scores.append(float(sims[i, j]))

    adj = float(sum(adj_scores) / max(1, len(adj_scores)))
    rnd = float(sum(rand_scores) / max(1, len(rand_scores)))
    return {
        "adjacent_cos": adj,
        "random_cos": rnd,
        "adj_minus_rand": adj - rnd,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="bert_warmup")
def main(cfg: DictConfig) -> None:
    wandb_env()
    seed_everything(int(cfg.seed))
    setup_logging(cfg.output_dir)

    dm = BasketBERTDataModule(
        processed_dir=cfg.data.processed_dir,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        mask_prob=float(cfg.train.mask_prob),
        val_mask_prob=float(cfg.train.val_mask_prob),
        max_items_per_basket=cfg.train.max_items_per_basket,
        max_train_baskets=cfg.train.max_train_baskets,
        max_val_baskets=cfg.train.max_val_baskets,
    )
    dm.setup()

    model = BasketBERT(
        num_items=dm.num_items,
        dim=int(cfg.model.D),
        num_heads=int(cfg.model.num_heads),
        num_layers=int(cfg.model.L1),
        dropout=float(cfg.train.dropout),
        pad_token_id=int(cfg.model.pad_token_id),
        mask_token_id=int(cfg.model.mask_token_id),
        item_id_offset=int(cfg.model.item_id_offset),
    )

    w2v_loaded = 0
    w2v_missing = 0
    if bool(cfg.train.use_word2vec_init):
        w2v_path = (
            Path(str(cfg.train.word2vec_path))
            if cfg.train.word2vec_path
            else _default_word2vec_path(cfg)
        )
        if not w2v_path.exists():
            raise FileNotFoundError(f"Word2Vec file not found: {w2v_path}")
        w2v_loaded, w2v_missing = model.init_item_embeddings_from_word2vec(w2v_path)

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=max(1, int(cfg.train.warmup_steps)),
    )

    run_name = (
        str(cfg.train.run_name)
        if cfg.train.run_name
        else f"bert-warmup-{Path(str(cfg.data.processed_dir)).name}"
    )
    run = wandb.init(
        project=str(cfg.train.wandb_project),
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(cfg.output_dir),
    )
    if bool(cfg.train.use_word2vec_init):
        run.summary["word2vec_loaded"] = w2v_loaded
        run.summary["word2vec_missing"] = w2v_missing

    output_dir = Path(str(cfg.output_dir))
    best_ckpt = output_dir / "bert_best.pt"
    last_ckpt = output_dir / "bert_last.pt"
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_steps = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = perf_counter()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attention_mask)
            logits = out["mlm_logits"]
            loss = model.mlm_loss(logits, labels)
            acc = _masked_accuracy(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_steps += 1
            epoch_loss += float(loss.item())
            epoch_acc += float(acc.item())

            if global_step % int(cfg.train.log_every_n_steps) == 0:
                run.log(
                    {
                        "train/mlm_loss_step": float(loss.item()),
                        "train/masked_acc_step": float(acc.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                    },
                    step=global_step,
                )

        train_loss = epoch_loss / max(1, epoch_steps)
        train_acc = epoch_acc / max(1, epoch_steps)
        val_metrics = _run_eval(model, val_loader, device)
        epoch_time = perf_counter() - t0

        run.log(
            {
                "epoch": epoch,
                "train/mlm_loss": train_loss,
                "train/masked_acc": train_acc,
                "train/perplexity": float(min(1e6, torch.exp(torch.tensor(train_loss)).item())),
                "train/lr_epoch": float(optimizer.param_groups[0]["lr"]),
                "train/epoch_seconds": epoch_time,
                **val_metrics,
            },
            step=global_step,
        )

        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_mlm_loss": val_metrics["val/mlm_loss"],
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            last_ckpt,
        )

        if val_metrics["val/mlm_loss"] < best_val:
            best_val = val_metrics["val/mlm_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_mlm_loss": best_val,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                best_ckpt,
            )

    dataset_name = Path(str(cfg.data.processed_dir)).name
    best_bundle = _save_encoder_bundle(model, output_dir, dataset_name)

    sanity_report = output_dir / "bert_sanity_checks.txt"
    if bool(cfg.train.run_sanity_checks):
        drift = _compute_item_neighbor_drift(model)
        cls_stats = _compute_cls_sanity(
            model,
            processed_dir=cfg.data.processed_dir,
            num_items=dm.num_items,
            max_baskets=int(cfg.train.sanity_max_baskets),
        )
        with open(sanity_report, "w") as f:
            f.write(f"dataset={dataset_name}\n")
            f.write(f"word2vec_loaded={w2v_loaded}\n")
            f.write(f"word2vec_missing={w2v_missing}\n")
            for k, v in drift.items():
                f.write(f"{k}={v:.6f}\n")
            for k, v in cls_stats.items():
                f.write(f"{k}={v:.6f}\n")
        run.summary.update(drift)
        run.summary.update(cls_stats)

    run.summary["encoder_bundle_path"] = str(best_bundle)
    run.summary["best_checkpoint"] = str(best_ckpt)
    run.summary["last_checkpoint"] = str(last_ckpt)
    run.summary["best_val_mlm_loss"] = best_val
    run.summary["output_dir"] = str(output_dir)
    run.finish()


if __name__ == "__main__":
    main()
