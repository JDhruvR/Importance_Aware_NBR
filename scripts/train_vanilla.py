"""Train vanilla or BERT+GPT NBR with Lightning."""

from __future__ import annotations

from pathlib import Path

import torch
import hydra
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from nbr.train.data_module import BasketDataModule
from nbr.train.vanilla_lightning import VanillaLitModule
from nbr.utils.logger import setup_logging, wandb_env
from nbr.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb_env()
    seed_everything(int(cfg.seed))
    setup_logging(cfg.output_dir)

    # Resolve bert_bundle_path and item_id_offset
    bert_bundle_path: str | None = None
    item_id_offset = 0
    if str(cfg.model.model_type) == "bert_gpt":
        raw_path = str(cfg.model.bert_bundle_path)
        p = Path(raw_path)
        if not p.is_absolute():
            import hydra.utils
            p = Path(hydra.utils.get_original_cwd()) / raw_path
        bert_bundle_path = str(p)
        
        # Peek at bundle to get the correct offset
        try:
            bundle = torch.load(bert_bundle_path, map_location="cpu")
            item_id_offset = int(bundle.get("item_id_offset", 2))
        except Exception:
            item_id_offset = 2  # default

    data = BasketDataModule(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.train.max_seq_len,
        num_workers=int(cfg.train.num_workers),
        item_id_offset=item_id_offset,
    )
    data.setup()

    model = VanillaLitModule(
        model_type=str(cfg.model.model_type),
        vocab_size=data.vocab_size,
        dim=int(cfg.model.D),
        num_heads=int(cfg.model.num_heads),
        encoder_layers=int(cfg.model.L1),
        gpt_layers=int(cfg.model.L2),
        dropout=float(cfg.train.dropout),
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        warmup_steps=int(cfg.train.warmup_steps),
        max_epochs=int(cfg.train.epochs),
        k_values=list(cfg.train.k_values),
        bert_bundle_path=bert_bundle_path,
        output_dir=str(cfg.output_dir),
        dataset_name=str(Path(str(cfg.data.processed_dir)).name),
    )

    wandb_logger = WandbLogger(
        project=str(cfg.train.wandb_project),
        name=str(cfg.train.run_name) if cfg.train.run_name else None,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=str(cfg.output_dir),
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=str(cfg.output_dir),
            monitor=str(cfg.train.monitor_metric),
            mode=str(cfg.train.monitor_mode),
            save_top_k=int(cfg.train.save_top_k),
            filename="{epoch}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        max_epochs=int(cfg.train.epochs),
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        gradient_clip_val=float(cfg.train.grad_clip),
        accumulate_grad_batches=int(cfg.train.accumulate_grad_batches),
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=data)
    
    # Final saving of the BEST model (not the last)
    best_ckpt_path = trainer.checkpoint_callback.best_model_path
    if best_ckpt_path:
        print(f"Loading best model from {best_ckpt_path} for final preservation...", flush=True)
        # Load best weights into the LightningModule
        checkpoint = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("No best checkpoint found. Saving last model state.", flush=True)

    # Save weights and bert bundle if needed
    output_dir = Path(cfg.output_dir)
    # Re-derive output paths
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = str(Path(str(cfg.data.processed_dir)).name)
    
    full_model_path = output_dir / f"vanilla_model_{dataset_name}.pt"
    torch.save(model.model.state_dict(), full_model_path)
    print(f"[vanilla] Best model state dict saved → {full_model_path}", flush=True)

    from nbr.models.bert_gpt.model import BertGptNBR
    if isinstance(model.model, BertGptNBR):
        saved = model.model.save_bert_bundle(str(output_dir), dataset_name)
        print(f"[bert_gpt] Best BERT bundle saved → {saved}", flush=True)

    # Test with best model
    trainer.test(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    main()
