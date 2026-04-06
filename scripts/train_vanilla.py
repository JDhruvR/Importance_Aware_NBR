"""Train vanilla or BERT+GPT baselines with Lightning."""

from __future__ import annotations

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

    data = BasketDataModule(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.train.max_seq_len,
        num_workers=int(cfg.train.num_workers),
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
    )

    wandb_logger = WandbLogger(
        project=str(cfg.train.wandb_project),
        name=str(cfg.train.run_name),
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
    trainer.test(model, datamodule=data, ckpt_path="best")


if __name__ == "__main__":
    main()
