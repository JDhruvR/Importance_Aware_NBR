"""Evaluate vanilla or BERT+GPT NBR with Lightning."""

from __future__ import annotations

from pathlib import Path

import torch
import hydra
from lightning import Trainer
from omegaconf import DictConfig

from nbr.train.data_module import BasketDataModule
from nbr.train.vanilla_lightning import VanillaLitModule
from nbr.utils.seed import seed_everything


import sys
ckpt_arg = [arg for arg in sys.argv if arg.startswith("ckpt_path=")]
if not ckpt_arg:
    print("Must provide ckpt_path=... in the command line (e.g. python -m scripts.eval_vanilla ckpt_path=outputs/.../epoch=29.ckpt)")
    sys.exit(1)

# Remove it from sys.argv so hydra doesn't crash on it
sys.argv.remove(ckpt_arg[0])
GLOBAL_CKPT = ckpt_arg[0].split("=", 1)[1]

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))

    ckpt_path = str(GLOBAL_CKPT)
    
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
        
        try:
            bundle = torch.load(bert_bundle_path, map_location="cpu")
            item_id_offset = int(bundle.get("item_id_offset", 2))
        except Exception:
            item_id_offset = 2

    data = BasketDataModule(
        processed_dir=cfg.data.processed_dir,
        batch_size=cfg.train.batch_size,
        max_seq_len=cfg.train.max_seq_len,
        num_workers=int(cfg.train.num_workers),
        item_id_offset=item_id_offset,
    )
    data.setup()

    print(f"[eval_vanilla] Loading model from checkpoint: {ckpt_path}")
    model = VanillaLitModule.load_from_checkpoint(
        ckpt_path,
        bert_bundle_path=bert_bundle_path,
        strict=False,
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        logger=False, # We don't need WandB just for eval
    )

    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
