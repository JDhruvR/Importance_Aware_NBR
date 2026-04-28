"""Evaluate a trained Vanilla NBR model from a checkpoint."""

import argparse
from pathlib import Path
import torch
from lightning import Trainer

from nbr.train.data_module import BasketDataModule
from nbr.train.vanilla_lightning import VanillaLitModule

def main():
    parser = argparse.ArgumentParser(description="Evaluate Vanilla NBR Model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .ckpt file")
    parser.add_argument("--processed_dir", type=str, default="data/processed/instacart", 
                        help="Path to processed data directory")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=10)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading model from: {ckpt_path}")
    # load_from_checkpoint automatically restores the model architecture 
    # based on saved hyperparameters.
    model = VanillaLitModule.load_from_checkpoint(args.ckpt)
    model.eval()

    print(f"Setting up data from: {args.processed_dir}")
    data = BasketDataModule(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=4,
    )

    # Run evaluation using Lightning's Trainer
    trainer = Trainer(
        accelerator="auto",
        devices=1,
        logger=False, # Disable logging for pure evaluation
        enable_checkpointing=False,
    )

    print("Starting evaluation on test set...")
    results = trainer.test(model, datamodule=data)

    print("\n" + "="*30)
    print("FINAL TEST METRICS")
    print("="*30)
    for metric, value in results[0].items():
        if metric.startswith("test/"):
            print(f"{metric:25}: {value:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()
