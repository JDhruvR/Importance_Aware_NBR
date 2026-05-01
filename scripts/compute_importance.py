"""Compute geometric importance scores from a trained BERT encoder bundle."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock
sys.modules['gensim'] = MagicMock()
sys.modules['gensim.models'] = MagicMock()
sys.modules['gensim.models.word2vec'] = MagicMock()

from pathlib import Path
from time import perf_counter

import argparse
if hasattr(argparse._ActionsContainer, '_check_help'):
    _original_check_help = argparse._ActionsContainer._check_help
    def patched_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = str(action.help)
        _original_check_help(self, action)
    argparse._ActionsContainer._check_help = patched_check_help

import hydra
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence

from nbr.data.split import split_user_baskets
from nbr.models.encoder import IntraBasketEncoder
from nbr.utils.device import get_device
from nbr.utils.seed import seed_everything
from nbr.utils.logger import setup_logging


def batch_baskets(baskets: list[list[int]], item_id_offset: int, max_batch_size: int):
    """Batch baskets efficiently.
    
    For a basket of size S, yields S+1 sequences (one full, S masked).
    Groups multiple baskets up to max_batch_size total sequences.
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_basket_sizes = []
    batch_items = []
    
    current_size = 0
    
    for basket in baskets:
        S = len(basket)
        if S == 0:
            continue
        
        if current_size + (S + 1) > max_batch_size and current_size > 0:
            yield _collate_baskets(batch_input_ids, batch_attention_mask, batch_basket_sizes, batch_items)
            batch_input_ids = []
            batch_attention_mask = []
            batch_basket_sizes = []
            batch_items = []
            current_size = 0
            
        basket_tensor = torch.tensor(basket, dtype=torch.long) + item_id_offset
        batch_input_ids.append(basket_tensor)
        batch_attention_mask.append(torch.ones(S, dtype=torch.bool))
        
        for i in range(S):
            batch_input_ids.append(basket_tensor)
            mask = torch.ones(S, dtype=torch.bool)
            mask[i] = False
            batch_attention_mask.append(mask)
            
        batch_basket_sizes.append(S)
        batch_items.append(basket)
        current_size += (S + 1)
        
    if current_size > 0:
        yield _collate_baskets(batch_input_ids, batch_attention_mask, batch_basket_sizes, batch_items)

def _collate_baskets(inputs, masks, sizes, items):
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(masks, batch_first=True, padding_value=False)
    return input_ids, attention_mask, sizes, items

@hydra.main(version_base=None, config_path="../configs", config_name="compute_importance")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    setup_logging(cfg.output_dir)
    device = get_device() if str(cfg.device) == "auto" else torch.device(cfg.device)
    
    processed_dir = Path(str(cfg.data.processed_dir))
    dataset_name = processed_dir.name
    
    print(f"[compute_importance] Dataset: {dataset_name}", flush=True)
    bundle_path = processed_dir / f"bert_encoder_bundle_{dataset_name}.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Encoder bundle not found at {bundle_path}")
        
    print(f"[compute_importance] Loading bundle from {bundle_path}...", flush=True)
    bundle = torch.load(bundle_path, map_location="cpu")
    
    dim = bundle["dim"]
    num_items = bundle["num_items"]
    item_id_offset = bundle["item_id_offset"]
    
    vocab_size = num_items + item_id_offset
    item_embedding = torch.nn.Embedding(vocab_size, dim, padding_idx=bundle["pad_token_id"])
    item_embedding.weight.data.copy_(bundle["state_dict"]["embedding.weight"])
    item_embedding = item_embedding.to(device)
    item_embedding.eval()
    
    encoder = IntraBasketEncoder(
        dim=dim,
        num_heads=int(cfg.model.num_heads),
        num_layers=int(cfg.model.L1),
        dropout=0.0
    )
    encoder.load_state_dict(bundle["state_dict"]["encoder"])
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f"[compute_importance] Loading baskets from {processed_dir / 'baskets.parquet'}...", flush=True)
    df = pl.read_parquet(processed_dir / "baskets.parquet")
    train_df, _, _ = split_user_baskets(df)
    
    # Group by user_id and order_idx
    baskets = train_df.group_by(["user_id", "order_idx"]).agg(pl.col("item_id")).select("item_id").to_series().to_list()
    
    if cfg.max_baskets is not None:
        baskets = baskets[:int(cfg.max_baskets)]
        
    N = len(baskets)
    print(f"[compute_importance] Processing {N} training baskets...", flush=True)
    
    accum_delta = np.zeros(num_items, dtype=np.float32)
    df_count = np.zeros(num_items, dtype=np.float32)
    
    # Pre-calculate df_count
    for basket in baskets:
        unique_items = np.unique(basket)
        df_count[unique_items] += 1
        
    start_time = perf_counter()
    processed_baskets = 0
    max_batch_size = int(cfg.batch_size)
    
    with torch.no_grad():
        for input_ids, attention_mask, sizes, items in batch_baskets(baskets, item_id_offset, max_batch_size):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            token_emb = item_embedding(input_ids)
            cls_repr, _ = encoder(token_emb, attention_mask)
            
            cls_reprs_split = torch.split(cls_repr, [S + 1 for S in sizes])
            
            for cls_k, S, items_k in zip(cls_reprs_split, sizes, items):
                cls_full = cls_k[0]       # (D,)
                cls_masked = cls_k[1:]    # (S, D)
                delta = torch.norm(cls_full.unsqueeze(0) - cls_masked, p=2, dim=1) # (S,)
                delta_sum = delta.sum()
                
                if delta_sum > 1e-6:
                    delta_norm = delta / delta_sum
                else:
                    delta_norm = torch.zeros_like(delta)
                
                # Accumulate
                delta_norm_cpu = delta_norm.cpu().numpy()
                for i, item_id in enumerate(items_k):
                    accum_delta[item_id] += delta_norm_cpu[i]
                    
            processed_baskets += len(sizes)
            if processed_baskets % 10000 == 0 or processed_baskets == N:
                elapsed = perf_counter() - start_time
                print(f"[compute_importance] Processed {processed_baskets}/{N} baskets ({elapsed:.1f}s)", flush=True)
                
    valid_df = df_count > 0
    raw_importance = np.zeros_like(accum_delta)
    raw_importance[valid_df] = accum_delta[valid_df] / df_count[valid_df]
    
    idf_factor = np.zeros_like(accum_delta)
    idf_factor[valid_df] = np.log(N / df_count[valid_df])
    
    alpha_idf = raw_importance * idf_factor
    
    out_path = processed_dir / "importance_scores.npz"
    np.savez_compressed(
        out_path,
        alpha_idf=alpha_idf,
        raw_importance=raw_importance,
        idf_factor=idf_factor
    )
    print(f"[compute_importance] Saved importance scores to {out_path}", flush=True)
    
    # Print statistics
    valid_alpha = alpha_idf[valid_df]
    valid_raw = raw_importance[valid_df]
    valid_idf = idf_factor[valid_df]
    
    if len(valid_alpha) > 0:
        print("\n--- alpha_idf statistics ---")
        print(f"Mean: {np.mean(valid_alpha):.6f}")
        print(f"Std:  {np.std(valid_alpha):.6f}")
        print(f"25th: {np.percentile(valid_alpha, 25):.6f}")
        print(f"50th: {np.percentile(valid_alpha, 50):.6f}")
        print(f"75th: {np.percentile(valid_alpha, 75):.6f}")
        
        print("\n--- raw_importance (delta_bar) statistics ---")
        print(f"Mean: {np.mean(valid_raw):.6f}")
        print(f"Std:  {np.std(valid_raw):.6f}")
        print(f"25th: {np.percentile(valid_raw, 25):.6f}")
        print(f"50th: {np.percentile(valid_raw, 50):.6f}")
        print(f"75th: {np.percentile(valid_raw, 75):.6f}")
        
        print("\n--- idf_factor statistics ---")
        print(f"Mean: {np.mean(valid_idf):.6f}")
        print(f"Std:  {np.std(valid_idf):.6f}")
        print(f"25th: {np.percentile(valid_idf, 25):.6f}")
        print(f"50th: {np.percentile(valid_idf, 50):.6f}")
        print(f"75th: {np.percentile(valid_idf, 75):.6f}")
    else:
        print("[compute_importance] Warning: No valid items found!")

if __name__ == "__main__":
    main()
