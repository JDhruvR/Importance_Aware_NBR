import argparse
from pathlib import Path

import polars as pl
from nbr.models.embeddings import Word2VecTrainer

# Define project root to build absolute paths
PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec embeddings on basket data."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["instacart", "tafeng", "dunnhumby"],
        help="Name of the dataset to process.",
    )
    parser.add_argument(
        "--dim", type=int, default=128, help="Embedding dimension."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--min-count", type=int, default=5, help="Min frequency of items to include."
    )
    args = parser.parse_args()

    # 1. Load Data
    processed_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
    baskets_path = processed_dir / "baskets.parquet"
    output_path = processed_dir / f"word2vec_dim{args.dim}.kv"

    print(f"Loading baskets from: {baskets_path}")
    df = pl.read_parquet(baskets_path)

    # 2. Group Items into Baskets
    print("Grouping items into baskets...")
    baskets_df = df.group_by(["user_id", "order_idx"]).agg(
        pl.col("item_id").alias("basket")
    )
    basket_sequences = baskets_df["basket"].to_list()
    print(f"Found {len(basket_sequences)} baskets.")
    # --- New Change: Dynamically set window size ---
    max_basket_len = baskets_df["basket"].list.len().max()
    print(f"Max basket length is {max_basket_len}. Setting window size to this value.")
    # --- End of Change ---

    # 3. Train Embeddings
    print(f"Training Word2Vec model with dim={args.dim}, epochs={args.epochs}...")
    trained_vectors = Word2VecTrainer.train(
        basket_sequences=basket_sequences,
        dim=args.dim,
        window=max_basket_len,  # Use the calculated max length
        epochs=args.epochs,
        min_count=args.min_count,
        workers=-1,
    )
    print("Training complete.")

    # 4. Save Embeddings
    print(f"Saving trained vectors to: {output_path}")
    trained_vectors.save(str(output_path))
    print("Done.")


if __name__ == "__main__":
    main()