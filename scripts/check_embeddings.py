import argparse
from pathlib import Path

import polars as pl
from gensim.models.word2vec import KeyedVectors

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Check Word2Vec embeddings.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["instacart", "tafeng", "dunnhumby"],
        help="Name of the dataset to check.",
    )
    parser.add_argument(
        "--dim", type=int, default=128, help="Embedding dimension used for training."
    )
    parser.add_argument(
        "--item-id", type=int, default=100, help="An example re-indexed item ID to check."
    )
    args = parser.parse_args()

    processed_dir = PROJECT_ROOT / "data" / "processed" / args.dataset
    kv_path = processed_dir / f"word2vec_dim{args.dim}.kv"
    items_path = processed_dir / "items.parquet"
    item_map_path = processed_dir / "item_map.parquet"

    # Load vectors
    print(f"Loading vectors from: {kv_path}")
    try:
        vectors = KeyedVectors.load(str(kv_path))
    except FileNotFoundError:
        print(f"Error: File not found at {kv_path}. Did you run train_word2vec.py?")
        return

    # --- Universal Name Lookup Logic ---
    id_to_name_map = {}
    try:
        print("Loading item metadata...")
        items_df = pl.read_parquet(items_path)
        
        # Find the correct name column, with dataset-specific fallbacks
        possible_name_cols = ["product_name", "product_description", "description"]
        if args.dataset == "dunnhumby":
            possible_name_cols.append("prod_code_raw")
        if args.dataset == "tafeng":
            possible_name_cols.append("product_subclass")

        name_col = next((col for col in possible_name_cols if col in items_df.columns), None)

        if not name_col:
            raise ValueError("Could not find a known item name/code column in items.parquet.")
        print(f"Found item name/code column: '{name_col}'")

        # Scenario 1: Mapping file exists (e.g., TaFeng)
        if item_map_path.exists():
            print(f"Found item map file: {item_map_path}. Using it for lookup.")
            item_map_df = pl.read_parquet(item_map_path)
            
            # Corrected: TaFeng's items.parquet uses 'item_id' for the original ID
            original_id_col = "original_item_id" if "original_item_id" in items_df.columns else "item_id"
            
            original_id_to_name = dict(zip(items_df[original_id_col], items_df[name_col], strict=True))
            new_id_to_original_id = dict(zip(item_map_df["item_id"], item_map_df["original_item_id"], strict=True))
            
            for new_id, orig_id in new_id_to_original_id.items():
                id_to_name_map[new_id] = original_id_to_name.get(orig_id, "N/A")

        # Scenario 2: No mapping file (e.g., Instacart, Dunnhumby)
        else:
            print("No item map file found. Assuming direct ID-to-name mapping.")
            id_col = "item_id" if "item_id" in items_df.columns else "original_item_id"
            if id_col not in items_df.columns:
                 raise ValueError(f"Could not find 'item_id' or 'original_item_id' in items.parquet.")
            print(f"Using '{id_col}' from items.parquet for direct lookup.")
            id_to_name_map = dict(zip(items_df[id_col], items_df[name_col], strict=True))

    except Exception as e:
        print(f"Warning: Could not load or parse item names: {e}")
        print("Will proceed with showing item IDs only.")
    # --- End of Logic ---

    print(f"\nVocabulary size: {len(vectors.key_to_index)}")
    print(f"Vector dimension: {vectors.vector_size}")

    item_key = str(args.item_id)

    def get_item_name(item_id_str: str) -> str:
        item_id_int = int(item_id_str)
        return str(id_to_name_map.get(item_id_int, "N/A"))

    if item_key in vectors:
        target_name = get_item_name(item_key)
        print(f"\n--- Sanity Check: Finding items similar to item '{item_key}' ({target_name}) ---")
        similar_items = vectors.most_similar(item_key, topn=10)
        print(f"Most similar items:")
        for item, score in similar_items:
            name = get_item_name(item)
            print(f"  - Item: {item} ({name}), Similarity: {score:.4f}")
    else:
        print(f"\nWarning: Item ID '{item_key}' not in the vocabulary.")
        # Show a random item instead
        random_key = list(vectors.key_to_index.keys())[0]
        random_name = get_item_name(random_key)
        print(
            f"\n--- Showing a random example for item '{random_key}' ({random_name}) instead ---"
        )
        similar_items = vectors.most_similar(random_key, topn=10)
        for item, score in similar_items:
            name = get_item_name(item)
            print(f"  - Item: {item} ({name}), Similarity: {score:.4f}")


if __name__ == "__main__":
    main()