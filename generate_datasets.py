"""
Generate train/test datasets with a balanced training set (oversampling delayed flights)
and a real-distribution test set.

Outputs:
- data/test_real.csv          -> real distribution, same as current sampled dataset
- data/train_balanced.csv     -> base real dataset + extra delayed flights sampled
"""

import os
import random
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import prepare_flight_features, get_model_dataset


DATA_DIR = Path("data")
OUTPUT_TEST = DATA_DIR / "test_real.csv"
OUTPUT_TRAIN = DATA_DIR / "train_balanced.csv"

# Train fraction (e.g., 0.6 => 60% train, 40% test)
TRAIN_FRACTION = 0.6

# Raw yearly files to draw additional delayed samples from
RAW_FILES = [
    DATA_DIR / "Combined_Flights_2018.csv",
    DATA_DIR / "Combined_Flights_2019.csv",
    DATA_DIR / "Combined_Flights_2020.csv",
    DATA_DIR / "Combined_Flights_2021.csv",
    DATA_DIR / "Combined_Flights_2022.csv",
]


def reservoir_sample_delayed(
    files: List[Path],
    n_samples: int,
    chunksize: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reservoir-sample delayed flights (>15 min) across multiple CSV files.
    Returns a DataFrame of size up to n_samples.
    """
    rng = random.Random(random_state)
    reservoir = []
    total_seen = 0

    if n_samples <= 0:
        return pd.DataFrame()

    for file_path in files:
        if not file_path.exists():
            continue
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            delayed_chunk = chunk[chunk.get("ArrDelayMinutes", 0) > 15]
            for _, row in delayed_chunk.iterrows():
                total_seen += 1
                if len(reservoir) < n_samples:
                    reservoir.append(row)
                else:
                    j = rng.randint(0, total_seen - 1)
                    if j < n_samples:
                        reservoir[j] = row
            if len(reservoir) >= n_samples:
                # We can continue to keep randomness, but this early break is fine for speed.
                continue
    return pd.DataFrame(reservoir)


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading base dataset (real distribution) for split...")
    df_base = get_model_dataset()  # already preprocessed, non-cancelled/diverted, target included

    # Stratified split to keep real distribution in test
    target = "IsArrDelayed"
    df_train_base, df_test = train_test_split(
        df_base,
        test_size=1 - TRAIN_FRACTION,
        stratify=df_base[target],
        random_state=42,
    )

    print(f"Saving test set to {OUTPUT_TEST} (rows: {len(df_test):,})")
    df_test.to_csv(OUTPUT_TEST, index=False)

    # Determine class counts in training portion
    pos = (df_train_base[target] == 1).sum()
    neg = (df_train_base[target] == 0).sum()
    print(f"Base train counts -> delayed: {pos:,}, on-time: {neg:,} (rate={pos/len(df_train_base):.3f})")

    # Desired balance: aim for 1:1 by adding delayed flights to train only
    desired_pos = neg
    needed_pos = max(0, desired_pos - pos)
    print(f"Sampling additional delayed flights needed: {needed_pos:,}")

    extra_df_processed = pd.DataFrame()
    if needed_pos > 0:
        print("Sampling delayed flights from raw yearly files...")
        extra_raw = reservoir_sample_delayed(RAW_FILES, n_samples=needed_pos, random_state=42)
        if not extra_raw.empty:
            print(f"Sampled {len(extra_raw):,} delayed rows; applying feature engineering...")
            extra_df_processed = prepare_flight_features(extra_raw)
            # Keep same columns as base
            keep_cols = df_base.columns
            extra_df_processed = extra_df_processed[keep_cols.intersection(extra_df_processed.columns)]
            # Ensure target present and delayed
            extra_df_processed = extra_df_processed[extra_df_processed[target] == 1]
            # Drop rows with missing target or critical cols
            extra_df_processed = extra_df_processed.dropna(subset=[target, "Airline", "Origin", "Dest"])
        else:
            print("Warning: No extra delayed rows sampled.")

    # Build balanced train set
    if extra_df_processed.empty:
        df_train = df_train_base.copy()
        print("No extra delayed samples added; train set equals base split.")
    else:
        df_train = pd.concat([df_train_base, extra_df_processed], ignore_index=True)
        print(f"Train set size after adding delayed samples: {len(df_train):,}")

    print(f"Saving train set to {OUTPUT_TRAIN}")
    df_train.to_csv(OUTPUT_TRAIN, index=False)

    print("Done.")


if __name__ == "__main__":
    main()

