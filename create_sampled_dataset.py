"""
One-time script to create a sampled flights CSV using reservoir sampling.

This reads all original yearly CSVs from the `data/` folder, takes
100,000 random rows from each using reservoir sampling (memory efficient),
concatenates them and saves a single combined sampled CSV back to `data/`.

Run from the project root:

    python create_sampled_dataset.py
"""

import os
import random
from io import StringIO
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILENAME = "sampled_flights_2018_2022.csv"
ROWS_PER_FILE = 100_000
RANDOM_STATE = 42


def reservoir_sample_csv(file_path: Path, k: int, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Memory-efficient random sampling from a large CSV using reservoir sampling.
    """
    random.seed(random_state)
    sample_lines: list[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for i, line in enumerate(f):
            if i < k:
                sample_lines.append(line)
            else:
                j = random.randint(0, i)
                if j < k:
                    sample_lines[j] = line

    csv_buffer = StringIO(header + "".join(sample_lines))
    return pd.read_csv(csv_buffer)


def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    csv_files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and f != OUTPUT_FILENAME and f.startswith("Combined")
    )

    if not csv_files:
        raise RuntimeError(f"No source CSV files found in {DATA_DIR}")

    print(f"Found {len(csv_files)} source CSV files in {DATA_DIR}:")
    for name in csv_files:
        print(f"  - {name}")

    dfs: list[pd.DataFrame] = []

    for filename in csv_files:
        file_path = DATA_DIR / filename
        print(f"\nSampling {ROWS_PER_FILE:,} rows from {file_path} ...")
        df_temp = reservoir_sample_csv(file_path, ROWS_PER_FILE, RANDOM_STATE)
        print(f"  -> sampled shape: {df_temp.shape}")
        dfs.append(df_temp)

    combined = pd.concat(dfs, ignore_index=True)

    # Ensure FlightDate is parsed to datetime in the saved CSV
    if "FlightDate" in combined.columns:
        combined["FlightDate"] = pd.to_datetime(combined["FlightDate"])

    output_path = DATA_DIR / OUTPUT_FILENAME
    print(f"\nSaving combined sampled dataset to {output_path} ...")
    combined.to_csv(output_path, index=False)
    print(f"Done. Final shape: {combined.shape}")


if __name__ == "__main__":
    main()


