
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
names = ["train", "validation", "test"]
data = {}

for name in names:
    matches = list(ROOT.glob(f"{name}-*.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {name} Parquet file, found {len(matches)}"
        )

    df = pd.read_parquet(matches[0])
    data[name] = df

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} SET")
    print("Rows:", len(df))
    print("Label counts:")
    print(df["label"].value_counts().sort_index().to_dict())
    print("Category counts:")
    print(df["category"].value_counts().to_string())
    print("Duplicate texts:", df["text"].duplicated().sum())
    print("Empty texts:", df["text"].astype(str).str.strip().eq("").sum())

    assert set(df["label"].unique()).issubset({0, 1})
    assert df["text"].notna().all()
    assert df["label"].notna().all()

print("\nGROUP OVERLAP CHECK")
group_sets = {
    name: set(df["group_id"].dropna())
    for name, df in data.items()
}

for i, first in enumerate(names):
    for second in names[i + 1:]:
        overlap = group_sets[first] & group_sets[second]
        print(f"{first} vs {second}: {len(overlap)} shared group IDs")

print("\nAUDIT COMPLETE")
