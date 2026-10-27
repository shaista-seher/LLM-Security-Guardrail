
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

for file in sorted(ROOT.glob("*.parquet")):
    print("\n" + "=" * 60)
    print("FILE:", file.name)

    df = pd.read_parquet(file)

    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))
    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    for col in df.columns:
        name = str(col).lower()
        if any(word in name for word in
               ["label", "class", "target", "injection"]):
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts(dropna=False).to_string())
