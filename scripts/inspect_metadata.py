"""Throwaway inspection script — look at the raw OASIS-1 metadata before
building the manifest. Not part of the permanent pipeline."""

import pandas as pd

path = "data/metadata/oasis_cross-sectional-5708aa0a98d82080.xlsx"

df = pd.read_excel(path)

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nDtypes:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nCDR value counts:\n", df["CDR"].value_counts(dropna=False))
