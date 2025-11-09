import pandas as pd

df = pd.read_csv("solubility_comparison.csv")
df = df[["SMILES", "Chemical name", "LogS exp (mol/L)"]]
df.to_csv("test_out.csv", index=False)
print(df.head())