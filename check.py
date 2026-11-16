import pandas as pd
df = pd.read_csv("dna_methylation.csv")
print(df.head())
print(df.columns.tolist(), len(df))
seq = [c for c in df.columns if c.lower() in ("sequence", "seq")][0]
lab = [c for c in df.columns if c.lower() in ("label", "class", "y")][0]
print(df[lab].value_counts())
print("Duplicate sequences:", df[seq].duplicated().sum())
print("Sequences with both labels:", (df.groupby(seq)[lab].nunique() > 1).sum())
print("Length counts:", df[seq].str.len().value_counts().head())
print(df.groupby(lab)[seq].apply(lambda s: s.str.count("CG").mean()))