import pandas as pd
import os
from src.data import clean_text, split_data_by_match

# 1. Beolvasás a data mappából
df_train_raw = pd.read_csv("data/CONDA_train.csv")
df_valid_raw = pd.read_csv("data/CONDA_valid.csv")
df_test_raw = pd.read_csv("data/CONDA_test.csv")

# 2. Összefűzés
df = pd.concat([df_train_raw, df_valid_raw, df_test_raw], ignore_index=True)
print(f"Sikeres összefűzés! Összesített adatok mérete: {len(df)} sor")

# 3. Szövegtisztítás
df['cleaned_text'] = df['utterance'].apply(clean_text)

# 4. Szigorú, matchId szerinti split (70/15/15)
train_df, val_df, test_df = split_data_by_match(df)

# 5. Mentés fájlokba a data/ mappába
os.makedirs("data", exist_ok=True)
train_df.to_csv("data/train_split.csv", index=False)
val_df.to_csv("data/val_split.csv", index=False)
test_df.to_csv("data/test_split.csv", index=False)

print("\nA szétosztott halmazok sikeresen elmentve:")
print(" - data/train_split.csv")
print(" - data/val_split.csv")
print(" - data/test_split.csv")