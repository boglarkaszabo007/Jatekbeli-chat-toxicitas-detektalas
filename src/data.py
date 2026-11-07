import os
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def load_data(file_path: str) -> pd.DataFrame:
    """
    Betölti a CONDA datasetet (feltételezve, hogy CSV formátumú).
    Elvárt oszlopok: matchId, utterance (vagy text), label (vagy explicit/implicit/action/others).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"A megadott fájl nem található: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Adatok sikeresen betöltve. Sorok száma: {len(df)}, Egyedi meccsek száma: {df['matchId'].nunique()}")
    return df

def clean_text(text: str) -> str:
    """
    Alapvető szövegtisztítás: kisbetűssé alakítás, felesleges whitespace-ek eltávolítása.
    Megjegyzés: A szlenges kifejezéseket, internetes szlenget és emojikat érdemes megtartani,
    mivel a toxicitás szempontjából ezek kritikus jelek lehetnek.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip()
    return text

def split_data_by_match(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
    """
    Meccs alapú (matchId) szétosztás train / val / test halmazokra adatszivárgás ellen.
    A test_size és val_size a teljes adathoz képest értendő arányok.
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['matchId']))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    

    relative_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['matchId']))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)
    
    print(f"Train halmaz: {len(df_train)} sor ({df_train['matchId'].nunique()} meccs)")
    print(f"Validation halmaz: {len(df_val)} sor ({df_val['matchId'].nunique()} meccs)")
    print(f"Test halmaz: {len(df_test)} sor ({df_test['matchId'].nunique()} meccs)")
    
    return df_train, df_val, df_test

if __name__ == "__main__":
    pass