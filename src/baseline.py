import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def run_baseline():
    print("Baseline modell futtatása az új, szétosztott adatokon...")
    
    # 1. Beolvasás az új splitelt fájlokból
    train_df = pd.read_csv("data/train_split.csv")
    val_df = pd.read_csv("data/val_split.csv")
    
    # NaN-ek eltávolítása a címkékből és a szövegekből
    train_df = train_df.dropna(subset=['intentClass', 'cleaned_text'])
    val_df = val_df.dropna(subset=['intentClass', 'cleaned_text'])
    
    X_train_text = train_df['cleaned_text'].astype(str)
    X_val_text = val_df['cleaned_text'].astype(str)
    
    y_train = train_df['intentClass']
    y_val = val_df['intentClass']
    
    # 2. TF-IDF Vectorizer (ngram_range 1,2 a szlenghez és kifejezésekhez)
    print("TF-IDF features generálása...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    
    # 3. Logisztikus Regresszió tanítása
    print("Logisztikus regresszió tanítása...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Értékelés a validációs halmazon
    y_pred = clf.predict(X_val)
    
    print("\n--- VALIDÁCIÓS EREDMÉNYEK (Baseline) ---")
    print(classification_report(y_val, y_pred))
    
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    print(f"Validation Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    run_baseline()