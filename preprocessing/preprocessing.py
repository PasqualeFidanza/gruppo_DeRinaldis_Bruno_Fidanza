import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:

    # Esplorazione dati
    print(f"le prime cinque righe sono: {df.head()}")
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(f"la forma del DataFrame è: {df.shape}")

    # Pre-processing data
    le = LabelEncoder()
    df = df.dropna()  
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    
    # Normalizzazione
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))

    # check 
    print(df.head())

    # matrice di correlazione
    corr = df.corr()
    corr_target = corr['G3'].sort_values(ascending=False)
    print(corr_target)
    #seleziono quelle più correlate
    strong_corr = corr_target[abs(corr_target) > 0.2]
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[strong_corr.index].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Matrice di Correlazione")
    plt.show()

    return df
