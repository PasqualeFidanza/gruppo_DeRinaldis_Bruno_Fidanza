import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

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


    # PCA
    X = df.drop(columns=['G3'])  # Escludo la colonna target
    y = df['G3']
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    print(f"Varianza spiegata dalle componenti principali: {explained_variance}")

    # X_pca df
    X_pca = pd.DataFrame(
        data=X_pca,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
        )
    X_pca['G3'] = y.values
    # check df
    print(f"Shape of PCA DataFrame: {X_pca.shape}")
    print(X_pca.head())

    return X_pca
