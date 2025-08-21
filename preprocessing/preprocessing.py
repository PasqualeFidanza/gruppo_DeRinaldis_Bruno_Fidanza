import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple

def preprocess_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, pass_mark: int = 10
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Preprocess data for binary classification using G3 as target.
    Suitable for Random Forest (no PCA or scaling needed).
    
    Args:
        df: Input DataFrame (must contain 'G3' column)
        test_size: Proportion of dataset for test split
        random_state: Random seed for reproducibility
        pass_mark: Threshold for passing grade (default 10)
    
    Returns:
        X_train, X_test, y_train, y_test: Cleaned and encoded training and test sets
    """
    
    print(f"Le prime cinque righe sono:\n{df.head()}")
    print(f"\nInfo del DataFrame:")
    print(df.info())
    print(f"\nValori mancanti:\n{df.isnull().sum()}")
    print(f"\nLa forma del DataFrame è: {df.shape}")
    df_clean = df.dropna().copy()
    print(f"\nDopo rimozione valori mancanti: {df_clean.shape}")
    
        
    X = df_clean.drop(columns=['G3'])
    y = (df_clean['G3'] >= pass_mark).astype(int)  # binary target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Distribuzione classi nel train set:\n{np.bincount(y_train)}")
    
    # Encode categorical features
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    
    if categorical_cols:
        print(f"\nColonne categoriche da codificare: {categorical_cols}")
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            label_encoders[col] = le
            
            X_test_col_str = X_test[col].astype(str)
            mask = X_test_col_str.isin(le.classes_)
            X_test[col] = 0  # unseen categories → 0
            X_test.loc[mask, col] = le.transform(X_test_col_str[mask])
    
    
    return X_train, X_test, y_train, y_test
