import pandas as pd
from preprocessing.preprocessing import preprocess_data
from supervised.train import supervised_train
if __name__ == "__main__":

    df_path= "student_data.csv"
    df = pd.read_csv(df_path)
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)
    supervised_train(X_train_scaled, X_test_scaled, y_train, y_test)