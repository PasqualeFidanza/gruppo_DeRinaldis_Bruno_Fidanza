import pandas as pd
from preprocessing.preprocessing import preprocess_data

if __name__ == "__main__":

    df_path= "student_data.csv"
    df = pd.read_csv(df_path)
    processed_df = preprocess_data(df)