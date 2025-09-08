import pandas as pd

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["ObservationDate"] = pd.to_datetime(df["ObservationDate"])
    df = df.fillna(0) # Fill NaN values with 0 for numerical columns
    return df

if __name__ == "__main__":
    df = load_and_preprocess_data("covid.csv")
    print(df.head())
    print(df.info())
    print(df.describe())

