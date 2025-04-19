import pandas as pd

def load_netflix_data(csv_path):
    """
    Loads the Netflix dataset from the given CSV path.
    Returns a pandas DataFrame.
    """
    return pd.read_csv(csv_path)
