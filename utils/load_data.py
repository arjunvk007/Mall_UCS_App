import pandas as pd

def load_data(path="mall_customers.csv"):
    return pd.read_csv(path)
