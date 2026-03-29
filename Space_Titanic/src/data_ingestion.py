import pandas as pd

def load_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test