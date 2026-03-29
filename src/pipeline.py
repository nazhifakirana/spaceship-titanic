import pandas as pd
import joblib
import os

from train_model import train_model


def run_pipeline():
    print("Loading data...")
    df = pd.read_csv("data/train.csv")

    X = df.drop("Transported", axis=1)
    y = df["Transported"]

    print("Training model...")
    model = train_model(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logreg_pipeline.pkl")

    print(" Model saved!")


if __name__ == "__main__":
    run_pipeline()