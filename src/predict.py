import joblib
import pandas as pd
from preprocess import load_data


model = joblib.load("model.joblib")

df = load_data("data/churn.csv")
X = df.drop("Churn", axis=1)

sample = X.iloc[:5]
predictions = model.predict(sample)

print("Predictions:", predictions)