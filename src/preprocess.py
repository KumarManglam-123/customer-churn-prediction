import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # 🔹 clean column names (remove spaces)
    df.columns = df.columns.str.strip()

    # drop customerID if present
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # convert TotalCharges to numeric if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.dropna(inplace=True)

    # convert target to 0/1
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # one-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df
