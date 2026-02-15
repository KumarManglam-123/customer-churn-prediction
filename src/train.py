import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv("data/Churn.csv")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# Drop customerID
df = df.drop("customerID", axis=1)

# Convert target to 0/1
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 🔹 SAVE model and columns (VERY IMPORTANT)
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Model and columns saved successfully!")
