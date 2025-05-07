# train_cause_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("accident.csv")

# Drop unnecessary columns
df.drop(columns=['Accident_ID', 'Date', 'Time'], inplace=True)

# Ensure numeric columns are correct
df['Number_of_Deaths'] = pd.to_numeric(df['Number_of_Deaths'], errors='coerce')
df['Number_of_Injuries'] = pd.to_numeric(df['Number_of_Injuries'], errors='coerce')
df['Speed_Limit'] = pd.to_numeric(df['Speed_Limit'], errors='coerce')

# Drop rows with missing target
df = df.dropna(subset=['Reason'])

# Define features and target
X = df.drop("Reason", axis=1)
y = df["Reason"]

# Define numerical and categorical features
numerical_features = ["Number_of_Deaths", "Number_of_Injuries", "Speed_Limit"]
categorical_features = [col for col in X.columns if col not in numerical_features]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

# Full pipeline
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(multi_class='multinomial', max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model_pipeline.fit(X_train, y_train)

# Save the model
joblib.dump(model_pipeline, "cause_model.pkl")
print("✅ Accident cause prediction model saved as 'cause_model.pkl'")
