import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

# Import dataset
print("Loading dataset...")
dataset = pd.read_csv(r'C:\Users\2026\Desktop\ML\DataSet\50_Startups.csv')
print(f"Dataset loaded: {dataset.shape}")

# Features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(f"X shape: {X.shape}, y shape: {y.shape}")

# One Hot Encoding
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Display predicted vs actual
np.set_printoptions(precision=2)

print(
    np.concatenate(
        (y_pred.reshape(len(y_pred), 1),
         y_test.reshape(len(y_test), 1)),
        axis=1
    )
)