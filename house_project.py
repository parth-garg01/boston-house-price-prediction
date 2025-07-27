import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
even_rows = raw_df.iloc[::2].reset_index(drop=True)
odd_rows = raw_df.iloc[1::2, :3].reset_index(drop=True)
full_df = pd.concat([even_rows, odd_rows], axis=1)
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    "MEDV" 
]
full_df.columns = column_names
X = full_df.drop("MEDV", axis=1) 
y = full_df["MEDV"]      
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X shape:", X.shape)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)





