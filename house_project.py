import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import skew
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
skewed_feats = X.apply(lambda x: skew(x)).sort_values(ascending=False)
skewed_cols = skewed_feats[abs(skewed_feats) > 0.75].index
X[skewed_cols] = np.log1p(X[skewed_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

