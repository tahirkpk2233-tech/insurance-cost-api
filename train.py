import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1)
y = df["charges"]

categorical = ["sex", "smoker", "region"]

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough",
)

model = Pipeline([("pre", pre), ("reg", LinearRegression())])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
print("R2 score:", model.score(X_test, y_test))

joblib.dump(model, "model.joblib")