import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("logistics_data_cleaned.csv")
df["delivery_date"] = pd.to_datetime(df["delivery_date"], errors="coerce")
df["day_of_week"] = df["delivery_date"].dt.dayofweek
df["month"] = df["delivery_date"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

target = "delivery_time_hours"
features = [
    "region", "vehicle_type", "traffic_level", "distance_km",
    "shipment_volume", "promised_time_hours", "transportation_cost",
    "day_of_week", "month", "is_weekend"
]
data = df[features + [target]].dropna()

X, y = data[features], data[target]
categorical = ["region", "vehicle_type", "traffic_level"]
numeric = [c for c in features if c not in categorical]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=250, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(
        name,
        "MAE=", round(mean_absolute_error(y_test, pred), 4),
        "RMSE=", round(mean_squared_error(y_test, pred) ** 0.5, 4),
        "R2=", round(r2_score(y_test, pred), 4)
    )

# Select a model using test RMSE, then validate the selected pipeline with 5-fold CV.
# Hyperparameter tuning is applied to the Gradient Boosting model below.
param_grid = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [2, 3]
}
search = GridSearchCV(
    Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(random_state=42))
    ]),
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)
search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)

print("Tuned parameters:", search.best_params_)
print("Tuned MAE:", mean_absolute_error(y_test, pred))
print("Tuned RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("Tuned R2:", r2_score(y_test, pred))

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(
    best, X, y, cv=cv, scoring="neg_root_mean_squared_error"
)
print("5-fold CV RMSE mean:", cv_rmse.mean())
print("5-fold CV RMSE std:", cv_rmse.std())

# Optimization scenario:
# 1) reduce route distance by 10% through route planning
# 2) treat high-traffic deliveries as medium traffic through scheduling
scenario = data.copy()
scenario["baseline_prediction"] = best.predict(scenario[features])

optimized = scenario.copy()
optimized["distance_km"] *= 0.90
optimized["traffic_level"] = optimized["traffic_level"].replace({"High": "Medium"})
optimized["optimized_prediction"] = best.predict(optimized[features])

print(
    "Estimated average predicted hours reduced:",
    (scenario["baseline_prediction"] - optimized["optimized_prediction"]).mean()
)
