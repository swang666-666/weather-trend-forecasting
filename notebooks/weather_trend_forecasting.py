import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------------
# 1. load the data
# -------------------------------

# make sure you download the dataset from Kaggle
# and place "Global Weather Repository.csv" inside the data folder
file_path = "data/Global Weather Repository"

df = pd.read_csv(file_path)

print("Shape of the dataset:")
print(df.shape)
print("\nFirst few rows:")
print(df.head())


# -------------------------------
# 2. basic inspection
# -------------------------------

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(20))


# -------------------------------
# 3. find the time column
# -------------------------------

time_candidates = [col for col in df.columns if "last" in col.lower() or "date" in col.lower() or "time" in col.lower()]
print("\nPossible time-related columns:")
print(time_candidates)

time_col = None
for col in df.columns:
    if "last" in col.lower() and "updated" in col.lower():
        time_col = col
        break

print("\nSelected time column:")
print(time_col)


# -------------------------------
# 4. work with time
# -------------------------------

df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
df = df.dropna(subset=[time_col]).copy()
df = df.sort_values(time_col).reset_index(drop=True)

df["year"] = df[time_col].dt.year
df["month"] = df[time_col].dt.month
df["day"] = df[time_col].dt.day
df["dayofweek"] = df[time_col].dt.dayofweek
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

print("\nTime column converted successfully.")


# -------------------------------
# 5. check temperature and precipitation columns
# -------------------------------

temp_candidates = [col for col in df.columns if "temp" in col.lower()]
precip_candidates = [col for col in df.columns if "precip" in col.lower() or "rain" in col.lower()]

print("\nTemperature-related columns:")
print(temp_candidates)

print("\nPrecipitation-related columns:")
print(precip_candidates)


# choose the main target after checking the real column names
target_col = "temperature_celsius"   # change this if your dataset uses a different name


# -------------------------------
# 6. simple cleaning
# -------------------------------

df = df.drop_duplicates().copy()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\nNumber of numeric columns:")
print(len(numeric_cols))


# -------------------------------
# 7. quick EDA plots
# -------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(df[target_col].dropna(), bins=40, kde=True)
plt.title("Distribution of Temperature")
plt.xlabel(target_col)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("outputs/figures/temperature_distribution.png")
plt.show()


temp_trend = df.groupby(df[time_col].dt.date)[target_col].mean().reset_index()
temp_trend.columns = ["date", "avg_temperature"]

plt.figure(figsize=(12, 5))
plt.plot(temp_trend["date"], temp_trend["avg_temperature"])
plt.title("Average Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Average Temperature")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/figures/temperature_trend.png")
plt.show()


# precipitation plot
precip_col = "precip_mm"   # change this if needed

if precip_col in df.columns:
    precip_trend = df.groupby(df[time_col].dt.date)[precip_col].mean().reset_index()
    precip_trend.columns = ["date", "avg_precip"]

    plt.figure(figsize=(12, 5))
    plt.plot(precip_trend["date"], precip_trend["avg_precip"])
    plt.title("Average Precipitation Over Time")
    plt.xlabel("Date")
    plt.ylabel("Average Precipitation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/figures/precipitation_trend.png")
    plt.show()
else:
    print(f"\nColumn '{precip_col}' not found. Please update the precipitation column name.")


# -------------------------------
# 8. prepare data for forecasting
# -------------------------------

model_df = df.dropna(subset=[target_col]).copy()
model_df = model_df.sort_values(time_col).reset_index(drop=True)

# create a few lag features so the model can use recent temperature history
model_df["temp_lag1"] = model_df[target_col].shift(1)
model_df["temp_lag2"] = model_df[target_col].shift(2)
model_df["temp_lag3"] = model_df[target_col].shift(3)
model_df["temp_roll3"] = model_df[target_col].rolling(3).mean()
model_df["temp_roll7"] = model_df[target_col].rolling(7).mean()

model_df = model_df.dropna().copy()

feature_cols = [
    "year", "month", "day", "dayofweek", "is_weekend",
    "temp_lag1", "temp_lag2", "temp_lag3", "temp_roll3", "temp_roll7"
]

for col in ["humidity", "wind_kph", "pressure_mb", "cloud", "uv"]:
    if col in model_df.columns:
        feature_cols.append(col)

X = model_df[feature_cols]
y = model_df[target_col]

print("\nFeatures used:")
print(feature_cols)


# -------------------------------
# 9. time-based train/test split
# -------------------------------

split_index = int(len(model_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# -------------------------------
# 10. baseline model
# -------------------------------

baseline_pred = X_test["temp_lag1"]

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
baseline_r2 = r2_score(y_test, baseline_pred)

print("\nBaseline model results:")
print("MAE:", baseline_mae)
print("RMSE:", baseline_rmse)
print("R2:", baseline_r2)


# -------------------------------
# 11. random forest model
# -------------------------------

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest results:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2:", rf_r2)


# -------------------------------
# 12. compare results
# -------------------------------

results = pd.DataFrame({
    "Model": ["Baseline (Lag-1)", "Random Forest"],
    "MAE": [baseline_mae, rf_mae],
    "RMSE": [baseline_rmse, rf_rmse],
    "R2": [baseline_r2, rf_r2]
})

print("\nModel comparison:")
print(results)


plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(baseline_pred.values[:100], label="Baseline")
plt.plot(rf_pred[:100], label="Random Forest")
plt.title("Actual vs Predicted Temperature")
plt.xlabel("Test Sample Index")
plt.ylabel("Temperature")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/model_predictions.png")
plt.show()


# -------------------------------
# 13. feature importance
# -------------------------------

importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df.head(10), x="importance", y="feature")
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance.png")
plt.show()


print("\nDone. If any column name error appears, update the target or precipitation column name based on the dataset.")
