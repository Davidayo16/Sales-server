import pandas as pd
import numpy as np
import joblib
import random
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Generate a robust dataset with relevant features
def generate_sales_data():
    # Random date range
    start_date = pd.to_datetime("2020-01-01")
    end_date = pd.to_datetime("2020-12-31")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Random data for features
    day_of_week = dates.dayofweek
    month = dates.month
    holiday_indicator = [random.choice([0, 1]) for _ in range(len(dates))]
    product_categories = ["Electronics", "Furniture", "Groceries"]
    regions = ["North", "South", "East"]
    weather_conditions = ["Sunny", "Cloudy", "Rainy"]
    economic_indicator = np.random.uniform(0.8, 1.2, len(dates))  # Simulating an economic factor

    # Creating the dataset
    data = {
        "Date": dates,
        "Day of Week": day_of_week,
        "Month": month,
        "Holiday Indicator": holiday_indicator,
        "Product Category": [random.choice(product_categories) for _ in range(len(dates))],
        "Region": [random.choice(regions) for _ in range(len(dates))],
        "Advertising Spend": np.random.randint(100, 1000, len(dates)),
        "Discount (%)": np.random.randint(0, 50, len(dates)),
        "Stock Levels": np.random.randint(50, 500, len(dates)),
        "Weather Condition": [random.choice(weather_conditions) for _ in range(len(dates))],
        "Economic Indicator": economic_indicator,
        "Sales": None  # Target variable to be calculated
    }

    df = pd.DataFrame(data)

    # Calculate sales based on some assumptions (This is a simplified version of actual sales calculation)
    df['Sales'] = (df['Advertising Spend'] * 0.4) + (df['Discount (%)'] * 2) + (df['Stock Levels'] * 0.3) + \
                  (df['Economic Indicator'] * 100) + np.random.normal(0, 100, len(dates))  # Add noise

    return df


# Preprocess the dataset
def preprocess_data(df):
    # One-hot encoding categorical variables
    df = pd.get_dummies(df, columns=['Product Category', 'Region', 'Weather Condition'], drop_first=True)

    # Dropping the 'Date' column as it's not needed for model training
    df.drop(['Date'], axis=1, inplace=True)

    # Splitting the data into features and target variable
    X = df.drop('Sales', axis=1)
    y = df['Sales']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, y_train, X_test, y_test, model, params=None):
    if params:
        grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)  # Use GridSearchCV for hyperparameter tuning
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        best_model = model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best Model: {best_model}")
    print(f"RMSE: {rmse:.2f}")

    # Feature Importance (if applicable)
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = best_model.feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("\nFeature Importances:")
        print(importance_df)

        plt.figure(figsize=(10, 6))
        plt.bar(importance_df['Feature'], importance_df['Importance'])
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

    # Plotting predictions vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Sales', color='blue')
    plt.plot(y_pred, label='Predicted Sales', color='red', linestyle='--')
    plt.legend()
    plt.title('Sales Forecasting: Actual vs Predicted')
    plt.show()

    return best_model, rmse  # Return the trained model and RMSE


def predict_sales_for_day(model, day_features, feature_columns, scaler):
    day_data = pd.DataFrame([day_features])

    # 1. Ensure day_data has the same columns and order as X_train
    day_data = day_data[feature_columns]  # Reorder and select columns

    # 2. Drop the 'Date' column (if it exists)
    if 'Date' in day_data.columns:
        day_data = day_data.drop('Date', axis=1)

    # 3. Scale the input features
    day_data_values = day_data.values
    day_features_scaled = scaler.transform(day_data_values)
    day_features_scaled_df = pd.DataFrame(day_features_scaled, columns = day_data.columns)

    # 4. Reindex to handle potential missing categorical columns
    aligned_day_data = day_features_scaled_df.reindex(columns=feature_columns, fill_value=0)

    prediction = model.predict(aligned_day_data)
    return prediction[0]


# Generate dataset
df = generate_sales_data()

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(df)

# Scale numerical features (important for many models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)


# 1. Linear Regression
print("Linear Regression:")
linear_model, linear_rmse = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, LinearRegression())

# 2. Ridge Regression (L1 regularization)
print("\nRidge Regression:")
ridge_params = {'alpha': np.logspace(-5, 5, 20)} # Range of alpha values to try
ridge_model, ridge_rmse = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, Ridge(), ridge_params)


# 3. Lasso Regression (L2 regularization)
print("\nLasso Regression:")
lasso_params = {'alpha': np.logspace(-5, 5, 20)} # Range of alpha values to try
lasso_model, lasso_rmse = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, Lasso(), lasso_params)


# 4. Random Forest Regressor
print("\nRandom Forest Regressor:")
rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_model, rf_rmse = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, RandomForestRegressor(random_state=42), rf_params)

# 5. Gradient Boosting Regressor
print("\nGradient Boosting Regressor:")
gb_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5], 'max_depth': [3, 5, 7]}
gb_model, gb_rmse = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test, GradientBoostingRegressor(random_state=42), gb_params)


# Choose the best model (based on lowest RMSE)
best_rmse = min(linear_rmse, ridge_rmse, lasso_rmse, rf_rmse, gb_rmse)
if best_rmse == linear_rmse:
    best_model = linear_model
elif best_rmse == ridge_rmse:
    best_model = ridge_model
elif best_rmse == lasso_rmse:
    best_model = lasso_model
elif best_rmse == rf_rmse:
    best_model = rf_model
else:
    best_model = gb_model

print(f"\nBest Model Overall: {type(best_model).__name__} (RMSE: {best_rmse:.2f})")

joblib.dump(best_model, 'best_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Example: Predict with the best model

best_model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

day_features = {
    "Date": pd.to_datetime("2020-02-07"),  # Include the date in day_features
    "Day of Week": 4,
    "Month": 2,
    "Holiday Indicator": 0,
    "Advertising Spend": 700,
    "Discount (%)": 20,
    "Stock Levels": 300,
    "Economic Indicator": 1.1,
    "Product Category_Electronics": 1,
    "Product Category_Furniture": 0,
    "Product Category_Groceries": 0,
    "Region_North": 1,
    "Region_South": 0,
    "Region_East": 0,
    "Weather Condition_Cloudy": 1,
    "Weather Condition_Rainy": 0,
    "Weather Condition_Sunny": 0,
}
predicted_sales = predict_sales_for_day(best_model, day_features, X_train_scaled.columns, scaler)
print(f"Predicted Sales for the day (using {type(best_model).__name__}): {predicted_sales:.2f}")
