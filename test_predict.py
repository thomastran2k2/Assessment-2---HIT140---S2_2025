#  Predictive Modelling

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from Data_Analysis_Dataset_2 import clean_dataset2

# Load merged dataset
merged_data = pd.read_csv("merged_dataset.csv")
df2 = clean_dataset2(show_raw_dataset2=False)

# Filter for Summer and Winter only
summer_data = merged_data[merged_data["season_x"]== "summer" ]
winter_data = merged_data[merged_data["season_x"]== "winter" ]

# Define Target and Predictor Variables
target_variable = "bat_landing_to_food"
predictors = [
    "seconds_after_rat_arrival",
    "risk",
    "reward",
    "food_availability",
    "rat_minutes",
    "rat_arrival_number"
]

# Handle missing values for relevant columns only
# summer_data = summer_data.dropna(subset=predictors + [target_variable])
# winter_data = winter_data.dropna(subset=predictors + [target_variable])

print(f"Summer dataset samples: {summer_data.shape[0]}")
print(f"Winter dataset samples: {winter_data.shape[0]}")


# Check multicollinearity by calculating VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# Function to run all models for each season
def model_execution(df, season_name, target_variable=target_variable, predictors=predictors):
    print(f" {season_name.upper()} RESULTS")


    X = df[predictors]
    y = df[target_variable]

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

#     # Correlation Analysis
#     correlation_result = df[predictors + [target_variable]].corr()[target_variable].sort_values(ascending=False)
#     print("\nCorrelation of predictors with target variable:\n")
#     print(correlation_result)

#     # Check collinearity between predictors
#     vif = calculate_vif(X_train)
#     print("\n VIF result:")
#     print(vif)

#     # Correlation matrix between predictors
#     print("\nCorrelation matrix between predictors:\n")
#     print(X_train[predictors].corr())

#     # Keep predictors having VIF < 5
#     selected_predictors = vif[vif["VIF"] < 5]["Variable"].tolist()
#     print(f"\nPredictors retained for MLR (VIF < 5.0): {selected_predictors}")

#     # Baseline Model
#     y_prediction_baseline = np.repeat(y_test.mean(), len(y_test))
#     baseline_mae = mean_absolute_error(y_test, y_prediction_baseline)
#     baseline_mse = mean_squared_error(y_test, y_prediction_baseline)
#     baseline_rmse = np.sqrt(baseline_mse)
#     baseline_r2 = r2_score(y_test, y_prediction_baseline)

#     print("\nBaseline Model Metrics:")
#     print(f"MAE: {baseline_mae:.3f}")
#     print(f"MSE: {baseline_mse:.3f}")
#     print(f"RMSE: {baseline_rmse:.3f}")
#     print(f"R-squared: {baseline_r2:.3f}")

    # Simple Linear Regression (SLR)
    slr_results = []
    for feature in predictors:
        slr_model = LinearRegression()
        slr_model.fit(X_train[[feature]], y_train)
        y_pred = slr_model.predict(X_test[[feature]])

        slr_mae = mean_absolute_error(y_test, y_pred)
        slr_mse = mean_squared_error(y_test, y_pred)
        slr_rmse = np.sqrt(slr_mse)
        slr_r2 = r2_score(y_test, y_pred)
        coef = slr_model.coef_[0]
        intercept = slr_model.intercept_
        slr_results.append([feature, slr_mae, slr_mse, slr_rmse, slr_r2, coef, intercept])

    slr_data = pd.DataFrame(slr_results, columns=['Feature', 'MAE', 'MSE', 'RMSE', 'R2', 'Coefficient', 'Intercept'])
    print("\nSimple Linear Regression (SLR) Results:")
    print(slr_data)

#     # Multiple Linear Regression using Scikit-learn
#     mlr = LinearRegression()
#     mlr.fit(X_train[selected_predictors], y_train)
#     y_pred_mlr = mlr.predict(X_test[selected_predictors])

#     mlr_mae = mean_absolute_error(y_test, y_pred_mlr)
#     mlr_mse = mean_squared_error(y_test, y_pred_mlr)
#     mlr_rmse = np.sqrt(mlr_mse)
#     mlr_r2 = r2_score(y_test, y_pred_mlr)

#     print("\nMLR (Scikit-learn) Metrics:")
#     print(f"MAE: {mlr_mae:.3f}")
#     print(f"MSE: {mlr_mse:.3f}")
#     print(f"RMSE: {mlr_rmse:.3f}")
#     print(f"R-squared: {mlr_r2:.3f}")

#     # OLS Regression
#     X_ols = sm.add_constant(X_train[selected_predictors])
#     ols_model = sm.OLS(y_train, X_ols).fit()
#     print("\nOLS Regression Summary:")
#     print(ols_model.summary())

#     # Comparison Summary
#     comparison = pd.DataFrame({
#         'Model': ['Baseline', 'Best SLR', 'MLR (Scikit)', 'OLS'],
#         'R-squared': [
#             baseline_r2,
#             slr_data['R2'].max(),
#             mlr_r2,
#             ols_model.rsquared
#         ],
#         'MAE': [baseline_mae, slr_data['MAE'].min(), mlr_mae, np.nan],
#         'RMSE': [baseline_rmse, slr_data['RMSE'].min(), mlr_rmse, np.nan]
#     })

#     print("\nModel Comparison Summary:")
#     print(comparison)

#     return {
#         'Season': season_name,
#         'Baseline_R2': baseline_r2,
#         'Best_SLR_R2': slr_data['R2'].max(),
#         'MLR_R2': mlr_r2,
#         'OLS_R2': ols_model.rsquared
#     }


# Run models for both season
if __name__ == "__main__":
    summer_results_bat_landing_to_food = model_execution(summer_data, "Summer")
    winter_results_bat_landing_to_food = model_execution(winter_data, "Winter")
    summer_results_bat_landing_number = model_execution(df2[df2["season"]== "summer" ], "Summer", target_variable="bat_landing_number", predictors=["rat_arrival_number", "food_availability", "hours_after_sunset"])
    winter_results_bat_landing_number = model_execution(df2[df2["season"]== "winter" ], "Winter", target_variable="bat_landing_number", predictors=["rat_arrival_number", "food_availability", "hours_after_sunset"])
    

    # # Final comparison between seasons
    # final_comparison = pd.DataFrame([summer_results, winter_results])
    # print("FINAL MODEL PERFORMANCE COMPARISON (SUMMER vs WINTER): ")
    # print(final_comparison)

