# Predictive Modelling
# Dataset 2
# Comparison of how predictors affect bat landing numbers across seasons

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load the Dataset
from Data_Analysis_Dataset_2 import clean_dataset2
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

Dataset_2['season'] = Dataset_2["month"].apply(lambda x: 'winter' if x in [1, 2, 11, 12] else 'summer')

# Divide summer and winter dataset
summer_data = Dataset_2[Dataset_2["season"].str.lower() == "summer"]
winter_data = Dataset_2[Dataset_2["season"].str.lower() == "winter"]

# Define target and predictors variable
target_variable = "bat_landing_number"
predictors = [
   "hours_after_sunset",
    "food_availability",
    "rat_minutes",
    "rat_arrival_number"
]

# Handle missing values in the relevant columns
summer_data = summer_data.dropna(subset=predictors + [target_variable])
winter_data = winter_data.dropna(subset=predictors + [target_variable])

# Check the divided dataset
print(f"Summer dataset samples: {summer_data.shape[0]}")
print(f"Winter dataset samples: {winter_data.shape[0]}",'\n')


# Create a function to execute models for both season
def run_models(df, season_type):
    print(f"{season_type.upper()} RESULTS:")

    X = df[predictors]
    y = df[target_variable]

    # Split data into train and test dataset into 80 and 20 percentage respectively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform correlation analysis
    correlation_result = df[predictors + [target_variable]].corr()[target_variable].sort_values(ascending=False)
    print("\nCorrelation of predictors with target_variable variable (" + season_type + "): \n")
    print(correlation_result)


    # Build baseline Model
    y_prediction_baseline = np.repeat(y_test.mean(), len(y_test))
    baseline_mae = mean_absolute_error(y_test, y_prediction_baseline)
    baseline_mse = mean_squared_error(y_test, y_prediction_baseline)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_test, y_prediction_baseline)

    print("\nResult of Baseline Model Metrics (" + season_type + "):")
    print(f"Mean Absolute Error: {baseline_mae:.3f}")
    print(f"Mean Squared Error: {baseline_mse:.3f}")
    print(f"Root Mean Squared Error: {baseline_rmse:.3f}")
    print(f"R-squared: {baseline_r2:.3f}")


    # Build Simple Linear Regression (SLR)
    slr_results = []
    for feature in predictors:
        model = LinearRegression()
        model.fit(X_train[[feature]], y_train)
        y_prediction_slr = model.predict(X_test[[feature]])

        mae_slr = mean_absolute_error(y_test, y_prediction_slr)
        mse_slr = mean_squared_error(y_test, y_prediction_slr)
        rmse_slr = np.sqrt(mse_slr)   # fixed small typo here (was np.sqrt(mae_slr))
        r2_slr = r2_score(y_test, y_prediction_slr)
        coefficient_slr = model.coef_[0]
        intercept_slr = model.intercept_

        slr_results.append([
            feature, coefficient_slr, intercept_slr,
            mae_slr, mse_slr, rmse_slr, r2_slr
        ])


    # Result of simple linear regression model
    slr_df = pd.DataFrame(
        slr_results,
        columns=["Feature", "Coefficient", "Intercept", "MAE", "MSE", "RMSE", "R2"]
    )
    print("\n Simple Linear Regression Result (" + season_type + "):")
    print(slr_df.to_string(index=False))


    # Build Multiple Linear Regression (MLR)

    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    y_prediction_mlr = mlr.predict(X_test)

    mlr_mae = mean_absolute_error(y_test, y_prediction_mlr)
    mlr_mse = mean_squared_error(y_test, y_prediction_mlr)
    mlr_rmse = np.sqrt(mlr_mse)
    mlr_r2 = r2_score(y_test, y_prediction_mlr)

    print("\nMLR using Scikit-learn Metrics (" + season_type + "):")
    print(f"Intercept: {mlr.intercept_:.4f}")
    print(f"Coefficients:")
    for name, coef in zip(predictors, mlr.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"MAE: {mlr_mae:.3f}")
    print(f"MSE: {mlr_mse:.3f}")
    print(f"RMSE: {mlr_rmse:.3f}")
    print(f"R-squared: {mlr_r2:.3f}")


    # Perform Ordinary Least Squares (OLS)

    X_ols = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_ols).fit()
    print("\nOLS Regression Summary (" + season_type + "):")
    print(ols_model.summary())


    # Summary Comparison
    model_comparison = pd.DataFrame({
        "Model": ["Baseline", "Best SLR", "MLR (Scikit)", "OLS"],
        "R-squared": [
            baseline_r2,
            slr_df["R2"].max(),
            mlr_r2,
            ols_model.rsquared
        ],
        "MAE": [baseline_mae, slr_df["MAE"].min(), mlr_mae, np.nan],
        "RMSE": [baseline_rmse, slr_df["RMSE"].min(), mlr_rmse, np.nan]
    })

    print("\nModel Comparison Summary:")
    print(model_comparison,'\n')

    return {
        "Season": season_type,
        "Baseline_R2": baseline_r2,
        "Best_SLR_R2": slr_df["R2"].max(),
        "MLR_R2": mlr_r2,
        "OLS_R2": ols_model.rsquared
    }


# Run models for both seasons
summer_results = run_models(summer_data, "Summer")
winter_results = run_models(winter_data, "Winter")


# Final Comparison
final_comparison = pd.DataFrame([summer_results, winter_results])
print("FINAL MODEL PERFORMANCE COMPARISON (SUMMER vs WINTER):")
print(final_comparison)
