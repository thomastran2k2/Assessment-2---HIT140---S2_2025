# Model Optimisation and Validation 
# Perform VIF, different train-test splits, and 5-fold cross-validation

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Import cleaned dataset 1 and 2
from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2

# Load each dataset separately
Dataset_1 = clean_dataset_1(show_raw_dataset1=False)
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

Dataset_2['season'] = Dataset_2["month"].apply(lambda x: 'winter' if x in [1, 2, 11, 12] else 'summer')

# Filter for Summer and Winter
summer_df1 = Dataset_1[Dataset_1["season"].str.lower() == "summer"]
winter_df1 = Dataset_1[Dataset_1["season"].str.lower() == "winter"]

summer_df2 = Dataset_2[Dataset_2["season"].str.lower() == "summer"]
winter_df2 = Dataset_2[Dataset_2["season"].str.lower() == "winter"]


# Function to check and remove multicollinearity (VIF)
def remove_multicollinearity(df, predictors, threshold=5.0):
    X = df[predictors].dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)
    retained = vif_data[vif_data["VIF"] < threshold]["Variable"].tolist()
    retained = [v for v in retained if v != "const"]
    print(f"\nPredictors retained (VIF < {threshold}): {retained}")
    return retained


# Function for model optimisation (train-test splits and k-fold)
def model_optimisation(df, target, predictors, season, dataset_name):
    print(f"\n MODEL OPTIMISATION RESULTS ({dataset_name.upper()} - {season.upper()})")


    predictors_final = remove_multicollinearity(df, predictors)
    X = df[predictors_final].dropna()
    y = df[target].loc[X.index]

    # Train-test split evaluation
    splits = [0.3, 0.2, 0.4]  
    split_results = []

    for s in splits:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=s, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        split_results.append([f"{int((1-s)*100)}-{int(s*100)}", mae, rmse, r2])

    split_df = pd.DataFrame(split_results, columns=["Split Ratio", "MAE", "RMSE", "R2"])
    print("\nTrain-Test Split Results:")
    print(split_df)

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = LinearRegression()
    cv_r2 = cross_val_score(model, X, y, cv=kf, scoring="r2")
    cv_mae = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))

    print("\n5-Fold Cross Validation Results:")
    print(f"Average R-squared: {cv_r2.mean():.3f}")
    print(f"Average Mean Absolute Error: {cv_mae.mean():.3f}")
    print(f"Average Root Mean Square Error: {cv_rmse.mean():.3f}")


# DATASET 1: Predicting bat_landing_to_food
target_1 = "bat_landing_to_food"
predictors_1 = ["seconds_after_rat_arrival", "risk", "reward", "hours_after_sunset"]

model_optimisation(summer_df1, target_1, predictors_1, "Summer", "Dataset 1")
model_optimisation(winter_df1, target_1, predictors_1, "Winter", "Dataset 1")


# DATASET 2: Predicting bat_landing_number
target_2 = "bat_landing_number"
predictors_2 = ["hours_after_sunset", "food_availability", "rat_minutes", "rat_arrival_number"]

model_optimisation(summer_df2, target_2, predictors_2, "Summer", "Dataset 2")
model_optimisation(winter_df2, target_2, predictors_2, "Winter", "Dataset 2")
