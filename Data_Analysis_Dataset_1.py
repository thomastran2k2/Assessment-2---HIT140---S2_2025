# Assignment 2: Investigation A Part
# Data Analysis for dataset 1

# Import all the necessary libraries
import pandas as pd
import numpy as np
import re


def clean_dataset_1(show_raw_dataset1=True):
    # Read the dataset 1
    Dataset_1 = pd.read_csv("dataset1.csv")

    if show_raw_dataset1:
        # Checking the first few rows of dataset 1
        print("Displaying first few rows of Dataset 1:")
        print(Dataset_1.head())

        # Check structure and distinct values of the dataset 1
        print("Checking the information of Dataset 1:")
        print(Dataset_1.info(), "\n")

        print("Distinct values in Dataset 1:")
        print(Dataset_1.nunique(), "\n")

        # Check the missing values in each column
        print("Missing values in Dataset 1:")
        print(Dataset_1.isnull().sum(), "\n")

    # Check for duplicates and drop if exists
    duplicate_data = Dataset_1[Dataset_1.duplicated()]
    if show_raw_dataset1:
        print("Duplicate rows in dataset 1:")
        print(duplicate_data)

    Dataset_1 = Dataset_1.drop_duplicates()

    if show_raw_dataset1:
        # Check distinct values for all columns to see whether the column has invalid values
        columns_in_Dataset_1 = [ 
            "start_time", "bat_landing_to_food", "habit", "rat_period_start", "rat_period_end",
            "seconds_after_rat_arrival", "risk", "reward", "month", "sunset_time",
            "hours_after_sunset", "season"
        ]

        for col in columns_in_Dataset_1:
            distinct_values_Dataset_1 = Dataset_1[col].unique()
            print(f"{col}: {len(distinct_values_Dataset_1)} distinct values")
            print(f"{distinct_values_Dataset_1}\n")

    # Convert datatype of time related columns to datetime
    datetime_columns = ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]
    for column in datetime_columns:
        if column in Dataset_1.columns:
            Dataset_1[column] = pd.to_datetime(Dataset_1[column], errors="coerce", dayfirst=True)

    # Convert categorical variables
    categorical_columns = ["habit", "month", "season"]
    for column in categorical_columns:
        if column in Dataset_1.columns:
            Dataset_1[column] = Dataset_1[column].astype("category")
    Dataset_1.loc[Dataset_1['risk']==0, 'habit'] = Dataset_1.loc[Dataset_1['risk']==0, 'habit'].fillna(Dataset_1.loc[Dataset_1['risk']==0, 'habit'].mode()[0])
    Dataset_1.loc[Dataset_1['risk']==1, 'habit'] = Dataset_1.loc[Dataset_1['risk']==1, 'habit'].fillna(Dataset_1.loc[Dataset_1['risk']==1, 'habit'].mode()[0])
    # Data cleaning: Convert the invalid and unmatched values of habit columns
    def clean_habit_column(value):
        # Remove junk numerical values 
        if re.search(r"\d", str(value)):
            return np.nan
        

        value = str(value).strip().lower()
            
        
        
        
        habit_mapping = {
            "fast": "fast",
            "pick": "pick",
            "pick_bat": "bat_pick",
            "pick_rat": "rat_pick",
            "pick_and_bat": "bat_pick",
            "pick_and_rat": "rat_pick",
            "pick_and_others": "pick",
            "bat_and_pick": "bat_pick",
            "bat_pick": "bat_pick",
            "rat_and_pick": "rat_pick",
            "all_pick": "pick",
            "pick_and_all": "pick",
            "both": "both_pick",
            "bat_rat_pick": "both_pick",
            "pick_rat_and_bat": "both_pick",
            "rat": "rat",
            "bat": "bat",
            "bats": "bat",
            "bat_fight": "bat_fight",
            "fight_rat": "rat_fight",
            "rat_bat": "rat_bat",
            "bat_and_rat": "rat_bat",
            "rat_and_bat": "rat_bat",
            "no_food": "no_food",
            "rat_and_no_food": "no_food",
            "other_bats": "other",         
            "other": "other",
            "others": "other",
            "other_directions": "other",
            "NA": "unknown",
            "nan": "unknown",
            "pick_eating_all": "pick",
            "pup_and_mon": "other",
            "other_bats/rat": "other",
            "rat_pick_and_bat": "both_pick",
            "pick_bat_rat": "both_pick",      
            "pick_rat_bat": "both_pick",    
            "rat_bat_fight": "fight"
        }
        return habit_mapping.get(value, "other")

    # Add new column for cleaned habit
    Dataset_1["habit_updated"] = clean_habit_column(Dataset_1["habit"])

    #Feature Engineering
    # Adding total time of rat on platform as start time and end time does not mean much
    Dataset_1["rat_time"] = Dataset_1['rat_period_end'] - Dataset_1['rat_period_start']
    #Convert the time into seconds
    Dataset_1['rat_time'] = Dataset_1['rat_time'].dt.total_seconds()
    #Check whether the bat start approach food after rat left
    Dataset_1['bat_after_rat'] = Dataset_1['rat_time'] - Dataset_1['seconds_after_rat_arrival'] < 0

    if show_raw_dataset1:
        # Check logic that rat period start must be before rat period end
        invalid_periods = Dataset_1[Dataset_1["rat_period_start"] > Dataset_1["rat_period_end"]]
        print("\nRows with invalid rat period:", invalid_periods.shape[0])

        # Check invalid negative values in numeric columns
        print("\nNegative values check:")
        print((Dataset_1[["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset"]] < 0).sum())

        # Check the condition of null values in hours_after_sunset column
        negative_hours_after_sunset = Dataset_1[Dataset_1["hours_after_sunset"] < 0]
        print(negative_hours_after_sunset[["start_time", "sunset_time", "hours_after_sunset"]])

    # Handle the negative values in hours_after_sunset column
    Dataset_1.loc[Dataset_1["hours_after_sunset"] < 0, "hours_after_sunset"] = 0

    if show_raw_dataset1:
        print(Dataset_1.dtypes)


    # # Month, season, risk, reward mapping
    #Not mentioned need to further discuss
    # month_map = {0: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun"}
    # Dataset_1["month"] = Dataset_1["month"].map(month_map)

    # season_map = {0: "Summer", 1: "Winter"}
    # Dataset_1["season"] = Dataset_1["season"].map(season_map)


    
    return Dataset_1


if __name__ == "__main__":
    Dataset_1_cleaned = clean_dataset_1(show_raw_dataset1=True)