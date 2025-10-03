# Data Analysis for dataset 2

# Import all the necessary libraries
import pandas as pd
import numpy as np
import re

def clean_dataset2(show_raw_dataset2=True):
    # Step 2: Read the dataset
    Dataset_2 = pd.read_csv("dataset2.csv")
    
    if show_raw_dataset2:
        # Check the first few rows of dataset 2
        print("Displaying first few rows of Raw Dataset 2:")
        print(Dataset_2.head())

        # Check structure and distinct values
        print("\nInformation of Raw Dataset 2:")
        print(Dataset_2.info(), "\n")

        print("Distinct values in Raw Dataset 2:")
        print(Dataset_2.nunique(), "\n")

        # Check missing values
        print("Missing values in Raw Dataset 2:")
        print(Dataset_2.isnull().sum(), "\n")

    # Check for duplicates and drop if exists
    duplicate_data = Dataset_2[Dataset_2.duplicated()]
    if show_raw_dataset2:
        print("Duplicate rows in dataset 2:")
        print(duplicate_data)

    Dataset_2 = Dataset_2.drop_duplicates()

    # Convert datatype of time related columns to datetime
    Dataset_2["time"] = pd.to_datetime(Dataset_2["time"], errors="coerce", dayfirst=True)
    Dataset_2["season_num"] = Dataset_2['season'].map({'winter': 1, 'summer': 0})
    Dataset_2['season'] = Dataset_2["month"].apply(lambda x: 'winter' if x in [1, 2, 11, 12] else 'summer')
            
    # Check the condition of negative values in hours_after_sunset column
    negative_hours_after_sunset = Dataset_2[Dataset_2["hours_after_sunset"] < 0]
    if show_raw_dataset2:
        print(negative_hours_after_sunset[["hours_after_sunset"]])

    # Handle the negative values in hours_after_sunset column
    Dataset_2.loc[Dataset_2["hours_after_sunset"] < 0, "hours_after_sunset"] = 0

    # Save the cleaned DataFrame to a new variable
    Dataset_2_cleaned = Dataset_2.copy() \

    

    if show_raw_dataset2:
        # Final check of datatypes
        print("\nFinal datatypes after cleaning:")
        print(Dataset_2_cleaned.dtypes)

        # Display cleaned dataset summary
        print("\nCleaned Dataset 2:")
        print(Dataset_2_cleaned.head())

        # Save the cleaned dataset 2 to a new CSV file
        dataset2_output_filename = "dataset2_cleaned.csv"
        Dataset_2_cleaned.to_csv(dataset2_output_filename, index=False) 
    
    return Dataset_2_cleaned

if __name__ == "__main__":
    Dataset_2_cleaned = clean_dataset2(show_raw_dataset2=True)