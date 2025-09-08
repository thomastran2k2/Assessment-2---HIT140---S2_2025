import pandas as pd
import scipy.stats as stats
import statsmodels.stats.weightstats as stm

from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2
Dataset_1 = clean_dataset_1(show_raw_dataset1=False)
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

Dataset_1 = Dataset_1.sort_values("rat_period_start").reset_index(drop=True)
Dataset_2 = Dataset_2.sort_values("time").reset_index(drop=True)

print(Dataset_1.head())
print(Dataset_2.head())
merged = pd.merge_asof(
    Dataset_1,
    Dataset_2,
    left_on="rat_period_start",
    right_on="time",
    direction="backward",     # match to the clostest earlier observation period
    tolerance=pd.Timedelta("30min")  # within ±30 minutes   
)
print(merged.head())
merged.to_csv("merged_dataset.csv", index=False)