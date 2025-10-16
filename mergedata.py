import pandas as pd
import scipy.stats as stats
import statsmodels.stats.weightstats as stm

from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2
Dataset_1 = clean_dataset_1(show_raw_dataset1=False)
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

def merge_datasets(Dataset_1, Dataset_2):
    Dataset_1 = Dataset_1.sort_values("rat_period_start").reset_index(drop=True)
    Dataset_2 = Dataset_2.sort_values("time").reset_index(drop=True)
    
    merged = pd.merge_asof(
        Dataset_1,               # swap order
        Dataset_2,               # so that we merge Dataset_1 to Dataset_2
        left_on="rat_period_start",
        right_on="time",
        direction="backward",
        tolerance=pd.Timedelta("30min")
    )
    

    
    return merged
if __name__ == "__main__":
    merged_data = merge_datasets(Dataset_1, Dataset_2)
    merged_data.to_csv("merged_dataset.csv", index=False)
