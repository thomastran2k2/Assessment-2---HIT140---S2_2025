import pandas as pd
import scipy.stats as stats

from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2

Dataset_1 = clean_dataset_1(show_raw_dataset1=False)
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

# Does rat present have an impact on the number of bats landings?
# H0: The mean of bat landings is equal for rat present and rat absent
# H1: The mean of bat landings for rat present is less than for rat absent

# Splitting the groups on the basis of rat present and absent
rat_present = Dataset_2[Dataset_2["rat_arrival_number"] >= 1]["bat_landing_number"]
rat_absent = Dataset_2[Dataset_2["rat_arrival_number"] == 0]["bat_landing_number"]

t_stats, p_val = stats.ttest_ind_from_stats(rat_present.mean(), rat_present.std(), rat_present.count(), rat_absent.mean(), rat_absent.std(), rat_absent.count(), equal_var=False, alternative='less')

print("\nTwo-sample t-test to check bat landings with rat present vs absent:")
print("\n The summary statistics are given below:")
print(f"\t Mean (rat present) = {rat_present.mean():.2f}")
print(f"\t Mean (rat absent) = {rat_absent.mean():.2f}")
print(f"\t Standard deviation (rat present) = {rat_present.std():.2f}")
print(f"\t Standard deviation (rat absent) = {rat_absent.std():.2f}")
print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)
print("\n Computing p-value ...")
print("\t p-value: %.9f" % p_val)
print("\n The result of hypothesis is:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
    print("\t Reject H0: Bat landings differ with rat presence")
else:
    print("\t We fail to reject the null hypothesis.")
    print("\t Fail to reject H0:  There is no evidence of difference in bat landings with rat presence")
