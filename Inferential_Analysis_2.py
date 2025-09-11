import pandas as pd
from scipy.stats import mannwhitneyu


from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2

Dataset_1 = clean_dataset_1(show_raw_dataset1=False)
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

# Does risk-taking bats approach food slower than risk-avoiding bats?
# H0: The distribution of bat landing to food times is the same or lower for risk-taking bats compared to risk-avoiding bats.
# H1: The distribution of bat landing to food times is greater for risk-taking bats (risk-taking bats approach food slower).



# Splitting the groups on the basis of risk taking and avoidance
risk_taking = Dataset_1[Dataset_1["risk"] == 1]["bat_landing_to_food"].dropna()
risk_avoidance = Dataset_1[Dataset_1["risk"] == 0]["bat_landing_to_food"].dropna()

print(risk_taking.mean(), risk_avoidance.mean())

# Perform one-sided two samples t-test
t_stats, p_val = mannwhitneyu(risk_taking, risk_avoidance, alternative='greater')

print("\nTwo-sample t-test to check bat behaviour while risk-taking vs risk-avoidance:")


print("\n Computing u statistics ...")
print("\t U-statistic (U): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.9f" % p_val)

print("\n The result of hypothesis is:")
if p_val < 0.05:
    print("\t We reject the null hypothesis.")
    print("\t Reject H0: Risk-taking vs risk-avoidance shows significant difference")
else:
    print("\t We fail to reject the null hypothesis.")
    print("\t Fail to reject H0:  There is no evidence of difference between risk-taking and avoidance")
    