import scipy.stats as stats
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu

from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2

df1 = clean_dataset_1(show_raw_dataset1=False)
df2 = clean_dataset2(show_raw_dataset2=False)


def risk_vs_season():
    # Does season have an impact on the risk-taking behaviour of bats?

    contingency = pd.crosstab(df1['season'], df1['risk'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print("Chi-square test for Risk vs Season:")
    print("Chi2:", chi2, "p-value:", p)
    if p < 0.05:
        print("We reject the null hypothesis. Season has a significant impact on risk-taking behaviour.")
    else:
        print("We fail to reject the null hypothesis. No significant impact of season on risk-taking behaviour.")

def
