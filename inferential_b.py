import scipy.stats as stats
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt

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

def bat_landing_to_food_by_season():
    print("Inferential Analysis on bat_landing_to_food by Season")
    # Does season have an impact on the time taken for bats to land and get food?
    winter = df1[df1['season']=="winter"]['bat_landing_to_food'].dropna()
    summer = df1[df1['season']=="summer"]['bat_landing_to_food'].dropna()
    t_stats, p_val = ttest_ind(winter, summer, alternative='two-sided')
    # Check normality (Shapiro-Wilk test)
    print(stats.shapiro(winter))
    print(stats.shapiro(summer))

    if stats.shapiro(winter).pvalue < 0.05 or stats.shapiro(summer).pvalue < 0.05:
        print("Data is not normally distributed, consider using a non-parametric test like Mann-Whitney U test.")
        u_stat, p_val = mannwhitneyu(winter, summer)
        print("Mann-Whitney U:", u_stat, "p-value:", p_val)
        if p_val < 0.05:
            print("We reject the null hypothesis. Season has a significant impact on bat_landing_to_food.")
        else:
            print("We fail to reject the null hypothesis. No significant impact of season on bat_landing_to_food.")
    else:
        # If normal:
        print("T-test for bat_landing_to_food by Season:")
        print("T-statistics:", t_stats, "p-value:", p_val)
        if p_val < 0.05:
            print("We reject the null hypothesis. Season has a significant impact on bat_landing_to_food.")
        else:
            print("We fail to reject the null hypothesis. No significant impact of season on bat_landing_to_food.")

def rat_arrival_number_by_season():
    print("Inferential Analysis on rat_arrival_number by Season")
    # Does season have an impact on the number of rat arrivals?
    winter = df2[df2['season']=="winter"]['rat_arrival_number'].dropna()
    summer = df2[df2['season']=="summer"]['rat_arrival_number'].dropna()
    t_stats, p_val = ttest_ind(winter, summer, alternative='two-sided')
    # Check normality (Shapiro-Wilk test)
    print(stats.shapiro(winter))
    print(stats.shapiro(summer))
    if stats.shapiro(winter).pvalue < 0.05 or stats.shapiro(summer).pvalue < 0.05:
        print("Data is not normally distributed, consider using a non-parametric test like Mann-Whitney U test.")
        u_stat, p_val = mannwhitneyu(winter, summer)
        print("Mann-Whitney U:", u_stat, "p-value:", p_val)
        if p_val < 0.05:
            print("We reject the null hypothesis. Season has a significant impact on rat_arrival_number.")
        else:
            print("We fail to reject the null hypothesis. No significant impact of season on rat_arrival_number.")
        return

    # If normal:
    print("T-test for rat_arrival_number by Season:")
    print("T-statistics:", t_stats, "p-value:", p_val)
    if p_val < 0.05:
        print("We reject the null hypothesis. Season has a significant impact on rat_arrival_number.")
    else:
        print("We fail to reject the null hypothesis. No significant impact of season on rat_arrival_number.")

def compare_bat_landings_by_season():
    print("Inferential Analysis on bat_landing_number by Season")
    winter = df2[df2['season']=="winter"]['bat_landing_number'].dropna()
    summer = df2[df2['season']=="summer"]['bat_landing_number'].dropna()
    t_stats, p_val = mannwhitneyu(winter, summer, alternative='two-sided')
    # Check normality (Shapiro-Wilk test)
    print(stats.shapiro(winter))
    print(stats.shapiro(summer))

    if stats.shapiro(winter).pvalue < 0.05 or stats.shapiro(summer).pvalue < 0.05:
        print("Data is not normally distributed, using Mann-Whitney U test.")

        # If not normal:
        u_stat, p_val = mannwhitneyu(winter, summer)
        print("Mann-Whitney U:", u_stat, "p-value:", p_val)
        if p_val < 0.05:
            print("We reject the null hypothesis. Season has a significant impact on bat_landing_to_food.")
        else:
            print("We fail to reject the null hypothesis. No significant impact of season on bat_landing_to_food.")
    else:
        # If normal:
        print("T-test for bat_landing_to_food by Season:")
        print("T-statistics:", t_stats, "p-value:", p_val)
        if p_val < 0.05:
            print("We reject the null hypothesis. Season has a significant impact on bat_landing_to_food.")
        else:
            print("We fail to reject the null hypothesis. No significant impact of season on bat_landing_to_food.")

def confidential_intervals():
    df1_summer = df1[df1['season']=="summer"]
    df1_winter = df1[df1['season']=="winter"]
    df2_summer = df2[df2['season']=="summer"]
    df2_winter = df2[df2['season']=="winter"]
    print("Confidence Intervals for bat_landing_to_food in Summer:")
    ci_food_summer = stats.t.interval(
        0.95,
        len(df1_summer['bat_landing_to_food'].dropna()) - 1,
        loc=df1_summer['bat_landing_to_food'].dropna().mean(),
        scale=stats.sem(df1_summer['bat_landing_to_food'].dropna())
    )
    print(f"({ci_food_summer[0]:.4f}, {ci_food_summer[1]:.4f})")
    print("Confidence Intervals for bat_landing_to_food in Winter:")
    ci_food_winter = stats.t.interval(
        0.95,
        len(df1_winter['bat_landing_to_food'].dropna()) - 1,
        loc=df1_winter['bat_landing_to_food'].dropna().mean(),
        scale=stats.sem(df1_winter['bat_landing_to_food'].dropna())
    )
    print(f"({ci_food_winter[0]:.4f}, {ci_food_winter[1]:.4f})")

    print("Confidence Intervals for bat_landing_number in Summer:")
    ci_bat_landing_number_summer = stats.t.interval(
        0.95,
        len(df2_summer['bat_landing_number'].dropna()) - 1,
        loc=df2_summer['bat_landing_number'].dropna().mean(),
        scale=stats.sem(df2_summer['bat_landing_number'].dropna())
    )
    print(f"({ci_bat_landing_number_summer[0]:.4f}, {ci_bat_landing_number_summer[1]:.4f})")

    print("Confidence Intervals for bat_landing_number in Winter:")
    ci_bat_landing_number_winter = stats.t.interval(
        0.95,
        len(df2_winter['bat_landing_number'].dropna()) - 1,
        loc=df2_winter['bat_landing_number'].dropna().mean(),
        scale=stats.sem(df2_winter['bat_landing_number'].dropna())
    )
    print(f"({ci_bat_landing_number_winter[0]:.4f}, {ci_bat_landing_number_winter[1]:.4f})")

    print("Confidence Intervals for rat_arrival_number in Summer:")
    ci_rat_arrival_number_summer = stats.t.interval(
        0.95,
        len(df2_summer['rat_arrival_number'].dropna()) - 1,
        loc=df2_summer['rat_arrival_number'].dropna().mean(),
        scale=stats.sem(df2_summer['rat_arrival_number'].dropna())
    )
    print(f"({ci_rat_arrival_number_summer[0]:.4f}, {ci_rat_arrival_number_summer[1]:.4f})")

    print("Confidence Intervals for rat_arrival_number in Winter:")
    ci_rat_arrival_number_winter = stats.t.interval(
        0.95,
        len(df2_winter['rat_arrival_number'].dropna()) - 1,
        loc=df2_winter['rat_arrival_number'].dropna().mean(),
        scale=stats.sem(df2_winter['rat_arrival_number'].dropna())
    )
    print(f"({ci_rat_arrival_number_winter[0]:.4f}, {ci_rat_arrival_number_winter[1]:.4f})")

    

    # Plot confidence intervals for bat_landing_to_food
    data_dict_food = {
        'category': ["summer", "winter"],
        'lower': [ci_food_summer[0], ci_food_winter[0]],
        'upper': [ci_food_summer[1], ci_food_winter[1]]
    }
    dataset_food = pd.DataFrame(data_dict_food)
    plt.figure(figsize=(6, 3))
    for lower, upper, y in zip(dataset_food['lower'], dataset_food['upper'], range(len(dataset_food))):
        plt.plot((lower, upper), (y, y), 'ro-', color='blue')
    plt.title("Confidence Intervals for bat_landing_to_food")
    plt.yticks(range(len(dataset_food)), list(dataset_food['category']))
    plt.xlabel("Value")
    plt.show()

    # Plot confidence intervals for bat_landing_number
    data_dict_bat = {
        'category': ["summer", "winter"],
        'lower': [ci_bat_landing_number_summer[0], ci_bat_landing_number_winter[0]],
        'upper': [ci_bat_landing_number_summer[1], ci_bat_landing_number_winter[1]]
    }
    dataset_bat = pd.DataFrame(data_dict_bat)
    plt.figure(figsize=(6, 3))
    for lower, upper, y in zip(dataset_bat['lower'], dataset_bat['upper'], range(len(dataset_bat))):
        plt.plot((lower, upper), (y, y), 'ro-', color='orange')
    plt.title("Confidence Intervals for bat_landing_number")
    plt.yticks(range(len(dataset_bat)), list(dataset_bat['category']))
    plt.xlabel("Value")
    plt.show()

    # Plot confidence intervals for rat_arrival_number
    data_dict_rat = {
        'category': ["summer", "winter"],
        'lower': [ci_rat_arrival_number_summer[0], ci_rat_arrival_number_winter[0]],
        'upper': [ci_rat_arrival_number_summer[1], ci_rat_arrival_number_winter[1]]
    }
    dataset_rat = pd.DataFrame(data_dict_rat)
    plt.figure(figsize=(6, 3))
    for lower, upper, y in zip(dataset_rat['lower'], dataset_rat['upper'], range(len(dataset_rat))):
        plt.plot((lower, upper), (y, y), 'ro-', color='green')
    plt.title("Confidence Intervals for rat_arrival_number")
    plt.yticks(range(len(dataset_rat)), list(dataset_rat['category']))
    plt.xlabel("Value")
    plt.show()

    
    
if __name__ == "__main__":
    # risk_vs_season()
    # compare_ba
    # t_landings_by_season()
    # rat_arrival_number_by_season()
    # bat_landing_to_food_by_season()
    confidential_intervals()

