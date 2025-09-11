import pandas as pd
import scipy.stats as stats
import statsmodels.stats.weightstats as stm
import matplotlib.pyplot as plt

from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2

Dataset_1 = clean_dataset_1(show_raw_dataset1=False)
Dataset_2 = clean_dataset2(show_raw_dataset2=False)

def confidence_interval_landing_time_risk(confidence=0.95):
    
    risk_taking = Dataset_1[Dataset_1["risk"] == 1]["bat_landing_to_food"]

    confidence_interval = stm._zconfint_generic(risk_taking.mean(), 
                                            risk_taking.std(),
                                            alpha=0.05, alternative='two-sided')  # for 95% confidence interval
    confidence_interval = (max(0, confidence_interval[0]), confidence_interval[1])  # ensure lower bound is not negative
    print(f"95% Confidence Interval (Risk Taking): {confidence_interval[0]:.2f} to {confidence_interval[1]:.2f}, mean: {risk_taking.mean():.2f}")


def confidence_interval_landing_time_avoidance(confidence=0.95):
    risk_avoidance = Dataset_1[Dataset_1["risk"] == 0]["bat_landing_to_food"]
    confidence_interval_avoidance = stm._zconfint_generic(risk_avoidance.mean(), 
                                            risk_avoidance.std(),
                                            alpha=0.05, alternative='two-sided')  # for 95% confidence interval
    confidence_interval_avoidance = (max(0, confidence_interval_avoidance[0]), confidence_interval_avoidance[1])  # ensure lower bound is not negative
    print(f"95% Confidence Interval (Risk Avoidance): {confidence_interval_avoidance[0]:.2f} to {confidence_interval_avoidance[1]:.2f}, mean: {risk_avoidance.mean():.2f}")

def confidence_interval_bat_no_rat():
    no_rat = Dataset_2[Dataset_2["rat_arrival_number"] == 0]["bat_landing_number"]
    confidence_interval_no_rat = stm._zconfint_generic(no_rat.mean(),
                                            no_rat.std(),
                                            alpha=0.05, alternative='two-sided')  # for 95% confidence interval
    confidence_interval_no_rat = (max(0, confidence_interval_no_rat[0]), confidence_interval_no_rat[1])  # ensure lower bound is not negative
    print(f"95% Confidence Interval (No Rat): {confidence_interval_no_rat[0]:.2f} to {confidence_interval_no_rat[1]:.2f}, mean: {no_rat.mean():.2f}")
    return confidence_interval_no_rat
def confidence_interval_bat_with_rat():
    with_rat = Dataset_2[Dataset_2["rat_arrival_number"] >= 1]["bat_landing_number"]
    confidence_interval_with_rat = stm._zconfint_generic(with_rat.mean(),
                                            with_rat.std(),
                                            alpha=0.05, alternative='two-sided')  # for 95% confidence interval
    confidence_interval_with_rat = (max(0, confidence_interval_with_rat[0]), confidence_interval_with_rat[1])  # ensure lower bound is not negative
    print(f"95% Confidence Interval (With Rat): {confidence_interval_with_rat[0]:.2f} to {confidence_interval_with_rat[1]:.2f}, mean: {with_rat.mean():.2f}")
    return confidence_interval_with_rat
 
def plot_all_confidence_intervals():
    intervals = []
    labels = []

    # Risk Taking
    risk_taking = Dataset_1[Dataset_1["risk"] == 1]["bat_landing_to_food"]
    ci_risk = stm._zconfint_generic(risk_taking.mean(), risk_taking.std(), alpha=0.05, alternative='two-sided')
    ci_risk = (max(0, ci_risk[0]), ci_risk[1])
    intervals.append(ci_risk)
    labels.append("Risk Taking")

    # Risk Avoidance
    risk_avoidance = Dataset_1[Dataset_1["risk"] == 0]["bat_landing_to_food"]
    ci_avoid = stm._zconfint_generic(risk_avoidance.mean(), risk_avoidance.std(), alpha=0.05, alternative='two-sided')
    ci_avoid = (max(0, ci_avoid[0]), ci_avoid[1])
    intervals.append(ci_avoid)
    labels.append("Risk Avoidance")

    # No Rat
    no_rat = Dataset_2[Dataset_2["rat_arrival_number"] == 0]["bat_landing_number"]
    ci_no_rat = stm._zconfint_generic(no_rat.mean(), no_rat.std(), alpha=0.05, alternative='two-sided')
    ci_no_rat = (max(0, ci_no_rat[0]), ci_no_rat[1])
    intervals.append(ci_no_rat)
    labels.append("No Rat")

    # With Rat
    with_rat = Dataset_2[Dataset_2["rat_arrival_number"] >= 1]["bat_landing_number"]
    ci_with_rat = stm._zconfint_generic(with_rat.mean(), with_rat.std(), alpha=0.05, alternative='two-sided')
    ci_with_rat = (max(0, ci_with_rat[0]), ci_with_rat[1])
    intervals.append(ci_with_rat)
    labels.append("With Rat")

    means = [(ci[0] + ci[1]) / 2 for ci in intervals]
    errors = [(ci[1] - ci[0]) / 2 for ci in intervals]

    plt.figure(figsize=(8, 5))
    plt.errorbar(labels, means, yerr=errors, fmt='o', capsize=8, linestyle='None', color='blue')
    plt.ylabel('Value')
    plt.title('95% Confidence Intervals')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    confidence_interval_landing_time_risk()
    confidence_interval_landing_time_avoidance()
    confidence_interval_bat_no_rat()
    confidence_interval_bat_with_rat()
    plot_all_confidence_intervals()
    







