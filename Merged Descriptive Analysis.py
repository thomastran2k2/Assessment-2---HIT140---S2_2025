import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Data_Analysis_Dataset_1 import load_and_clean_dataset1
from Data_Analysis_Dataset_2 import load_and_clean_dataset2

Dataset_1 = load_and_clean_dataset1(show_raw_dataset1=False)
Dataset_2 = load_and_clean_dataset2(show_raw_data=False)


print(Dataset_1['bat_landing_to_food'].describe())
print(Dataset_2['bat_landing_number'].describe())
print(Dataset_2['rat_arrival_number'].describe())


print(Dataset_1['risk'].value_counts())
print(Dataset_1['reward'].value_counts())
print(Dataset_1['season'].value_counts())
print(Dataset_2['month'].value_counts())

print("Mean bat landings per 30-min:", Dataset_2['bat_landing_number'].mean())
print("Mean rat arrivals per 30-min:", Dataset_2['rat_arrival_number'].mean())


#Count Plots for the Categorical variables
# Risk
plt.figure(figsize=(6,4))
sns.countplot(x="risk", data=Dataset_1)
plt.title("Risk-taking vs Risk-avoidance")
plt.show()

# Reward
plt.figure(figsize=(6,4))
sns.countplot(x="reward", data=Dataset_1)
plt.title("Reward vs No Reward")
plt.show()

# Season
plt.figure(figsize=(6,4))
sns.countplot(x="season", data=Dataset_1)
plt.title("Season Distribution")
plt.show()

# Month
plt.figure(figsize=(8,5))
sns.countplot(x="month", data=Dataset_1)
plt.title("Month Distribution")
plt.show()

# Scatter plot: bat vs rat
plt.figure(figsize=(7,5))
sns.scatterplot(x="bat_landing_number", y="rat_arrival_number", data=Dataset_1, alpha=0.6)
plt.title("Bat Landings vs Rat Arrivals")
plt.show()


# Histogram: Bat landings
plt.hist(Dataset_2['bat_landing_number'], bins=30, edgecolor='black')
plt.title("Distribution of Bat Landings (per 30-min)")
plt.xlabel("Bat landings")
plt.ylabel("Frequency")
plt.show()

# Histogram: Rat arrivals
plt.hist(Dataset_2['rat_arrival_number'], bins=30, edgecolor='black')
plt.title("Distribution of Rat Arrivals (per 30-min)")
plt.xlabel("Rat arrivals")
plt.ylabel("Frequency")
plt.show()

# Histogram: Bat landings to food
plt.hist(Dataset_1['bat_landing_to_food'], bins=30, edgecolor='black')
plt.title("Distribution of Bat Landings to Food")
plt.xlabel("Bat landings to food")
plt.ylabel("Frequency")
plt.show()

print(Dataset_1["season"].unique())
plt.figure(figsize=(6,5))
sns.boxplot(data=Dataset_1, x="season", y="bat_landing_to_food", palette="Set2")
plt.title("Vigilance (Time to Food) by Season")
plt.ylabel("Seconds to Approach Food")
plt.xlabel("Season")
plt.show()

#Scatter: bat_landing_to_food vs hours_after_sunset
plt.figure(figsize=(6,5))
sns.scatterplot(data=Dataset_1, x="hours_after_sunset", y="bat_landing_to_food", hue="season", alpha=0.7)
plt.title("Vigilance vs Hours After Sunset")
plt.ylabel("Seconds to Approach Food")
plt.xlabel("Hours After Sunset")
plt.show()

# Scatter: bat_landing_to_food vs seconds_after_rat_arrival
plt.figure(figsize=(6,5))
sns.scatterplot(data=Dataset_1, x="seconds_after_rat_arrival", y="bat_landing_to_food", alpha=0.7)
plt.title("Vigilance vs Seconds After Rat Arrival")
plt.ylabel("Seconds to Approach Food")
plt.xlabel("Seconds After Rat Arrival")
plt.show()

# Count of risk by season
plt.figure(figsize=(6,5))
sns.countplot(data=Dataset_1, x="season", hue="risk", palette="Set1")
plt.title("Risk-Taking by Season")
plt.ylabel("Number of Bat Landings")
plt.xlabel("Season")
plt.legend(["Risk-Avoidance (0)", "Risk-Taking (1)"])
plt.show()

# Risk vs Reward (stacked bar)
risk_reward = Dataset_1.groupby(["risk","reward"]).size().unstack(fill_value=0)
risk_reward.plot(kind="bar", stacked=True, figsize=(6,5), colormap="Set3")
plt.title("Risk-Taking vs Foraging Success")
plt.ylabel("Number of Bat Landings")
plt.xlabel("Risk Behaviour")
plt.legend(["No Reward (0)", "Reward (1)"])
plt.show()

#Count of reward by season
plt.figure(figsize=(6,5))
sns.countplot(data=Dataset_1, x="season", hue="reward", palette="Set2")
plt.title("Foraging Success by Season")
plt.ylabel("Number of Bat Landings")
plt.xlabel("Season")
plt.legend(["No Reward (0)", "Reward (1)"])
plt.show()