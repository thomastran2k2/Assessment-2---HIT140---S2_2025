import pandas as pd
import matplotlib.pyplot as plt
from Inferential_Analysis_3 import Dataset_1
import numpy as np
from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2
from mergedata import merge_datasets
import seaborn as sns

df = merge_datasets(clean_dataset_1(), clean_dataset2())
df2 = clean_dataset2(show_raw_dataset2=False)
df1 = clean_dataset_1(show_raw_dataset1=False)


# df['season'] = df['season'].map({0: 'winter', 1: 'summer'})



def landing_counts(season):
    winter_landing = df[df['season'] == 'winter']
    summer_landing = df[df['season'] == 'summer']
    counts = [len(winter_landing), len(summer_landing)]
    labels = ['Winter', 'Summer']
    plt.bar(labels, counts, color=['blue', 'orange'])
    plt.xlabel('Season')
    plt.ylabel('Number of Landings')
    plt.title('Number of Landings by Season')
    plt.show()
def risk_behaviour(season):
    winter_risk = df[df['season'] == 'winter']['risk'].mean()
    summer_risk = df[df['season'] == 'summer']['risk'].mean()
    risks = [winter_risk, summer_risk]
    labels = ['Winter', 'Summer']
    plt.bar(labels, risks, color=['blue', 'orange'])
    plt.xlabel('Season')
    plt.ylabel('Average Risk Taking')
    plt.title('Average Risk Taking by Season')
    plt.show()

def risk_taking_seasonal():
    df.head()
    risk_taking_winter = df[df["risk"] == 1][df["season"] == "winter"]["bat_landing_to_food"].dropna()
    risk_avoidance_winter = df[df["risk"] == 0][df["season"] == "winter"]["bat_landing_to_food"].dropna()
    risk_taking_summer = df[df["risk"] == 1][df["season"] == "summer"]["bat_landing_to_food"].dropna()
    risk_avoidance_summer = df[df["risk"] == 0][df["season"] == "summer"]["bat_landing_to_food"].dropna()

    means = [
        risk_taking_winter.mean(),
        risk_avoidance_winter.mean(),
        risk_taking_summer.mean(),
        risk_avoidance_summer.mean()
    ]
    labels = [
        'Winter Risk Taking',
        'Winter Risk Avoidance',
        'Summer Risk Taking',
        'Summer Risk Avoidance'
    ]
    plt.bar(labels, means, color=['blue', 'lightblue', 'orange', 'gold'])
    plt.ylabel('Mean bat_landing_to_food')
    plt.title('Mean bat_landing_to_food by Risk and Season')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()

def data_exploration():
    print(df['season'].value_counts())
    print(df['bat_landing_to_food'].describe())
    print(df['bat_landing_to_food'].isnull().sum())
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="season", y="bat_landing_to_food", data=df)
    plt.title('Boxplot of bat_landing_to_food by Season')
    plt.xlabel('Season')
    plt.yscale('log')
    plt.ylabel('Bat Landing to Food')
    plt.show()


    
def number_of_bats_seasonal():

    print(df[(df['rat_arrival_number'] == 0) & (df["season"] == "winter")]['bat_landing_number'])
    df2_rat_mean = df2[(df2['rat_arrival_number'] != 0) & (df2["season"] == "winter")]['bat_landing_number'].mean()
    df2_no_rat_mean = df2[(df2['rat_arrival_number'] == 0) & (df2["season"] == "winter")]['bat_landing_number'].mean()
    df2_rat_mean_summer = df2[(df2['rat_arrival_number'] != 0) & (df2["season"] == "summer")]['bat_landing_number'].mean()
    df2_no_rat_mean_summer = df2[(df2['rat_arrival_number'] == 0) & (df2["season"] == "summer")]['bat_landing_number'].mean()

    means = [
        df2_rat_mean,
        df2_no_rat_mean,
        df2_rat_mean_summer,
        df2_no_rat_mean_summer
    ]

    print(means )
    labels = [
        'Winter Rat Present',
        'Winter No Rat Present',
        'Summer Rat Present',
        'Summer No Rat Present'
    ]
    plt.bar(labels, means, color=['blue', 'lightblue', 'orange', 'gold'])
    plt.ylabel('Mean bat_landing_number')
    plt.title('Mean bat_landing_number by Rat Presence and Season')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()  

def season_exploratory():
    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['month', 'hours_after_sunset']:
        if col in numeric_cols:
            numeric_cols.remove(col)  # Remove month to avoid redundancy with season
    sns.pairplot(df2, hue="season", vars=numeric_cols, diag_kind="kde")
    plt.suptitle('Scatter Plot Matrix by Season', y=1.02)
    plt.show()

def season_exploratory2():
    numeric_cols = df1.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['month', 'hours_after_sunset', 'food_availbility', 'habit_updated', 'reward']:
        if col in numeric_cols:
            numeric_cols.remove(col)  # Remove month to avoid redundancy with season
    plt.figure(figsize=(12, 8))
    sns.pairplot(df1, hue="season", vars=numeric_cols, diag_kind="kde")
    plt.suptitle('Scatter Plot Matrix by Season', y=1.02)
    plt.show()

def descriptive_investigationB(df1, df2):
    # ----------------- Dataset 1 summaries -----------------
    print("=== Dataset 1 (Bat landings & behaviour) ===")
    print(df1.groupby("season")["risk"].mean().rename("Risk-taking rate"))
    print(df1.groupby("season")["reward"].mean().rename("Reward rate"))
    print(df1.groupby("season")["seconds_after_rat_arrival"].mean().rename("Mean seconds after rat arrival"))
    print(df1.groupby("season")["bat_landing_to_food"].mean().rename("Mean landing-to-food time (sec)"))
    
    # Risk-taking proportions per season
    risk_counts = pd.crosstab(df1["season"], df1["risk"], normalize="index")
    reward_counts = pd.crosstab(df1["season"], df1["reward"], normalize="index")
    print("\nRisk-taking proportions by season:\n", risk_counts)
    print("\nReward proportions by season:\n", reward_counts)
    
    # ----------------- Dataset 2 summaries -----------------
    print("\n=== Dataset 2 (Rat arrivals & observation periods) ===")
    print(df2.groupby("season")["rat_arrival_number"].mean().rename("Mean rat arrivals"))
    print(df2.groupby("season")["bat_landing_number"].mean().rename("Mean bat landings"))
    print(df2.groupby("season")["rat_minutes"].mean().rename("Mean rat minutes on platform"))
    print(df2.groupby("season")["food_availability"].mean().rename("Mean food availability"))
    
    # ----------------- Visualisations -----------------
    # Bat landings by season
    plt.figure(figsize=(7,5))
    sns.boxplot(x="season", y="bat_landing_number", data=df2)
    plt.title("Bat landings per 30-min (by season)")
    plt.tight_layout()
    plt.show()

    # Rat arrivals by season
    plt.figure(figsize=(7,5))
    sns.boxplot(x="season", y="rat_arrival_number", data=df2)
    plt.title("Rat arrivals per 30-min (by season)")
    plt.tight_layout()
    plt.show()

    # Risk-taking by season
    plt.figure(figsize=(7,5))
    sns.barplot(x="season", y="risk", data=df1, estimator="mean")
    plt.title("Risk-taking rate by season")
    plt.tight_layout()
    plt.show()

    # Rewarding behaviour by season
    plt.figure(figsize=(7,5))
    sns.barplot(x="season", y="reward", data=df1, estimator="mean")
    plt.title("Rewarding behaviour rate by season")
    plt.tight_layout()
    plt.show()





    

if __name__ == "__main__":
    # landing_counts('season')
        # risk_behaviour('season')
    # risk_taking_seasonal()
    # number_of_bats_seasonal()
    
    # data_exploration()
    # season_exploratory2()
    descriptive_investigationB(df1, df2)

