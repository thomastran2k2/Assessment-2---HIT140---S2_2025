from Inferential_Analysis_1_CI import Dataset_1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from Data_Analysis_Dataset_1 import clean_dataset_1
from Data_Analysis_Dataset_2 import clean_dataset2


#Import csv files into dataframes
df1 = clean_dataset_1(show_raw_dataset1=False)
df2 = clean_dataset2(show_raw_dataset2=False)


#-------------------------------Data Wrangling------------------------------------------
# #Data Cleaning

# #Determine where the null values come from
# nvl = null_values_df1 = df1.isnull().sum()
# print(nvl)

# #Fill the null values 
# df1[df1['risk']==0]['habit'].fillna(df1[df1['risk']==0]['habit'].mode(), inplace=True)
# df1[df1['risk']==1]['habit'].fillna(df1[df1['risk']==1]['habit'].mode(), inplace=True)
# #Convert datetime values
# df1['start_time'] = pd.to_datetime(df1['start_time'], errors='coerce', dayfirst=True)
# df1['rat_period_start'] = pd.to_datetime(df1['rat_period_start'], errors='coerce', dayfirst=True)
# df1['start_time'] = pd.to_datetime(df1['start_time'], errors='coerce', dayfirst=True)
# df1['rat_period_end'] = pd.to_datetime(df1['rat_period_end'], errors='coerce', dayfirst=True)
# df1['sunset_time'] = pd.to_datetime(df1['sunset_time'], errors='coerce', dayfirst=True)

# df2["time"] = pd.to_datetime(df2["time"], errors='coerce', dayfirst=True)

# mask = df1['habit'].astype(str).str.contains('fight', case=False, na=False)

# #Cleaning habit data
# df1.loc[mask, 'habit'] = 'fight'   # in-place overwrite of the column where matched


# unique = df1['habit'].unique()

# #Feature engineering

# #Adding total time of rat on platform as start time and end time does not mean much

# df1["rat_time"] = df1['rat_period_end'] - df1['rat_period_start']
# #Convert the time into seconds
# df1['rat_time'] = df1['rat_time'].dt.total_seconds()
# #Check whether the bat start approach food after rat left
# df1['bat_after_rat'] = df1['rat_time'] - df1['seconds_after_rat_arrival'] < 0 


#---------------Risk vs No-risk Taking Behaviour, Reward vs No-Reward for each of the behaviour
def plot_risk_behaviour(df1):
    risk_taking_mean = np.mean(df1['risk'])
    risk_data = np.array([risk_taking_mean, 1 - risk_taking_mean])
    pie1_labels = ["risk-taking", "no risk-taking"]

    risk_taking_success = np.mean(df1[df1['risk'] == 1]['reward'])
    risk_taking_failure = 1 - risk_taking_success

    no_risk_success = np.mean(df1[df1['risk'] == 0]['reward'])
    no_risk_failure = 1 - no_risk_success

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,8))
    #Pie chart that shows the percentage of risk-taking behaviour versus not risk-taking behaviour
    ax1.pie(risk_data, labels = pie1_labels, autopct= '%1.1f%%')
    ax1.set_title("Risk-Taking Behaviour")
    ax2.pie([risk_taking_success, risk_taking_failure], 
            labels=['Success', 'Failure'], autopct='%1.1f%%')
    ax2.set_title("Risk-Taking Behaviour Outcomes")

    ax3.pie([no_risk_success, no_risk_failure], 
            labels=['Success', 'Failure'], autopct='%1.1f%%')
    ax3.set_title("No Risk-Taking Behaviour Outcomes")

    plt.tight_layout()
    plt.show()

##-------------------Find the relationship between number of rat arrivals and number of bat landings----------------

def rat_vs_bat_number(df2):
    #Find the mean of bat landings for each value of rat arrivals
    df2_bat_mean = df2.groupby('rat_arrival_number')['bat_landing_number'].mean().values

    #Find the unique values of bat landings
    df2_uniq_rat = np.array(df2['rat_arrival_number'].unique())

    #Find the mean number of bat landings during rat presence and not
    df2_rat_mean = df2[df2['rat_arrival_number'] != 0]['bat_landing_number'].mean()
    df2_no_rat_mean = df2[df2['rat_arrival_number'] == 0]['bat_landing_number'].mean()
    df2_rat_std = df2[df2['rat_arrival_number'] != 0]['bat_landing_number'].std()
    df2_no_rat_std = df2[df2['rat_arrival_number'] == 0]['bat_landing_number'].std()
    print(f"Mean Bat Landings (Rat Present): {df2_rat_mean:.2f}, (No Rat Present): {df2_no_rat_mean:.2f}, Stdev (Rat Present): {df2_rat_std:.2f}, (No Rat Present): {df2_no_rat_std:.2f}")
    #Plot
    plt.figure(figsize=(8,6))
    plt.bar(['Rat Present', 'No Rat Present'], [df2_rat_mean, df2_no_rat_mean])
    plt.ylabel('Mean Bat Landings (30-min period)')
    plt.title('Mean Bat Landings for Rat Arrivals vs No Rat Arrivals')
    plt.errorbar(['Rat Present', 'No Rat Present'], [df2_rat_mean, df2_no_rat_mean], yerr=[df2_rat_std, df2_no_rat_std], fmt='o', color='black', capsize=5)
    plt.show()



# def rat_vs_bat_number_sunset(df2):
#     #Find the mean of bat landings for each value of rat arrivals
#     df2_bat_mean = df2.groupby('rat_arrival_number')['bat_landing_number'].mean().values

#     #Find the unique values of bat landings
#     df2_uniq_rat = np.array(df2['rat_arrival_number'].unique())

#     #Find the mean number of bat landings during rat presence and not
#     df2_rat_mean = df2[df2['rat_arrival_number'] != 0]['bat_landing_number'].mean()
#     df2_no_rat_mean = df2[df2['rat_arrival_number'] == 0]['bat_landing_number'].mean()

#     #Plot
#     plt.figure(figsize=(8,6))
#     plt.bar(['Rat Present', 'No Rat Present'], [df2_rat_mean, df2_no_rat_mean])
#     plt.ylabel('Mean Bat Landings (30-min period)')
#     plt.title('Mean Bat Landings for Rat Arrivals vs No Rat Arrivals')
#     plt.show()




#---------------------Measure seconds after arrival when risk vs no-risk behaviour occur -------------------------------
def plot_second_after_rat_arrival(df1):
    mean1 = df1[df1['risk'] == 1]['seconds_after_rat_arrival'].mean()
    mean2 = df1[df1['risk'] == 0]['seconds_after_rat_arrival'].mean()

    #Plot
    plt.bar(['Risk-Taking', 'No Risk-Taking'], [mean1, mean2])
    plt.ylabel('Mean Bat Landing Time After Rat Arrival (seconds)')
    plt.title('Mean Bat Landing Time to Food During Risk-Taking vs No Risk-Taking Behaviour')
    plt.show()

    print(mean1,mean2)
def demonstrate(df2):
    df1['habit'].astype('numeric', errors='coerce')

#Compare behaviour between and after sunset

# print(np.min(df1['hours_after_sunset']), np.max(df1['hours_after_sunset']))
# for i in range(0,13):
#     mean2 = df1[df1['hours_after_sunset'] >= i]['risk'].mean()
#     print(mean2)
    # print(df1.isnull().sum())
    # print(df1['habit'].unique())


#--------------------- Mean bat landing time during risk-taking behavior vs not risk-taking behaviour
def plot_bat_landing_time_risk(df1):
    mean1 = df1[df1['risk'] == 1]['bat_landing_to_food'].mean()
    mean2 = df1[df1['risk'] == 0]['bat_landing_to_food'].mean()
    stdev1 = df1[df1['risk'] == 1]['bat_landing_to_food'].std()
    stdev2 = df1[df1['risk'] == 0]['bat_landing_to_food'].std()

    #Plot
    plt.bar(['Risk-Taking', 'No Risk-Taking'], [mean1, mean2])
    plt.ylabel('Mean Bat Landing Time to Food (seconds)')
    plt.title('Mean Bat Landing Time to Food During Risk-Taking vs No Risk-Taking Behaviour')
    #Adding standard deviation bar
    plt.errorbar(['Risk-Taking', 'No Risk-Taking'], [mean1, mean2], yerr=[stdev1, stdev2], fmt='o', color='black', capsize=5)
    #Adding text labels on top of bars
    print(f"Mean Bat Landing Time (Risk-Taking): {mean1:.2f}, (No Risk-Taking): {mean2:.2f}, Stdev (Risk-Taking): {stdev1:.2f}, (No Risk-Taking): {stdev2:.2f}")
    for i, v in enumerate([mean1, mean2]):
        plt.text(i, v, f'{v:.1f}', ha='center', va='bottom')
    plt.show()


    print(mean1,mean2)

def exploratory_analysis(df1, df2):
    df1_features = ['bat_landing_to_food','rat_time','seconds_after_rat_arrival','risk','reward','sunset_time','hours_after_sunset']

    sns.pairplot(df1[df1_features])
    sns.pairplot(df2)
    plt.show()

def multiple_plot(Dataset_1=df1, Dataset_2=df2):



    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Distributions of Bat Landings and Rat Arrivals", fontsize=16)

    # Histogram: Bat landings when rat present
    axs[0, 0].hist(Dataset_2[Dataset_2["rat_arrival_number"] >= 1]['bat_landing_number'], bins=30, edgecolor='black')
    axs[0, 0].set_title("Bat Landings (Rat Present)")
    axs[0, 0].set_xlabel("Bat landings")
    axs[0, 0].set_ylabel("Frequency")

    # Histogram: Bat landings when no rat present
    axs[0, 1].hist(Dataset_2[Dataset_2["rat_arrival_number"] == 0]['bat_landing_number'], bins=30, edgecolor='black')
    axs[0, 1].set_title("Bat Landings (No Rat Present)")
    axs[0, 1].set_xlabel("Bat landings")
    axs[0, 1].set_ylabel("Frequency")

    # Histogram: Rat arrivals
    axs[0, 2].hist(Dataset_2['rat_arrival_number'], bins=30, edgecolor='black')
    axs[0, 2].set_title("Rat Arrivals (per 30-min)")
    axs[0, 2].set_xlabel("Rat arrivals")
    axs[0, 2].set_ylabel("Frequency")

    # Histogram: Bat landings to food when risk-taking behaviour
    axs[1, 0].hist(Dataset_1[Dataset_1['risk'] == 1]['bat_landing_to_food'], bins=30, edgecolor='black')
    axs[1, 0].set_title("Bat Landings to Food (Risk-Taking)")
    axs[1, 0].set_xlabel("Bat landings to food")
    axs[1, 0].set_ylabel("Frequency")

    # Histogram: Bat landings to food when no risk-taking behaviour
    axs[1, 1].hist(Dataset_1[Dataset_1['risk'] == 0]['bat_landing_to_food'], bins=30, edgecolor='black')
    axs[1, 1].set_title("Bat Landings to Food (No Risk-Taking)")
    axs[1, 1].set_xlabel("Bat landings to food")
    axs[1, 1].set_ylabel("Frequency")

    # Hide the last subplot if not used
    axs[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




if __name__ == "__main__":
    
    # demonstrate(df1)
    # plot_risk_behaviour(df1)
    # plot_bat_landing_time_risk(df1)

    # rat_vs_bat_number(df2) 
    multiple_plot(df1, df2)
    # exploratory_analysis(df1, df2)

