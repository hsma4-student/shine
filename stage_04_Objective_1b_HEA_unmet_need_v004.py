# link to stack overflow post for set up issue I had when running this first time as .py file and getting an error of 'Importing the numpy c-extensions failed'
# https://stackoverflow.com/questions/58868528/importing-the-numpy-c-extensions-failed
# 1. Open VS Code's Command Palette menu by pressing Ctrl+Shift+P or F1
# 2. Choose "Terminal: Select Default Profile" entry
# 3. Then pick "Command Prompt" option 
# 4. Restart VS Code

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
import os
#import openpyxl
import copy

#import stats library for chi square test
from scipy import stats

# -----------------------------------------------------
# <<< set up folder structures >>>
# -----------------------------------------------------

# Check whether the "Assets_produced_by_code folder exists in the current directory.
# If it doesn't exist, create it with the following directory structure:
# Assets_produced_by_code (top level)
# 01_pre_processing_assets<br>
# 02_HEA_assets
# 03_DNA_ML_assets
# 04_Carbon_emissions_assets

#subdirectory folder names for assets folder
preprocessing_assets_path = 'Assets_produced_by_code/01_pre_processing_assets'
hea_assets_path = 'Assets_produced_by_code/02_HEA_assets'
dna_assets_path = 'Assets_produced_by_code/03_DNA_ML_assets'
carbon_emissions_assets_path = 'Assets_produced_by_code/04_Carbon_emissions_assets'

#logic to check if Assets_produced_by_code folder exists, if not, create it with subdirs too
if os.path.exists('Assets_produced_by_code') == False:
    os.makedirs('Assets_produced_by_code')
    os.makedirs(preprocessing_assets_path)
    os.makedirs(hea_assets_path)
    os.makedirs(dna_assets_path)
    os.makedirs(carbon_emissions_assets_path)
    print("New directories have been created to store the outputs from the code.")

print("Existing 'Assets_produced_by_code' directory located.")

# -----------------------------------------------------
#set up parameters
# -----------------------------------------------------

#identify the value for num_bins from the user parameters file
#this will be the number of bin-ranges when splitting up the eligible population in age bands, 
# for the diff in proportion calculation.
filename = "raw_data/user_and_data_parameters/user_and_data_params.xlsx"
num_bins = pd.read_excel(filename, 'HEA_parameters', index_col=None, usecols = "C", header = 1, nrows=0)
num_bins = list(num_bins)[0]

attend_reason_field_name = 'attend_reason'

age_with_median_field_name = 'age' #above is no longer needed as standardised field names in processing file

la_of_interest_user_list_selection = pd.read_excel(filename, 'HEA_parameters', index_col=None, usecols = "C", header = 3, nrows=0) #ref to change zxcv
la_of_interest_user_list_selection = list(la_of_interest_user_list_selection)[0]

#counters for file names
df_counter = 1
chart_counter = 1


# -----------------------------------------------------
#define functions
# -----------------------------------------------------

#define functions
def subset_pop_estimate(source_df):
    """
    function used during the preparation stages to prep for quantifying population size for 
    demographic in scope. 
    This function simply takes the source_df, and retains ALL rows, but ONLY the first 6 columns.
    This is because the ONS public population estimates by single year of age and LSOA are published
    in a format that means the single years of age commence from col 7 onwards. 
    True as of Aug 2022.
    """
    target_df = source_df.iloc[:,:7]        
    return target_df

# -----------------------------------

def sum_age_range(source_df, target_df, min_age, max_age, condition):
    """
    function to sum across columns for each row, from column labelled 'min_age' 
    to column labelled 'max_age' inclusive 
    """
    target_df[f"Total_{min_age}_to_{max_age}_{condition}"] = source_df.loc[:, min_age:max_age].sum(axis=1)
    return target_df

# -----------------------------------

def sum_per_age_group(source_df, target_df, min_age, max_age, condition):
    target_df[f"{condition}_PopAgeRange:_{min_age}-{max_age}"] = source_df.loc[:, min_age:max_age].sum(axis=1)
    return target_df

# -----------------------------------

def calc_prevalence_rate(prevalence_type):
    """
    function to enable user to provide their prevalence rate
    """
    if prevalence_type == 1: #age standardised
        prev_rate_type = "age standardised"
    else: #crude
        prev_rate_type = "crude"
    
    prevalence_per_1k_pop = int(input(f"\nPlease enter the numerator for your {prev_rate_type} prevalence figure for the service in scope. E.g. if the prevalence is 100 per 100,000 population, type 100, then press [ENTER]\n"))
    denominator = int(input("\nPlease enter the denominator for your {prev_rate_type} prevalence rate (e.g. 1000, 10000, 100000, etc.). Don't use commas to separate the figure.\n")) 
    prev_multiplier = prevalence_per_1k_pop / denominator
    return prev_multiplier

# -----------------------------------

def calc_list_percents_for_list_ints(list_of_ints):
    """
    Function to take in a list of integers, and calculate the relative % of the total that each item in the list represents. 
    Then, return this as a new list of float percents, in the same order as the original list. 
    This can be used to then calculate the diff between proportions
    """

    array_of_ints = np.array(list_of_ints)
    total = sum(list_of_ints)
    array_percents = (array_of_ints / total) # * 100
    list_percents = array_percents.tolist()
    return(list_percents)

# -----------------------------------

def calc_diff_between_proportions(list_prop_1, list_prop_2):
    array_prop_1 = np.array(list_prop_1)
    array_prop_2 = np.array(list_prop_2)
    diff_between_props = (array_prop_1 - array_prop_2).tolist()
    return(diff_between_props)

# -----------------------------------
#working code below:
def sum_pop_in_each_age_range_test(num_bins, dict_matched_utlas, la_to_subset_population, dict_condition_age_ranges, condition, dict_condition_bin_range, dict_condition_start_ages, source_df, target_df):
    #take first 6 cols from source df to retain lsoa names, codes etc
    target_df = source_df.iloc[:,:6]
    target_df['UTLA_Name'] = source_df['UTLA19NM']

    #subset the df to only contain rows associated with the UTLA of interest
    mask = target_df['UTLA_Name'] == dict_matched_utlas[la_to_subset_population]
    target_df = target_df[mask]
    
    bin_size = dict_condition_bin_range[condition]
    
    #create list to contain lists of single year of age field names - these are the ages (24, 25, 26.. etc.) col headers in the ons pop estimate lsoa to single year of age file
    list_of_syoa_field_names = []

    #um_items = range(len(dict_condition_start_ages['GUM']))
    for item_num in dict_condition_start_ages[condition]:
        temp_list = []
        for num in range(bin_size):
            temp_list.append(str(item_num + num))
        list_of_syoa_field_names.append(temp_list)

    #list_of_sum_cols = []
    list_col_names = []
    
    for num in range(num_bins):
        list_col_names.append(f"{condition}_PopAged_{dict_condition_age_ranges[condition][num]}")
        target_df[f"{condition}_PopAged_{dict_condition_age_ranges[condition][num]}"] = source_df.loc[:,list_of_syoa_field_names[num]].sum(axis=1)

    return (list_col_names, target_df)

# -----------------------------------

#experimental function code below for more straightforward approach to reading in LA of interest:
def sum_pop_in_each_age_range_revised(num_bins, la_to_subset_population, dict_condition_age_ranges, condition, dict_condition_bin_range, dict_condition_start_ages, source_df, target_df):
    #take first 6 cols from source df to retain lsoa names, codes etc
    target_df = source_df.iloc[:,:6]
    target_df['UTLA_Name'] = source_df['UTLA19NM']

    #subset the df to only contain rows associated with the UTLA of interest
    mask = target_df['UTLA_Name'] == la_to_subset_population
    target_df = target_df[mask]
    
    bin_size = dict_condition_bin_range[condition]
    
    #create list to contain lists of single year of age field names - these are the ages (24, 25, 26.. etc.) col headers in the ons pop estimate lsoa to single year of age file
    list_of_syoa_field_names = []

    #um_items = range(len(dict_condition_start_ages['GUM']))
    for item_num in dict_condition_start_ages[condition]:
        temp_list = []
        for num in range(bin_size):
            temp_list.append(str(item_num + num))
        list_of_syoa_field_names.append(temp_list)

    #list_of_sum_cols = []
    list_col_names = []
    
    for num in range(num_bins):
        list_col_names.append(f"{condition}_PopAged_{dict_condition_age_ranges[condition][num]}")
        target_df[f"{condition}_PopAged_{dict_condition_age_ranges[condition][num]}"] = source_df.loc[:,list_of_syoa_field_names[num]].sum(axis=1)

    return (list_col_names, target_df)

# -----------------------------------

def calc_95_ci_diff_proportions(list_of_conditions, num_bins, dict_condition_service_percents,dict_condition_diff_percents, dict_service_counts_condition_age_range):
    z_score = 1.96

    dict_condition_95_confidence_interval = {}
    dict_condition_95_confidence_interval_sig_bool = {}

    for condition in list_of_conditions:
        pop_size = dict_service_counts_condition_age_range[condition].sum()
        list_confidence_intervals = []
        list_confidence_intervals_significant_or_not = []

        for num in range(num_bins):
            confidence_interval = z_score * ((dict_condition_service_percents[condition][num] * (1 - dict_condition_service_percents[condition][num]) / pop_size) ** 0.5)
            list_confidence_intervals.append(confidence_interval)

            if abs(dict_condition_diff_percents[condition][num]) < confidence_interval:
                list_confidence_intervals_significant_or_not.append('ns')
            else:
                list_confidence_intervals_significant_or_not.append('SIG')
            #list_confidence_intervals_significant_or_not.append(abs(dict_condition_diff_percents[condition][num]) < confidence_interval)
        
        dict_condition_95_confidence_interval[condition] = list_confidence_intervals
        dict_condition_95_confidence_interval_sig_bool[condition] = list_confidence_intervals_significant_or_not
    
    return (dict_condition_95_confidence_interval, dict_condition_95_confidence_interval_sig_bool)

# -----------------------------------

def calc_95_ci_diff_proportions_gender(condition, num_bins, dict_condition_service_percents,dict_condition_diff_percents, dict_service_counts_condition_gender):
    z_score = 1.96
    #dict_condition_95_confidence_interval = {} #produced outside of function for gender, to solve where multiple conditions in scope, each with differing pop genders (persons / males etc.)
    #dict_condition_95_confidence_interval_sig_bool = {} #produced outside of function, as above.

    pop_size = dict_service_counts_condition_gender[condition].sum()
    list_confidence_intervals = []
    list_confidence_intervals_significant_or_not = []

    for num in range(num_bins):
        confidence_interval = z_score * ((dict_condition_service_percents[condition][num] * (1 - dict_condition_service_percents[condition][num]) / pop_size) ** 0.5)
        list_confidence_intervals.append(confidence_interval)

        if abs(dict_condition_diff_percents[condition][num]) < confidence_interval:
            list_confidence_intervals_significant_or_not.append('ns')
        else:
            list_confidence_intervals_significant_or_not.append('SIG')
        #list_confidence_intervals_significant_or_not.append(abs(dict_condition_diff_percents[condition][num]) < confidence_interval)
    
    #dict_condition_95_confidence_interval[condition] = list_confidence_intervals
    #dict_condition_95_confidence_interval_sig_bool[condition] = list_confidence_intervals_significant_or_not
    
    return (list_confidence_intervals, list_confidence_intervals_significant_or_not)

# -----------------------------------

#function to provide numbered prefixes to file names for files saved as outputs from the code
def number_saved_file(counter):
    """
    To be used as follows: 
    file_prefix = number_saved_file(counter)
    """

    #check if counter between 1 and 9, if so return counter prefixed by two leading zeros.
    #e.g. if counter = 1, this would return 001
    if counter in range (1,10):
        return(f"00{counter}")
    elif counter in range(10,100):
        return(f"0{counter}")
    else:
        return(counter)

# -----------------------------------

def create_bar_plot(chart_counter, string_for_title, string_for_file_name, list_of_conditions, dict_condition_demographic_labels, dict_condition_service_percents, dict_condition_population_percents, x_tick_rotation_int, dict_condition_95_confidence_interval, service_colour, pop_colour, dict_condition_95_confidence_interval_sig_bool, dict_condition_demographic_file_path):

    """
    Function to plot a chart for each condition the user is using the code for
    and compare the service proportionate make up for the given feature of interest
    to the population make up for the same feature.

    note: this code runs on matplotlib version 3.2.2 
    
    more recent versions of matplotlib use a different method for setting xticklabels.

    if the code doesn't run and fails in this function, check the version of matplotlib.

    if the version is >3.2.2 then add the 'label' kwarg to the ax.bar calls in the function,
    with the value dict_condition_demographic_labels[condition] as in label = dict_condition_demographic_labels[condition]
    
    """    
    
    def addlabels(x,service,sig_or_not):
        for i in range(x):
            plt.text(i,service[i],sig_or_not[i])

    dict_condition_chart_file_paths = {}

    for condition in list_of_conditions:
        
        N = len(dict_condition_demographic_labels[condition])
        service = dict_condition_service_percents[condition]
        service_std = dict_condition_95_confidence_interval[condition]

        fig, ax = plt.subplots()

        ind = np.arange(N)    # the x locations for the groups
        width = 0.35         # the width of the bars
        #c = '#41B6E6' #NHS Light Blue
        c = service_colour #NHS Light Blue 
        
        ax.bar(ind, service, width, bottom=0, yerr=service_std, label='Service', color = c)

        pop = dict_condition_population_percents[condition]
        
        #c = '#768692' #NHS Mid Grey
        c = pop_colour

        ax.bar(ind + width, pop, width, bottom=0, label='Population', color = c)

        ax.set_title(f"{string_for_title} ({condition})")
        
        addlabels(N,service,dict_condition_95_confidence_interval_sig_bool[condition])

        ax.set_xticks(ind + width / 2) 
        ax.set_xticklabels(dict_condition_demographic_labels[condition])
        plt.xticks(rotation = x_tick_rotation_int) #new line to set rotation of x labels
        
        ax.set_ylabel('Proportion per category')

        ax.legend()
        ax.autoscale_view()

        file_prefix = number_saved_file(chart_counter)
        temp_filename = f"chart{file_prefix}_{string_for_file_name}-{condition}.png"
        file_path = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}'
        #plt.savefig(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}')
        plt.savefig(file_path, bbox_inches='tight')
        dict_condition_chart_file_paths[condition] = file_path
        chart_counter+=1
        print("File saved.")

        plt.show()
    
    dict_condition_demographic_file_path['charts'] = dict_condition_chart_file_paths

    return (chart_counter, dict_condition_demographic_file_path)


# -----------------------------------

#now we need to get the total Male and Female population, respectively, for the relevant age ranges, for each condition in scope
def calc_male_and_female_pop_size(df_males, la_of_interest_user_list_selection, df_females, dict_min_age_for_each_condition, dict_max_age_for_each_condition, list_of_conditions, dict_prev_for_each_condition):
    """
    function to subset the single year of age dataframes for males and females, for each condition, as applicable to the age range for that condition, and derive the male/female pop estimate
    """

    #MALES
    dict_condition_subtotal_male_pop_size = {}
    dict_condition_subtotal_female_pop_size = {}

    for condition in list_of_conditions:
        #filter males df to just the la of interest
        #df_males['UTLA19NM'] = df_males['UTLA19NM'].str.lower()
        #mask = df_males['UTLA19NM'] == dict_matched_utlas[la_to_subset_population] #original
        mask = df_males['UTLA19NM'] == la_of_interest_user_list_selection
        df_males = df_males[mask]

        #get sub-total male population for the age range as relevant to this condition
        male_pop_total = df_males.loc[:, dict_min_age_for_each_condition[condition]:dict_max_age_for_each_condition[condition]].sum(axis=1).sum()
        dict_condition_subtotal_male_pop_size[condition] = int(round(male_pop_total * dict_prev_for_each_condition[condition], 0))
        
        #debug print
        #print(dict_condition_subtotal_male_pop_size[condition])
    
    #FEMALES
        #filter females df to just the la of interest
        #df_females['UTLA19NM'] = df_females['UTLA19NM'].str.lower()
        #mask = df_females['UTLA19NM'] == dict_matched_utlas[la_to_subset_population]
        mask = df_females['UTLA19NM'] == la_of_interest_user_list_selection
        df_females = df_females[mask]

        #get sub-total male population for the age range as relevant to this condition
        female_pop_total = df_females.loc[:, dict_min_age_for_each_condition[condition]:dict_max_age_for_each_condition[condition]].sum(axis=1).sum()
        dict_condition_subtotal_female_pop_size[condition] = int(round(female_pop_total * dict_prev_for_each_condition[condition], 0))

        #debug print
        #print(dict_condition_subtotal_female_pop_size[condition])

    dict_condition_gender_list_F_M = {}

    for condition in list_of_conditions:
        temp_list = []
        temp_list.append(dict_condition_subtotal_female_pop_size[condition])
        temp_list.append(dict_condition_subtotal_male_pop_size[condition])
        dict_condition_gender_list_F_M[condition] = temp_list

    return(dict_condition_gender_list_F_M)


# -----------------------------------
# NEW CODE ENTERED 07/09/22 TO READ-IN USER PARAMS AND SIGNIFICANTLY REDUCE USER INTERACTION WITH THE CODE

def create_condition_to_param_dict(df, desired_col_list, value_column_string, number_of_conditions,cast_as_type):
    
    """
    Function to produce a subset df from a source, and turn this into a dictionary consisting of condition as the key and the associated values of the target column name in the source df
    """
    
    subset_df = df.iloc[ : , desired_col_list]

    #convert to a dictionary
    temp_dict = subset_df.to_dict()

    #populate a dict to subsequent populate with condition (key) to prev rate multiplier (values)
    dict_condition_to_col_of_interest = {}

    for num in range(number_of_conditions):
        #extract required content from temp nested dict created above
        condition = temp_dict['condition'][num]
        col_of_interest = cast_as_type(temp_dict[value_column_string][num])

        dict_condition_to_col_of_interest[condition] = col_of_interest

    
    return dict_condition_to_col_of_interest

# -----------------------------------

def run_chi_square_test(
        dict_condition_category_labels_1,
        dict_condition_category_service_counts_2,
        dict_condition_category_pop_counts_4,
        dict_condition_category_pop_counts_as_percents_5
):
    """
    Function to run the chi-square goodness of fit test. 
    The output of this function has been validated using the following website: https://www.statskingdom.com/310GoodnessChi.html
    The results from this function matched identically the results from the above website.
    """
    #scratch cell to solve problem
    #service level data
    #dict_condition_age_ranges #age ranges (demographic categories) #1
    #dict_service_counts_condition_age_range #service counts by age range #2

    #dictionary comprehension to derive sum of the count of patients in all ages, for each condition in the SERVICE #3
    dict_service_total_count_per_condition_3 = {key: dict_condition_category_service_counts_2[key].sum() for key in dict_condition_category_service_counts_2.keys()} 

    #population level comparative data
    #dict_condition_age_sub_totals #population counts by age range #4
    #dict_condition_population_percents_age #list of population counts as percent of whole in each age range, #5

    #derive the EXPECTED counts in the service data IF the proportions seen in the population were to be applied to the total patients seen in the servce #6
    dict_expected_service_counts_per_condition_6 = {}
    for key in dict_service_total_count_per_condition_3.keys():
        list_service_expected_counts = [dict_service_total_count_per_condition_3[key] * x for x in dict_condition_category_pop_counts_as_percents_5[key]]
        dict_expected_service_counts_per_condition_6[key] = pd.Series(list_service_expected_counts)

    #dictionary comprehension to derive sum of the count of patients in all ages, for each condition in the POPULATION #7 (sum of #4)
    dict_pop_total_count_per_condition = {key: pd.Series(dict_condition_category_pop_counts_4[key]).sum() for key in dict_condition_category_pop_counts_4.keys()}

    #run chi square test for each condition
    dict_chi_square_test_results_per_condition = {}
    dict_chi_square_test_statistic_per_condition = {}
    dict_chi_square_test_pvalue_per_condition = {}
    dict_chi_square_test_sig_at_alpha_per_condition = {}

    for key in dict_condition_category_labels_1.keys():
        chi_square_test = stats.chisquare(dict_condition_category_service_counts_2[key], dict_expected_service_counts_per_condition_6[key])

        #assign the 2 outputs from the chi square test above to their own variables
        chi_square_statistic = chi_square_test[0]
        chi_square_pvalue = chi_square_test[1]

        #check whether the chi_square_pvalue is less than 0.05 (assuming this is the chosen significance level)
        ahlpa=0.05
        #chi_square_sig_at_alpha contains a Bool value of True if the result was significant. 
        chi_square_sig_at_alpha = chi_square_pvalue < 0.05


        dict_chi_square_test_results_per_condition[key] = chi_square_test
        dict_chi_square_test_statistic_per_condition[key] = chi_square_statistic
        dict_chi_square_test_pvalue_per_condition[key] = chi_square_pvalue
        dict_chi_square_test_sig_at_alpha_per_condition[key] = chi_square_sig_at_alpha

        #check the chi square assumptions that the min value for observed and expected counts is >=5
        #This populates a dictionary with keys of condition label and Bool values, indicating whether the requirement for all expected and observed counts 
        #to be at least 5 is met. Where this condition is met, a Bool of True is present, otherwise, False is present.
        dict_chi_square_conditions_met = {}
        for key in dict_expected_service_counts_per_condition_6.keys():
            dict_chi_square_conditions_met[key] = dict_expected_service_counts_per_condition_6['GUM'].min() >= 5 and dict_condition_category_service_counts_2[key].min() >= 5

    return(dict_chi_square_test_results_per_condition, dict_chi_square_test_statistic_per_condition, dict_chi_square_test_pvalue_per_condition, dict_chi_square_test_sig_at_alpha_per_condition)


# -----------------------------------
def summarise_chi_square_results_into_df(
    list_of_conditions,
    dict_chi_square_test_results_per_condition,
    dict_chi_square_test_statistic_per_condition,
    dict_chi_square_test_pvalue_per_condition,
    dict_chi_square_test_sig_at_alpha_per_condition,
    master_list_of_data_frames,
    demographic_label
):

    """
    Function to summarise the results from the chisquare test for a given demographic into a df
    append this to the master list of data frames so that once all results are converted and appended
    to that list of dataframes, the whole list of dataframes can be concatenated into a single df of results
    """

    list_of_data_frames = []
    for condition in list_of_conditions:
        
        if condition in dict_chi_square_test_results_per_condition.keys():

            temp_dict = {}
            temp_dict['Condition'] = condition
            temp_dict['Demographic'] = demographic_label
            temp_dict['Chi_square_statistic'] = round(dict_chi_square_test_statistic_per_condition[condition],5)
            temp_dict['P_value'] = round(dict_chi_square_test_pvalue_per_condition[condition],5)
            
            if dict_chi_square_test_sig_at_alpha_per_condition[condition] == True:
                temp_dict['Significant?'] = 'yes'
            else:
                temp_dict['Significant?'] = 'no'
            
            temp_df = pd.DataFrame.from_dict(temp_dict, orient='index').T
            list_of_data_frames.append(temp_df)
    
    for list_item in list_of_data_frames:
        master_list_of_data_frames.append(list_item)
    
    return(master_list_of_data_frames)


# -----------------------------------
#Read-in user parameters
# -----------------------------------

#code to identify the number of conditions the user is running the tool for
filename = "raw_data/user_and_data_parameters/user_and_data_params.xlsx"
number_of_conditions_modelling_for = pd.read_excel(filename, 'HEA_condition_details', index_col=None, usecols = "D", header = 1, nrows=1)
number_of_conditions_modelling_for = int(number_of_conditions_modelling_for.iloc[0,0])

#read in the params defined by the user for each condition in scope. This will be used in subsequent steps to create a number of dictionaries required for HEA code to run.
#This approach removes a number of steps of user interaction with the code in the development environment, and enables separation of concerns in terms of sourcing ref material for the model 
# and actually running the model. 

df_condition_params_original = pd.read_excel(filename, 'HEA_condition_details', index_col=None, usecols = "B, C, D, E, F, G, H, I, J", header = 5, nrows=number_of_conditions_modelling_for)
df_condition_params_original = df_condition_params_original.fillna(1)

#ID the column description ised in the params file
column_names_from_condition_df = list(df_condition_params_original.columns)

#create variables containing the text for each column name from the user params df read in above. By doing this, if there is a change to the word of the column, so long as the shape and order doesnt change
#the code will still run as we aren't hard-coding the text as a reference point.
temp_condition_col_name = column_names_from_condition_df[0]
temp_prev_type_col_name = column_names_from_condition_df[1]
temp_numerator_col_name = column_names_from_condition_df[2]
temp_denominator_col_name = column_names_from_condition_df[3]
temp_min_age_col_name = column_names_from_condition_df[4]
temp_max_age_col_name = column_names_from_condition_df[5]
temp_gender_col_name = column_names_from_condition_df[6]
temp_proportion_this_need_represents = column_names_from_condition_df[7]
temp_proportion_seen_by_this_service = column_names_from_condition_df[8]

df_condition_params_original['prevalence_multiplier'] = ((df_condition_params_original[temp_numerator_col_name] / df_condition_params_original[temp_denominator_col_name]) / df_condition_params_original[temp_proportion_this_need_represents]) * df_condition_params_original[temp_proportion_seen_by_this_service]

#rename the df cols to have shorter meaningful names rather than the instructional column names from the user params file. 
df_condition_params_original.rename(columns= {
    temp_condition_col_name: "condition", 
    temp_prev_type_col_name: "prevalence_type",
    temp_numerator_col_name: "numerator",
    temp_denominator_col_name: "denominator",
    temp_min_age_col_name: "min_age",
    temp_max_age_col_name: "max_age",
    temp_gender_col_name: "gender",
    temp_proportion_this_need_represents: "proportion_need_represents",
    temp_proportion_seen_by_this_service: "proportion_seen_by_service",
    }, 
    inplace=True)

#create variables containing the text for the shorter column names we just assigned
revised_column_names_from_condition_df = list(df_condition_params_original.columns)

condition_col_name = revised_column_names_from_condition_df[0]
prev_type_col_name = revised_column_names_from_condition_df[1]
numerator_col_name = revised_column_names_from_condition_df[2]
denominator_col_name = revised_column_names_from_condition_df[3]
min_age_col_name = revised_column_names_from_condition_df[4]
max_age_col_name = revised_column_names_from_condition_df[5]
gender_col_name = revised_column_names_from_condition_df[6]
prev_multiplier_col_name = revised_column_names_from_condition_df[9]

#create list_of_conditions variable
list_of_conditions = df_condition_params_original[condition_col_name].tolist()

#create dictionary of condition to prevalence rate type
dict_condition_to_prevalence_rate = create_condition_to_param_dict(df_condition_params_original, [0, 1], prev_type_col_name, number_of_conditions_modelling_for, str)

#use above dict of condition to prevalence rate type to 
# create dictionary of condition to prevalence rate reference integer where
#1 = Age standardised rate
#2 = Crude rate
#3 = Census or no rate

dict_prev_ref_for_each_condition = {}

for condition in list_of_conditions:
    ref_num = int(dict_condition_to_prevalence_rate[condition].split(":")[0])
    dict_prev_ref_for_each_condition[condition] = ref_num

#create multiplier for prev rate multiplier
dict_prev_for_each_condition = create_condition_to_param_dict(df_condition_params_original, [0, 9], prev_multiplier_col_name, number_of_conditions_modelling_for, float)

#create dictionary of condition name to min age
dict_min_age_for_each_condition = create_condition_to_param_dict(df_condition_params_original, [0, 4], min_age_col_name, number_of_conditions_modelling_for, str)

#create dictionary of condition name to max age
dict_max_age_for_each_condition = create_condition_to_param_dict(df_condition_params_original, [0, 5], max_age_col_name, number_of_conditions_modelling_for, str)

#create dictionary of condition name to gender seen
dict_pop_gender_for_each_condition_text = create_condition_to_param_dict(df_condition_params_original, [0, 6], gender_col_name, number_of_conditions_modelling_for, str)
#use the above dict to create a dictionary of condition to gender integer, where
#1 = Persons
#2 = Males only
#3 = Females only

dict_pop_gender_for_each_condition = {}

for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition_text[condition] == "Persons":
        dict_pop_gender_for_each_condition[condition] = 1
    elif dict_pop_gender_for_each_condition_text[condition] == "Males only":
        dict_pop_gender_for_each_condition[condition] = 2
    else:
        dict_pop_gender_for_each_condition[condition] = 3

# -----------------------------------
# <<< Code Begins >>>
# -----------------------------------

#Read in population estimates by LSOA and identify the fields included in each
#Note: added _ to the end of each df var name due to needing to join the upper tier la df for the utla name.
df_all_persons_ = pd.read_csv("raw_data/ons_population_data/2020_persons_pop_lsoa_syoa.csv")
df_males_ = pd.read_csv("raw_data/ons_population_data/2020_males_pop_lsoa_syoa.csv", encoding='latin-1')
df_females_ = pd.read_csv("raw_data/ons_population_data/2020_females_pop_lsoa_syoa.csv")

#read in the lower tier to upper tier look up file from open geography portal
df_lower_tier_to_upper_tier_la = pd.read_csv("raw_data/open_geography_portal_lookups/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority__April_2019__Lookup_in_England_and_Wales.csv")

#create dataframes for pop size, with the upper tier la code and name joined to the end (right) of the original dataframe
df_all_persons = pd.merge(df_all_persons_, df_lower_tier_to_upper_tier_la, left_on='LA Code (2018 boundaries)', right_on='LTLA19CD')
df_males = pd.merge(df_males_, df_lower_tier_to_upper_tier_la, left_on='LA Code (2018 boundaries)', right_on='LTLA19CD')
df_females = pd.merge(df_females_, df_lower_tier_to_upper_tier_la, left_on='LA Code (2018 boundaries)', right_on='LTLA19CD')

#read in the lsoa to imd decile data set, available from : https://data-communities.opendata.arcgis.com/datasets/communities::indices-of-multiple-deprivation-imd-2019-1/about
df_lsoa_imd_full = pd.read_csv('raw_data/ministry_of_housing_communities_local_gov/Indices_of_Multiple_Deprivation_(IMD)_2019.csv')
df_lsoa_imd_only = pd.DataFrame()
df_lsoa_imd_only['lsoa11cd'] = df_lsoa_imd_full['lsoa11cd']
df_lsoa_imd_only['imd_decile'] = df_lsoa_imd_full['IMD_Decile']
#df_lsoa_imd_only.head()

#NEW code to merge imd decile into the df - need to run through file to check this hasnt introduced bugs
df_all_persons = pd.merge(df_all_persons, df_lsoa_imd_only, left_on='LSOA Code', right_on='lsoa11cd')
df_males = pd.merge(df_males, df_lsoa_imd_only, left_on='LSOA Code', right_on='lsoa11cd')
df_females = pd.merge(df_females, df_lsoa_imd_only, left_on='LSOA Code', right_on='lsoa11cd')

#get all column names and assign to variables for each pop type
all_persons_fields = df_all_persons.columns
males_fields = df_males.columns
females_fields = df_females.columns

#Create empty data frames to build up in subsequent steps with the identified
#population age range
df_all_persons_selected_age = pd.DataFrame()
df_males_selected_age = pd.DataFrame()
df_females_selected_age = pd.DataFrame()

#---------------------------------------------------

#altered dict content - check hasn't thrown any code out in UAT
dict_prevalence_rate = {
    1: "Age standardised rate",
    2: "Crude rate",
    3: "No rate - census only"
}

#---------------------------------------------------
#create list of conditions to include in the model

"""#add error control (e.g. must be at least 1 condition ? )
number_of_conditions_modelling_for = int(input("Please enter the number of conditions/services you would like to include in this Health Equity Assessment. \nIn doing this, consider how granular you would like to estimate unmet need. \nIt may be easier to start with high level groupings rather than individual conditions. For example, for sexual health services, there are two primary uses of the service, genitourinary conditions and contraceptive requirements. \n>>> "))

list_of_conditions = []
for condition_number in range(number_of_conditions_modelling_for):
    condition_name = input(f"Please enter ONE word to describe the nature of condition/service {condition_number + 1}. PLEASE NOTE: The word you use will appear in the final report and all charts produced related to this condition.: >>> ")
    list_of_conditions.append(condition_name)
"""
#---------------------------------------------------

#ITEMS TO SAVE TO SAVE INTO THE WORD REPORT
#number_of_conditions_modelling_for #can use this in report to summarise "the HEA model has been used for one condition (condition)"

if number_of_conditions_modelling_for == 1:
    text_condition_summary = f"The HEA model has been used to model one condition ({list_of_conditions[0]})."
elif number_of_conditions_modelling_for == 2:
    text_condition_summary = f"The HEA model has been used to model two conditions ({' and '.join(list_of_conditions)})."
else:
    text_condition_summary = f"The HEA model has been used to model {number_of_conditions_modelling_for} conditions ({f', '.join(list_of_conditions[:-1]) + ' and ' + list_of_conditions[-1]})."

#Sentence to use in the report summarising the conditions being modelled = text_condition_summary  

temp_filename = "text_condition_summary.txt"
filepath = f'{hea_assets_path}/{temp_filename}'

#open text file
text_file = open(filepath, "w")
 
#write string to file
text_file.write(text_condition_summary)
 
#close file
text_file.close()
print("Summary of conditions being modelled saved as txt file.")

#---------------------------------------------------

#DEV NOTE: "while try except" needed here to control for data entry error
if number_of_conditions_modelling_for >1:
    plural = "genders"
else:
    plural = "gender"

#---------------------------------------------------
#Preparation finished
#---------------------------------------------------
# <<< HEA Starts Here : >>>
#---------------------------------------------------

#This code is used to calculate equal age ranges (bin_range) between the min and max age the user has said the service sees. 
#The // in the calculation provides a rounded down answer. This ensures whole ages and equal ranges are returned.
#The trade-off is where a service_age_range cannot be divided equally by the num_bins parameter, the last (oldest) age range will not extend all 
#the way to the oldest age the service sees. The margin of error will be anything from 1 to bin_range-1 years. 
#So, for example, if bin_range is equal to 10, the above trade-off means the data used to calculate any statistical difference between proportions
#could exclude upto the oldest 9 years of patients in the service (but, it would also exclude this same 9 years from the population comparison,
# so we are still 'comparing apples with apples')

#dictionary to store the bin range for each condition
dict_condition_bin_range = {}

#dictionary of user_entered condition names (keys) to a list of the start ages for the age ranges (values)
dict_condition_start_ages = {}

#dictionary of user_entered condition names (keys) to a list of the end ages for the age ranges (values)
dict_condition_end_ages = {}

# -----------------------------
#AGE SECTION STARTS
# -----------------------------

#loop over all conditions the user has entered, derive start and end ages for equal age ranges, with the number of age ranges according to user parameter.
#update the above two dictionaries with the outputs from each loop
for condition in list_of_conditions:
    min_age = int(dict_min_age_for_each_condition[condition])
    max_age = int(dict_max_age_for_each_condition[condition])
    
    service_age_range = max_age - min_age
    bin_range = service_age_range // num_bins
    dict_condition_bin_range[condition] = bin_range #store the derived bin range in the dictionary 
    print(f"The interval for each age range for the condition type '{condition}' is : {bin_range} years.")

    list_start_ages = []
    list_end_ages = []

    #using list comprehension, derive the start ages for each age range relevant to the user-entered parameters
    condition_start_ages = [list_start_ages.append(min_age + age) for age in range(0, service_age_range, bin_range)]
    dict_condition_start_ages[condition] = list_start_ages

    #using list comprehension, derive the end ages for each age range relevant to the user-entered parameters
    condition_end_ages = [list_end_ages.append(min_age + age + (bin_range-1)) for age in range(0, service_age_range, bin_range)]
    dict_condition_end_ages[condition] = list_end_ages

    #print(f"{condition} : {dict_condition_end_ages}") # test print - delete


#check whether the number of start and end ages for each age range is equal to the number of bins.
#if it is, add one more start and end age to the respective lists, using the appropriate bin range interval from the dict created above
#This is to ensure the age ranges can be correctly applied to the df shortly (that step requires that labels and bins are not the same length)
for condition in list_of_conditions:
    if len(dict_condition_start_ages[condition]) == num_bins:
        dict_condition_start_ages[condition].append(dict_condition_start_ages[condition][-1]+dict_condition_bin_range[condition])
    if len(dict_condition_end_ages[condition]) == num_bins:
        dict_condition_end_ages[condition].append(dict_condition_end_ages[condition][-1]+dict_condition_bin_range[condition])

#create a dictionary for later use (in charts etc.) which contains the concatenated string of the age_range for each bin/group
dict_condition_age_ranges = {}

for condition in list_of_conditions:
    temp_list_age_ranges = []
    for num in range(num_bins):
        temp_list_age_ranges.append(f"{dict_condition_start_ages[condition][num]}-{dict_condition_end_ages[condition][num]}")

    dict_condition_age_ranges[condition] = temp_list_age_ranges

#test print - delete in final
print()
print(dict_condition_age_ranges)
print(dict_condition_start_ages)
print(dict_condition_end_ages)
print()
#---------------------------------------------------
#read in the processed HEA file (not encoded) from the pre-processing file

processed_hea_file = pd.read_csv('processed_data/1A_HEA/processed_data_1A_HEA_not_encoded_standard_col_names.csv')

#---------------------------------------------------

#Revised to have >1 attendance reason associated with each condition in scope
#This solves the problem for sexual health services, for example, where attendance reason may be GU, Con, or Both
# Where the total contraception activity is Con + Both, and the total GU activity is GU + Both
# as 'Both' is a dual appointment reason. 

#mapping user entered conditions to the attend_reasons present in the dataset (this works for now, but an alternative could be to incorporate this into the user paraems xlsx file)
print(text_condition_summary)
print()
print(f"We now need to subset the dataset so we have a new dataset for each of these conditions.")
print()
print("Below is a numbered list of all attendance reasons present in the dataset:")

dict_condition_list_attend_reasons = {}

for condition in list_of_conditions:
    dict_attend_reasons = {}
    counter = 1 #was 0

    #The first line in the for loop below is a way of reliably ensuring the dictionary created for the user to subsequently select / assign the attendance reason to the 
    #condition name is in the same order (alphabetical) every time the code is run.
    for reason in list(dict.fromkeys(sorted(processed_hea_file[attend_reason_field_name].squeeze()))):
        dict_attend_reasons[counter] = reason
        counter += 1

    #for num in list(range(counter))[:-1]:
    #    print(f"{num+1}: {dict_attend_reasons[num+1]}")

    unique_options_attend_reason = list(dict.fromkeys(sorted(processed_hea_file[attend_reason_field_name].squeeze())))
                
    attend_reasons_for_this_condition = []

    #loop to choose which fields to drop from the missing value process
    while True:
        print("\nThe attendance reasons to choose from are listed below:")
        for key_number in dict_attend_reasons:
            print(f"{key_number} : {dict_attend_reasons[key_number]}")
        print("0 : To continue with the current selection")

        while True:
            try:
                attend_reason_selection = int(input(f"\nPlease enter the number of the attendance reason associated with the {condition.upper()} condition, or press 0 to continue with the current selection:"))
                if attend_reason_selection in range(counter):
                    break
                else:
                    print("\nThat is not a correct selection.")
            except ValueError:
                print("Value Error - try again.")

        if attend_reason_selection != 0:
            attend_reasons_for_this_condition.append(dict_attend_reasons[attend_reason_selection]) #create new list of appt statuses user wants to exclude
            del dict_attend_reasons[attend_reason_selection]
        elif attend_reason_selection == 0:
            break
    dict_condition_list_attend_reasons[condition] = attend_reasons_for_this_condition

#---------------------------------------------------

#converted below to doc string as doesnt look to be used, if breaks code, un-doc and fix
"""
column_headers = list(processed_hea_file.columns)
temp_df = pd.DataFrame(columns=column_headers)
#temp_df

for condition in list_of_conditions:
    for reason in dict_condition_list_attend_reasons[condition]:
        print(f"{condition} : {reason}")
"""

#---------------------------------------------------

#read in the processed file output from the pre-processing code
#processed_hea_file = pd.read_csv('processed_data/1A_HEA/processed_data_1A_HEA_not_encoded.csv') #original file read in, included col names with the "_with_missing" and "_with_median" suffixes
processed_hea_file = pd.read_csv('processed_data/1A_HEA/processed_data_1A_HEA_not_encoded_standard_col_names.csv') #new file identical content as above line, but uses original col names

#create nested dictionaries to populate with dfs for each attendance reason associated with each condition
nested_dict_condition_attend_reason = {}
nested_dict_condition_attend_reason_all_ages = {}

for condition in list_of_conditions:
    #create a temp dictionary to populate with attend reasons (key(s) ) as relevant to this condition, and values of filtered df to these attend reasons
    temp_dict = {}
    temp_dict_all_ages = {}

    for attendance_reason in dict_condition_list_attend_reasons[condition]:

        mask = processed_hea_file[attend_reason_field_name] == attendance_reason #create mask for each attend reason in attend_reason
        select_attend_reason_df = processed_hea_file[mask] #filter using mask

        #then for each df, add new column to indicate the relevant bin age range the row relates to from dict[condition]: df of age ranges
        bins = dict_condition_start_ages[condition]
        labels = dict_condition_age_ranges[condition]

        select_attend_reason_df['age_group'] = pd.cut(select_attend_reason_df[age_with_median_field_name], bins=bins, labels=labels, right=False)
        #the output from the above code returns NaN for age values beyond the upper limit, so remove these
        select_attend_reason_df_dropna = select_attend_reason_df.dropna(subset = ['age_group'])

        #add this subset df to the dictionary above, with condition name as the key, subset df with age ranges as applicable to that specific condition, as values
        temp_dict[attendance_reason] = select_attend_reason_df_dropna

        #two-step process to subset the df to ages within the user-entered age-range (min to max age)
        select_attend_reason_df_ages_in_range_step_1 = select_attend_reason_df[(select_attend_reason_df[age_with_median_field_name] >= int(dict_min_age_for_each_condition[condition]))]
        select_attend_reason_df_ages_in_range_step_2 = select_attend_reason_df_ages_in_range_step_1[(select_attend_reason_df_ages_in_range_step_1[age_with_median_field_name] <= int(dict_max_age_for_each_condition[condition]))]
        temp_dict_all_ages[attendance_reason] = select_attend_reason_df_ages_in_range_step_2

    #after iterating over all attendance reasons the user selected for this condition (the nested for loop), repeat the sequence for the next condition the user is modelling for
    nested_dict_condition_attend_reason[condition] = temp_dict
    nested_dict_condition_attend_reason_all_ages[condition] = temp_dict_all_ages

#Once the nested dictionary is populated, for all conditions in scope, we need to combine all dfs created for each condition, so we have a single df for each condition

dict_condition_dataset = {}
dict_condition_dataset_all_ages_in_range = {}

for condition in list_of_conditions:
    dict_condition_dataset[condition] = pd.concat(nested_dict_condition_attend_reason[condition].values(), ignore_index=True)
    dict_condition_dataset_all_ages_in_range[condition] = pd.concat(nested_dict_condition_attend_reason_all_ages[condition].values(), ignore_index=True)

#---------------------------------------------------


#Copied this cell from the statistical difference between proportions (gender) section
#By placing this code here, we can then reliably subset the df at this point based on the 
# selected gender (persons, males, females) which will allow accuracte count per age group and 
# estimate of unmet need in due course #when running the code from scratch, 
# check this doesnt have independancies with variables created later in the code

#DEV NOTE: check if 'missing' appears in gender_with_missing col, when using 'persons' and 
# not male or female. if it does, update the 'if' section of the below if elif else, 
# to remove rows with gender of 'missing'. 

#read in the gender sheet from user params as a new df
df_gender_groups_to_high_level = pd.read_excel(filename, 'Gender_mapping', skiprows=1)

#create a dict to populate with service data (granular) gender categories (key) to high level categories (values)
dict_service_gender_groups_to_high_level = {}

#create a tuple from each row in the df, and populate the above dict. Due to the ONS single year of age estimates only being available for genders Male and Female, the user params file has been limited accordingly.
#This means the code below will return NaN for any gender value present without an assigned Male or Female string in the user params file.
for row in df_gender_groups_to_high_level.itertuples():
    dict_service_gender_groups_to_high_level[row.Service_data_categories] = row.High_level_categories

# ----------------------------

#identify the string used as the field name for the gender field from the user parameters file
gender_field_name = 'gender_at_birth'
print(f"The field name that will be used for the gender field is: {gender_field_name}")

# --------------------------------

#This loop produces a version of the df as applicable to the users' selections for gender (all persons, males only, females only) and age range already applied at this point.

#create a new dictionary called dict_condition_dataset_mapped_gender_subset
        #This is essentially the same as dict_condition_dataset_mapped_gender but this time, the dataframe values in the dictionary have had all rows with associated gender values other than Male removed
        #This reflects the user input / choice that their service works with just Males

dict_condition_dataset_mapped_gender_subset = {}
dict_condition_dataset_all_ages_in_range_mapped_gender_subset = {}


for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1: #persons
        #pass #basically, 'do nothing' and retain the whole persons df as per any processing completed to this point.  

        #replace the granular gender categories in the df with the high level / less granular gender groups we are limited to, from the user parameters file
        list_genders_present_in_service_data = list(dict_service_gender_groups_to_high_level.keys())

        #create a copy of the dictionary consisting of condition name (keys) and filtered df (values)
        #We are creating a copy here because the subsequent lines of code in this section will replace the (potentially more granular) gender labels with less granular groups/labels
        #We are only using Male and Female gender in this code because we are limited by the available data sources (at the time of writing, we could only source small area population estimates by Male/Female)
        dict_condition_dataset_mapped_gender = {k:v for k, v in dict_condition_dataset.items()}

        #loop through conditions present, then for each condition, loop through the genders present in the df and replace these with the high level groups from the user parameters file
        #at this point, due to the inability to locate small area population estimates by any gender other than Male or Female, this assigns the label 'Currently excluded - population data not available' to
        #any gender label other than those assigned by the user to Male or Female in the user_and_data_params.xlsx set up file.
        #dict_condition_dataset_all_ages_in_range_mapped_gender_subset = {}
        
        #for condition in list_of_conditions:
        for gender in list_genders_present_in_service_data:
            
            #new line 03/8 - intention is to replace gender labels in the df that contains all rows from min to max age for the condition, even if they fall outside the last complete age-range 'bin'
            dict_condition_dataset_all_ages_in_range[condition][gender_field_name] = dict_condition_dataset_all_ages_in_range[condition][gender_field_name].str.replace(gender, dict_service_gender_groups_to_high_level[gender])

            #filtering index where gender_with_missing == 'missing'
            indexes = dict_condition_dataset_all_ages_in_range[condition][dict_condition_dataset_all_ages_in_range[condition][gender_field_name] == 'Currently excluded - population data not available'].index
    
            #droping rows based on gender_with_missing column value of 'missing' - effectively this means the percentages for gender are based on 'patients with a gender recorded'
            dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition] = dict_condition_dataset_all_ages_in_range[condition].drop(indexes)

            #filtering index where gender_with_missing == 'missing'
            indexes = dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition][dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition][gender_field_name] == 'missing'].index
        
            #droping rows based on gender_with_missing column value of 'missing' - effectively this means the percentages for gender are based on 'patients with a gender of Male or Female recorded'
            dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition].drop(indexes,inplace=True)

        # ---------------------------------------

    elif dict_pop_gender_for_each_condition[condition] == 2: #males
        
        #replace the granular gender categories in the df with the high level / less granular gender groups we are limited to, from the user parameters file
        list_genders_present_in_service_data = list(dict_service_gender_groups_to_high_level.keys())

        #create a copy of the dictionary consisting of condition name (keys) and filtered df (values)
        #We are creating a copy here because the subsequent lines of code in this section will replace the (potentially more granular) gender labels with less granular groups/labels
        #We are only using Male and Female gender in this code because we are limited by the available data sources (at the time of writing, we could only source small area population estimates by Male/Female)
        dict_condition_dataset_mapped_gender = {k:v for k, v in dict_condition_dataset.items()}

        #loop through conditions present, then for each condition, loop through the genders present in the df and replace these with the high level groups from the user parameters file
        #at this point, due to the inability to locate small area population estimates by any gender other than Male or Female, this assigns the label 'Currently excluded - population data not available' to
        #any gender label other than those assigned by the user to Male or Female in the user_and_data_params.xlsx set up file.
        #for condition in list_of_conditions:
        for gender in list_genders_present_in_service_data:
            dict_condition_dataset_mapped_gender[condition][gender_field_name] = dict_condition_dataset_mapped_gender[condition][gender_field_name].str.replace(gender, dict_service_gender_groups_to_high_level[gender])
            
            #new line 03/8 - intention is to replace gender labels in the df that contains all rows from min to max age for the condition, even if they fall outside the last complete age-range 'bin'
            dict_condition_dataset_all_ages_in_range[condition][gender_field_name] = dict_condition_dataset_all_ages_in_range[condition][gender_field_name].str.replace(gender, dict_service_gender_groups_to_high_level[gender])

        #for condition in list_of_conditions:
        #dict_condition_dataset_mapped_gender[condition]

        #filtering index where gender_with_missing == 'missing'
        indexes = dict_condition_dataset_mapped_gender[condition][dict_condition_dataset_mapped_gender[condition][gender_field_name] == 'missing'].index
    
        #droping rows based on gender_with_missing column value of 'missing' - effectively this means the percentages for gender are based on 'patients with a gender recorded'
        dict_condition_dataset_mapped_gender[condition].drop(indexes,inplace=True)

        #new line 3/8
        dict_condition_dataset_all_ages_in_range[condition].drop(indexes, inplace=True)
        # ---------------------------------------

        #create a new dictionary called dict_condition_dataset_mapped_gender_subset
        #This is essentially the same as dict_condition_dataset_mapped_gender but this time, the dataframe values in the dictionary have had all rows with associated gender values other than Male removed
        #This reflects the user input / choice that their service works with just Males

        #dict_condition_dataset_mapped_gender_subset = {}
        #dict_condition_dataset_all_ages_in_range_mapped_gender_subset = {}

        #for condition in list_of_conditions:
        #mask = dict_condition_dataset_mapped_gender[condition][f"{gender_field_name}_with_missing"] != 'Currently excluded - population data not available' #create mask for genders other than 'N/a - Remove'. Due to no available population data, only Male and Female genders are retained for the analysis.
        #subset_gender_df = dict_condition_dataset_mapped_gender[condition][mask] #filter using mask
        #dict_condition_dataset_mapped_gender_subset[condition] = subset_gender_df

        #update the service data df with all ages still present, to filter to just males
        mask = dict_condition_dataset_mapped_gender[condition][gender_field_name] == 'Male' #create mask for genders other than 'Males'.
        subset_gender_df = dict_condition_dataset_mapped_gender[condition][mask] #filter using mask
        dict_condition_dataset_mapped_gender_subset[condition] = subset_gender_df

        #update the service data df that has already been filtered to just the ages within the desired age range, to just Males
        mask = dict_condition_dataset_all_ages_in_range[condition][gender_field_name] == 'Male' #create mask for genders other than 'Males'.
        subset_gender_df = dict_condition_dataset_all_ages_in_range[condition][mask] #filter using mask
        dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition] = subset_gender_df

    else: #females

        #replace the granular gender categories in the df with the high level / less granular gender groups we are limited to, from the user parameters file
        list_genders_present_in_service_data = list(dict_service_gender_groups_to_high_level.keys())

        #create a copy of the dictionary consisting of condition name (keys) and filtered df (values)
        #We are creating a copy here because the subsequent lines of code in this section will replace the (potentially more granular) gender labels with less granular groups/labels
        #We are only using Male and Female gender in this code because we are limited by the available data sources (at the time of writing, we could only source small area population estimates by Male/Female)
        dict_condition_dataset_mapped_gender = {k:v for k, v in dict_condition_dataset.items()}

        #loop through conditions present, then for each condition, loop through the genders present in the df and replace these with the high level groups from the user parameters file
        #for condition in list_of_conditions:
        for gender in list_genders_present_in_service_data:
            dict_condition_dataset_mapped_gender[condition][gender_field_name] = dict_condition_dataset_mapped_gender[condition][gender_field_name].str.replace(gender, dict_service_gender_groups_to_high_level[gender])

            #new line 03/8 - intention is to replace gender labels in the df that contains all rows from min to max age for the condition, even if they fall outside the last complete age-range 'bin'
            dict_condition_dataset_all_ages_in_range[condition][gender_field_name] = dict_condition_dataset_all_ages_in_range[condition][gender_field_name].str.replace(gender, dict_service_gender_groups_to_high_level[gender])


        #for condition in list_of_conditions:
        #dict_condition_dataset_mapped_gender[condition]

        #filtering index where gender_with_missing == 'missing'
        indexes = dict_condition_dataset_mapped_gender[condition][dict_condition_dataset_mapped_gender[condition][gender_field_name] == 'missing'].index
    
        #droping rows based on gender_with_missing column value of 'missing' - effectively this means the percentages for gender are based on 'patients with a gender recorded'
        dict_condition_dataset_mapped_gender[condition].drop(indexes,inplace=True)

        #new line 3/8
        dict_condition_dataset_all_ages_in_range[condition].drop(indexes, inplace=True)

        # ---------------------------------------

        #create a new dictionary called dict_condition_dataset_mapped_gender_subset
        #This is essentially the same as dict_condition_dataset_mapped_gender but this time, the dataframe values in the dictionary have had all rows with associated gender values other than Male or Female removed
        #This is a limitation of the available public population estimates, which in turn, is a limitation of this code.
        #dict_condition_dataset_mapped_gender_subset = {}

        #for condition in list_of_conditions:
        #mask = dict_condition_dataset_mapped_gender[condition][f"{gender_field_name}_with_missing"] != 'Currently excluded - population data not available' #create mask for genders other than 'N/a - Remove'. Due to no available population data, only Male and Female genders are retained for the analysis.
        #subset_gender_df = dict_condition_dataset_mapped_gender[condition][mask] #filter using mask
        #dict_condition_dataset_mapped_gender_subset[condition] = subset_gender_df

        #update the service data df with all ages still present, to filter to just males
        mask = dict_condition_dataset_mapped_gender[condition][gender_field_name] == 'Female' #create mask for genders other than 'Females'.
        subset_gender_df = dict_condition_dataset_mapped_gender[condition][mask] #filter using mask
        dict_condition_dataset_mapped_gender_subset[condition] = subset_gender_df

        #update the service data df that has already been filtered to just the ages within the desired age range, to just Males
        mask = dict_condition_dataset_all_ages_in_range[condition][gender_field_name] == 'Female' #create mask for genders other than 'Females'.
        subset_gender_df = dict_condition_dataset_all_ages_in_range[condition][mask] #filter using mask
        dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition] = subset_gender_df

#---------------------------------------------------

# The below if/else logic in next cell is to check if user stated gender for each condition 
# is all persons, males or females and sum each condition_age_range accordingly - 
# should now work for any gender, needs validating

#now we want to create the counts of patients within each age range and assign these to a dictionary using condition name as the constant key
dict_service_counts_condition_age_range = {}

for condition in list_of_conditions:
    
    #persons
    if dict_pop_gender_for_each_condition[condition] == 1: 
        count_by_age_range = dict_condition_dataset[condition].value_counts(subset=['age_group']).sort_index(ascending=True)
        dict_service_counts_condition_age_range[condition] = count_by_age_range
    
    #Males or Females:
    else:
        #original code:
        #count_by_age_range = dict_condition_dataset_mapped_gender_subset[condition].value_counts(subset=['age_group']).sort_index(ascending=True)
        #dict_service_counts_condition_age_range[condition] = count_by_age_range

        #new code 3/8 - this should ensure the age ranges counts are based on the appropriate Male or Female df created earlier in the code, when the user has selected to use Male or Female gender for their analysis
        #count_by_age_range = dict_condition_dataset_mapped_gender_subset[condition].value_counts(subset=['age_group']).sort_index(ascending=True)
        count_by_age_range = dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition].value_counts(subset=['age_group']).sort_index(ascending=True)
        dict_service_counts_condition_age_range[condition] = count_by_age_range

#---------------------------------------------------

#Now import the population data for the age range in scope, for each condition. 
# Then run statistical test for the difference between proportions, using the service data 
# for each condition and age range from dictionary called dict_service_counts_condition_age_range. 
# This dictionary is updated in the code above to include the right source data 
# (persons or relevant gender only) so there is just this one single variable dict to call on 
# for this next stage

#---------------------------------------------------

#cell to create and populate dictionary, with keys of condition names, and values df's containing the sum of people in the pop in each age range

#dict of condition to dataframe containing the sub total sum for population by lsoa for each age range for the given condition
dict_condition_df_pop_age_range_selected_gender = {}
#replace all calls to dict_condition_df_pop_age_range with dict_condition_df_pop_age_range_selected_gender

#dict to contain the condition (key) and col_names as a list (values) present in the df in the above dict
dict_condition_col_names = {}

for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1: #persons
        #create a df for each condition, filtered to just LA of interest, with sum of each pop in each LSOA for each gender
        list_col_names, temp_df = sum_pop_in_each_age_range_revised(num_bins, la_of_interest_user_list_selection, dict_condition_age_ranges, condition, dict_condition_bin_range, dict_condition_start_ages, df_all_persons, df_all_persons_selected_age)
        
        #add subset df as the value to the dict with condition as key
        dict_condition_df_pop_age_range_selected_gender[condition] = temp_df
        dict_condition_col_names[condition] = list_col_names

    elif dict_pop_gender_for_each_condition[condition] == 2: #males
        #create a df for each condition, filtered to just LA of interest, with sum of each pop in each LSOA for MALES ONLY
        list_col_names, temp_df = sum_pop_in_each_age_range_revised(num_bins, la_of_interest_user_list_selection, dict_condition_age_ranges, condition, dict_condition_bin_range, dict_condition_start_ages, df_males, df_males_selected_age)
        
        #add subset df as the value to the dict with condition as key
        dict_condition_df_pop_age_range_selected_gender[condition] = temp_df
        dict_condition_col_names[condition] = list_col_names
    
    else: #females
        #create a df for each condition, filtered to just LA of interest, with sum of each pop in each LSOA for FEMALES ONLY
        list_col_names, temp_df = sum_pop_in_each_age_range_revised(num_bins, la_of_interest_user_list_selection, dict_condition_age_ranges, condition, dict_condition_bin_range, dict_condition_start_ages, df_females, df_females_selected_age)

        #add subset df as the value to the dict with condition as key
        dict_condition_df_pop_age_range_selected_gender[condition] = temp_df
        dict_condition_col_names[condition] = list_col_names

#---------------------------------------------------

#the below code accounts for the prevalence type the user entered as follows:
#it loops through each condition
#in each condition's loop it then looks through the age-rage col names for the given condition
#it then looks in the df associated with the relevant condition which has been processed earlier in the code to only include
#the selected gender and ages within the user-entered range
#it then sums up the values in each row, where each row represents an lsoa and the values represent estimated pop size for that lsoa
#and then multiplies the total pop size for the gender/age combination by the user-entered prevalence rate for that same condition
#the user entered prevalence rate acts as a mulipler, and is 1 where the user selected 'I dont have a prevalence rate', 
# which means no change (essentially, a census), or a value <1 where the user has entered a crude rate with numerator and denominator
#this currently works for 'no rate' and 'crude rate' scenarios. 
#further work / thinking through required for inclusion of age-standardised rates.

dict_condition_age_sub_totals = {}

for condition in list_of_conditions:
    temp_list_pop_age_range_totals = []
    for col_name in dict_condition_col_names[condition]:
        df_sum = int(round(dict_condition_df_pop_age_range_selected_gender[condition][col_name].sum(axis=0)*dict_prev_for_each_condition[condition],0))
        temp_list_pop_age_range_totals.append(df_sum)
    dict_condition_age_sub_totals[condition] = temp_list_pop_age_range_totals

#---------------------------------------------------

#We now have lists consiting of the sum of people in each age range who presented to the 
# service (irrespective whether they attended = expressed need), and the population at 
# large for the LA in which the service operates. Now we can calculate the difference between 
# the proportionate make-up of the service profile relative to the population.

#---------------------------------------------------

#create dictionaries to store the percentages present in each age range for service and population make-up, respectively. 
#These lists will be in the same order as the original counts which is the same order as the age ranges (because lists are ordered, this is ordering is preserved throughout)
dict_condition_service_percents = {}
dict_condition_population_percents_age = {}

#loop through each condition present and convert the counts to percents using the function created earlier
for condition in list_of_conditions:
    #first convert service data to percents and add as a list of float values to the relevant dictionary above
    service_percents = calc_list_percents_for_list_ints(list(dict_service_counts_condition_age_range[condition]))
    dict_condition_service_percents[condition] = service_percents

    #Then, convert population data to percents and add as a list of float values to the relevant dictionary above
    pop_percents = calc_list_percents_for_list_ints(list(dict_condition_age_sub_totals[condition]))
    dict_condition_population_percents_age[condition] = pop_percents

#---------------------------------------------------

#Now calculate the difference between percents in each age range, and assign these 
# differences as a list (value) for each condition (key) in a dictionary

#---------------------------------------------------

#create new dictionary to contain key (condition) value (list of difference between percentages in pop and service for each age-range)
dict_condition_diff_percents = {}

for condition in list_of_conditions:
    diff_between_pop_and_service_list = calc_diff_between_proportions(dict_condition_service_percents[condition], dict_condition_population_percents_age[condition])
    dict_condition_diff_percents[condition] = diff_between_pop_and_service_list

#---------------------------------------------------

dict_condition_95_confidence_interval_age, dict_condition_95_confidence_interval_sig_bool_age = calc_95_ci_diff_proportions(list_of_conditions, num_bins, dict_condition_service_percents,dict_condition_diff_percents, dict_service_counts_condition_age_range)

#---------------------------------------------------

#Create a summary df for each condition, for age, to indicate any proportion in any 
# age group, that is significantly different to the population at large 

#---------------------------------------------------

dict_condition_df_results_age = {}

for condition in list_of_conditions:
    data_list = []
    for num in range(num_bins):
        temp_list = []
        temp_list.append(dict_condition_age_ranges[condition][num])
        temp_list.append(dict_condition_service_percents[condition][num])
        temp_list.append(dict_condition_95_confidence_interval_age[condition][num])
        temp_list.append(dict_condition_population_percents_age[condition][num])
        temp_list.append(dict_condition_95_confidence_interval_sig_bool_age[condition][num])
        data_list.append(temp_list)
    
    df = pd.DataFrame(data_list, columns=['age_range', 'service_proportion','95%_CI', 'population_proportion', 'Significant?'])
    dict_condition_df_results_age[condition] = df

#---------------------------------------------------

# Save output df(s) to file in the assets_produced_by_code / HEA folder

#---------------------------------------------------

for condition in list_of_conditions:
    dict_condition_df_results_age[condition] = dict_condition_df_results_age[condition].reset_index(drop=True)
    file_prefix = number_saved_file(df_counter)
    temp_filename = f"df{file_prefix}_StatTestDiffProportionSigDiffsAge-{condition}.csv"
    dict_condition_df_results_age[condition].to_csv(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}', index=False)
    df_counter+=1
    print("File saved.")

#---------------------------------------------------

#nested dictionary for all file paths to be added to. 
#the structure of the dict will be demographic e.g. age, imd, ethnicity, gender (outer key), to dictionary of file paths (values), which itself is a dictionary with condition (key) and filepaths (values)
#dict_condition_output_file_paths = {}

#create 4 dictionaries, to contain the file names for each chart, for each condition.
#these can later be called when we want to read the processed charts into the 'produce report' code, and insert them as images there.
dict_condition_age_file_paths = {}
dict_condition_ethnicity_file_paths = {}
dict_condition_gender_file_paths = {}
dict_condition_imd_file_paths = {}

#dictionary to store unmet need dataframes
dict_condition_unmet_need_file_paths = {}

#---------------------------------------------------

chart_counter, dict_condition_chart_file_paths = create_bar_plot(chart_counter, 'Age make-up in service vs. population', 'AgeMakeUpInServiceVsPopulation', list_of_conditions, dict_condition_age_ranges, dict_condition_service_percents, dict_condition_population_percents_age, 0, dict_condition_95_confidence_interval_age, '#41B6E6', '#768692', dict_condition_95_confidence_interval_sig_bool_age, dict_condition_age_file_paths)

#---------------------------------------------------
#Chi Square Test or Age
#call the chi square function to run the chi square goodness of fit test for the age demographic
dict_chi_square_test_results_per_condition_age, dict_chi_square_test_statistic_per_condition_age, dict_chi_square_test_pvalue_per_condition_age, dict_chi_square_test_sig_at_alpha_per_condition_age = run_chi_square_test(dict_condition_age_ranges, dict_service_counts_condition_age_range, dict_condition_age_sub_totals, dict_condition_population_percents_age)

#---------------------------------------------------
# GENDER SECTION STARTS
#---------------------------------------------------

#First run if / else control flow, to check whether the user has selected 'persons' for 
# the given condition. If the user has selected Males or Females for a given conditions' 
# gender selection, skip the gender section for that condition as it doesn't make sense to 
# run a statistical test for difference between two proportions for just Males or just 
# Females (as both would be 100%..). 

#---------------------------------------------------

#test print - delete in final
#dict_pop_gender_for_each_condition

#---------------------------------------------------

#Need to check if any processing in the next 2 cells already done when preparing the 
# data for age comparison. delete any duplicate code. add if/else, and run gender code 
# if gender selected was persons. skip if not. 

#---------------------------------------------------
dict_condition_gender_labels = {} #moved outside of the for-if loops below
dict_service_counts_condition_gender = {} #moved outside of the for-if loops below

for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1: #persons

        #read in the genders as they are presented in the user params file reference lists sheet - this is used here as the order of this should remain constant.
        df_subset_genders = pd.read_excel(filename, 'reference_lists', skiprows = 15, nrows=2,  usecols = 'A')

        #convert the above df into a list, so we have an ordererd list of the two gender items. This way we can reliably iterate through Males, then Females, later in the code
        list_subset_genders = list(df_subset_genders.squeeze())

        # ------------------------------

        #test print - delete in final
        list_subset_genders.sort()
        list_subset_genders

        # -----------------------------

        #create a dictionary of condition (key) and list of genders in Female then Male order (values). 
        # This will be used when creating the charts. 
        #dict_condition_gender_labels = {}
        #for condition in list_of_conditions:
        dict_condition_gender_labels[condition] = list_subset_genders

        #now we want to create the counts of Male and Female patients and assign these to a dictionary using condition name as the constant key
        #dict_service_counts_condition_gender = {}

        #for condition in list_of_conditions:
        count_by_gender = dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition].value_counts(subset=[gender_field_name]).sort_index(ascending=True)
        dict_service_counts_condition_gender[condition] = count_by_gender

        #dict_service_counts_condition_gender

#------------------------------

    else: 
        pass

#---------------------------------------------------

#run logic to check whether the selected gender for the condition is persons. if it is, proceed with stat test diff proportions on gender. If it isn't, skip this section for that condition.
#use calc_male_and_female_pop_size function to derive the Male and Female pop for the relevant age range for each condition. Output is a dictionary, condition as key, list of Female pop followed by Male pop as values.

#allow user to indicate approach to modelling unmet need (either all persons, just males, just females, males and females separately)
print("\nThe population estimates can be based upon:")
dict_population_gender = {
    1: "Persons", 
    2: "Males only", 
    3: "Females only",
    #4: "Males and Females (select this if you have differing prevalence rates for each gender)"
    }

run_ci_for_gender = False
#create dictionaries to store the percentages present in each gender group for service and population make-up, respectively. 
#These lists will be in the same order as the original counts which is the same order as the genders in the ref lists sheet in the user params file (because lists are ordered, this is ordering is preserved throughout)
dict_condition_gender_service_percents = {}
dict_condition_gender_population_percents = {}

#create new dictionary to contain key (condition) value (list of difference between percentages in pop and service for each gender)
dict_condition_diff_percents_gender = {}

subset_conditions_for_gender = []
list_conditions_no_gender_comparison_made = []

#dict to check output for dict_condition_gender_list_F_M - this is later used for chi square test
dict_gender_pop_counts = {}

for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1:
        run_ci_for_gender = True
        subset_conditions_for_gender.append(condition)
        dict_condition_gender_list_F_M = calc_male_and_female_pop_size(df_males, la_of_interest_user_list_selection, df_females, dict_min_age_for_each_condition, dict_max_age_for_each_condition, list_of_conditions, dict_prev_for_each_condition)
        
        dict_gender_pop_counts[condition] = dict_condition_gender_list_F_M[condition]
        # ----------------------------------

        #loop through each condition present and convert the counts to percents using the function created earlier
        #for condition in list_of_conditions:
        #first convert service data to percents and add as a list of float values to the relevant dictionary above
        service_gender_percents = calc_list_percents_for_list_ints(list(dict_service_counts_condition_gender[condition]))
        dict_condition_gender_service_percents[condition] = service_gender_percents

        #Then, convert population data to percents and add as a list of float values to the relevant dictionary above
        pop_gender_percents = calc_list_percents_for_list_ints(list(dict_condition_gender_list_F_M[condition]))
        dict_condition_gender_population_percents[condition] = pop_gender_percents

        # -----------------------------------

        #for condition in list_of_conditions:
        diff_between_pop_and_service_list = calc_diff_between_proportions(dict_condition_gender_service_percents[condition], dict_condition_gender_population_percents[condition])
        dict_condition_diff_percents_gender[condition] = diff_between_pop_and_service_list

        # --------------------------------------------

        #dict_condition_df_results_gender

    else:
        print(f'Gender selected for "{condition}" was "{dict_population_gender[dict_pop_gender_for_each_condition[condition]]}" and not "persons", therefore gender comparison inappropriate and not undertaken for the {condition} condition.')
        list_conditions_no_gender_comparison_made.append(condition)

# --------------------------------------------

dict_condition_95_confidence_interval_gender = {} #moved from if-for loop below
dict_condition_95_confidence_interval_sig_bool_gender = {} #moved from if-for loop below
dict_condition_df_results_gender = {} #moved from if-for loop below

if run_ci_for_gender == True:
    dict_temp_gender_file_paths = {}

    for condition in subset_conditions_for_gender:
    #because the calc_95_ci_diff_proportions function created to cycle through the number of bins and use the bin number as the index for each item in a list, this doesn't immediately port to gender categories
    #as, there are only 2 Genders in-scope, where as the original number bins is 5 by default.
    #In order to still make use of the calc_95_ci_diff_proportions function without revision, we still need a num_bins value to enter into the function, though this time, in reality this is the number of gender categories
    #and not number of age ranges. Regardless, this still serves our purposes of providing the correct number to iterate over and use each iteration as the index value for the list(s) in the function call. 
        num_bins_gender = len(list_subset_genders)

        # -------------------------------------------

        #call the function to calc the 95% CI for the diff between proportion in each gender group in the service compared to the population at large
        #original code
        #dict_condition_95_confidence_interval_gender, dict_condition_95_confidence_interval_sig_bool_gender = calc_95_ci_diff_proportions(list_of_conditions, num_bins_gender, dict_condition_gender_service_percents, dict_condition_diff_percents_gender, dict_service_counts_condition_gender)
        
        #dict_condition_95_confidence_interval_gender = {}
        #dict_condition_95_confidence_interval_sig_bool_gender = {} 
        
        list_confidence_intervals, list_confidence_intervals_significant_or_not = calc_95_ci_diff_proportions_gender(condition, num_bins_gender, dict_condition_gender_service_percents, dict_condition_diff_percents_gender, dict_service_counts_condition_gender)
        
        dict_condition_95_confidence_interval_gender[condition] = list_confidence_intervals
        dict_condition_95_confidence_interval_sig_bool_gender[condition] = list_confidence_intervals_significant_or_not
        # ------------------------------------------
        
        #dict_condition_df_results_gender = {}

        #for condition in list_of_conditions:
        if dict_pop_gender_for_each_condition[condition] == 1:
            
            data_list = []
            for num in range(num_bins_gender):
                temp_list = []
                temp_list.append(sorted(list_subset_genders)[num])
                temp_list.append(dict_condition_gender_service_percents[condition][num])
                temp_list.append(dict_condition_95_confidence_interval_gender[condition][num])
                temp_list.append(dict_condition_gender_population_percents[condition][num])
                temp_list.append(dict_condition_95_confidence_interval_sig_bool_gender[condition][num])
                data_list.append(temp_list)
            
            df = pd.DataFrame(data_list, columns=['gender', 'service_proportion', '95%_CI', 'population_proportion', 'Significant?'])
            dict_condition_df_results_gender[condition] = df


            #save as assets the df for each condition, showing the 95% CI, and whether or not this was significantly different to the relevant local authority's population at large
            #for condition in list_of_conditions:
            dict_condition_df_results_gender[condition] = dict_condition_df_results_gender[condition].reset_index(drop=True)
            file_prefix = number_saved_file(df_counter)
            temp_filename = f"df{file_prefix}_StatTestDiffProportionSigDiffsGender-{condition}.csv"
            dict_condition_df_results_gender[condition].to_csv(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}', index=False)
            df_counter+=1
            dict_temp_gender_file_paths[condition] = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}'
            print("File saved.")
    dict_condition_gender_file_paths['dfs'] = dict_temp_gender_file_paths
    
    chart_counter, dict_condition_gender_file_paths = create_bar_plot(chart_counter, 'Gender make-up in service vs. population', 'GenderMakeUpInServiceVsPopulation', subset_conditions_for_gender, dict_condition_gender_labels, dict_condition_gender_service_percents, dict_condition_gender_population_percents, 0, dict_condition_95_confidence_interval_gender, '#41B6E6', '#768692', dict_condition_95_confidence_interval_sig_bool_gender, dict_condition_gender_file_paths)


# ---------------------------------------------
#Chi Square test for Gender
#call the chi square function to run the chi square goodness of fit test for the gender demographic
if run_ci_for_gender == True:
    dict_chi_square_test_results_per_condition_gender, dict_chi_square_test_statistic_per_condition_gender, dict_chi_square_test_pvalue_per_condition_gender, dict_chi_square_test_sig_at_alpha_per_condition_gender = run_chi_square_test(dict_condition_gender_labels, dict_service_counts_condition_gender, dict_gender_pop_counts, dict_condition_gender_population_percents)


# ---------------------------------------------
# Gender section ends

# ---------------------------------------------
# <<< Ethnicity section starts >>>
# ---------------------------------------------

#Now map the ethnicities present in the dataset to the high level ethnicities from nomis web. 
# The aim of this section is to finalise additional processing on the dataframe(s) for the 
# condition(S) being modelled, to then be able to run the analyses on the summary stats from 
# the processed service data.


#read in the ethnicity sheet from user params as a new df
df_ethnic_groups_to_high_level = pd.read_excel(filename, 'Ethnicity_mapping', skiprows=1)

#create a dict to populate with service data (granular) ethnicity categories (key) to high level categories (values)
dict_service_ethnic_groups_to_high_level = {}

#create a tuple from each row in the df, and populate the above dict
for row in df_ethnic_groups_to_high_level.itertuples():
    dict_service_ethnic_groups_to_high_level[row.Service_data_categories] = row.High_level_categories

#----------

#identify the string used as the field name for the ethnicity field from the user parameters file
ethnicity_field_name = 'ethnicity'
print(f"The field name that will be used for the ethnicity field is: {ethnicity_field_name}")

#---------

for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1: #persons
        
        #replace the granular ethnicity categories in the df with the high level / less granular ethnic groups, from the user parameters file
        list_ethnicities_present_in_service_data = list(dict_service_ethnic_groups_to_high_level.keys())

        #create a copy of the dictionary consisting of condition name (keys) and filtered df (values)
        #We are creating a copy here because the subsequent lines of code in this section will replace the (potentially more granular) ethnicity labels with less granular groups/labels
        #These are taken from: https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups#:~:text=The%20ethnic%20groups%20will%20be%3A%20Asian%2C%20Scottish%20Asian,British%20Pakistani%20Indian%2C%20Scottish%20Indian%20or%20British%20Indian

        dict_condition_dataset_mapped_ethnicity = {k:v for k, v in dict_condition_dataset.items()}

        #dict_condition_dataset_mapped_ethnicity = {}

        #for each loop through of conditions present, loop through the ethnicities present in the df and replace these with the high level groups from the user parameters file
        for ethnicity in list_ethnicities_present_in_service_data:
            dict_condition_dataset_mapped_ethnicity[condition][ethnicity_field_name] = dict_condition_dataset_mapped_ethnicity[condition][ethnicity_field_name].str.replace(ethnicity, dict_service_ethnic_groups_to_high_level[ethnicity])

        dict_condition_dataset_mapped_ethnicity[condition]

        #filtering index where ethnicity_with_missing == 'missing'
        indexes = dict_condition_dataset_mapped_ethnicity[condition][dict_condition_dataset_mapped_ethnicity[condition][ethnicity_field_name] == 'missing'].index
        
        #dropping rows based on ethnicity_with_missing column value of 'missing' - effectively this means the percentages for ethnicity are based on 'patients with an ethnicity recorded'
        dict_condition_dataset_mapped_ethnicity[condition].drop(indexes,inplace=True)
    
    else: #males or females
        
        #replace the granular ethnicity categories in the df with the high level / less granular ethnic groups, from the user parameters file
        list_ethnicities_present_in_service_data = list(dict_service_ethnic_groups_to_high_level.keys())

    #create a copy of the dictionary consisting of condition name (keys) and filtered df (values)
    #We are creating a copy here because the subsequent lines of code in this section will replace the (potentially more granular) ethnicity labels with less granular groups/labels
    #These are taken from: https://www.ethnicity-facts-figures.service.gov.uk/style-guide/ethnic-groups#:~:text=The%20ethnic%20groups%20will%20be%3A%20Asian%2C%20Scottish%20Asian,British%20Pakistani%20Indian%2C%20Scottish%20Indian%20or%20British%20Indian

        dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset = {k:v for k, v in dict_condition_dataset_all_ages_in_range_mapped_gender_subset.items()}

    #dict_condition_dataset_mapped_ethnicity = {}

    #loop through conditions present, then for each condition, loop through the ethnicities present in the df and replace these with the high level groups from the user parameters file
    #for condition in list_of_conditions:
        for ethnicity in list_ethnicities_present_in_service_data:
            dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition][ethnicity_field_name] = dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition][ethnicity_field_name].str.replace(ethnicity, dict_service_ethnic_groups_to_high_level[ethnicity])

        #dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition]

        #filtering index where ethnicity_with_missing == 'missing'
        indexes = dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition][dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition][ethnicity_field_name] == 'missing'].index
        
        #droping rows based on ethnicity_with_missing column value of 'missing' - effectively this means the percentages for ethnicity are based on 'patients with an ethnicity recorded'
        dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition].drop(indexes,inplace=True)

        ## CHECK WHETHER OUTPUT AT THIS POINT WHEN MALES OR FEMALES SELECTED INCLUDES JUST MALES OR FEMALES, IF NOT, COPY LAST 2 LINES OF CODE IN THIS CELL TO REDUCE TO JUST CHOSEN GENDER

# ---------------------------------------------

#Logic being used: the df's used for ethnicity and later, imd, inc all rows with ages 
# within the age (bin) ranges the code derived. This means there is a small margin of error 
# currently, where the ACTUAL service age range may be e.g. 16-65, BUT, 5 equal bins with a 
# bin-range of 9 years, would span from 16 to 60, and remove / exclude any patient with an 
# age of 61+. For ethnicity this shouldnt be an issue as the pop estimates from nomis are at 
# UTLA level only. For IMD, which is available to LSOA and Single Year of Age level, only the 
# ages in the SAME age range are included (so 16-60 in the above example) meaning we are still
# comparing apples and apples.

# ---------------------------------------------

#populate a new dictionary, with key of condition name as usual, and values as the relevant df from the relevant dictionary according to gender selected for the analysis
#this way we have a single dictionary variable to call on for next steps.

dict_condition_ethnicity_df_to_use = {} #also used in imd section
for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1: #persons
        dict_condition_ethnicity_df_to_use[condition] = dict_condition_dataset_mapped_ethnicity[condition] #any ages within the age range 'bins'
    else:
        dict_condition_ethnicity_df_to_use[condition] = dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset[condition]

# ---------------------------------------------

#now we want to create the counts of patients with each ethnicity and assign these to a dictionary using condition name as the constant key
dict_service_counts_condition_ethnicity = {}

for condition in list_of_conditions:
    count_by_ethnic_group = dict_condition_ethnicity_df_to_use[condition].value_counts(subset=[ethnicity_field_name]).sort_index(ascending=True)
    dict_service_counts_condition_ethnicity[condition] = count_by_ethnic_group

# ----------------------------------------------

#read in the data downloaded from nomis.web containing the estimated population count per high level ethnic group, for each unitary or upper tier local authority
nomis_ethnicity_pop_per_utla = pd.read_csv('raw_data/nomis_web/nomis_ethnicity_unitary_county.csv')

#convert the utla column to lower case text so as to be in the same format as all other matching columbs
nomis_ethnicity_pop_per_utla['local authority: county / unitary (prior to April 2015)'] = nomis_ethnicity_pop_per_utla['local authority: county / unitary (prior to April 2015)']

#create a mask, which takes the form of a boolean series, showing False for every UTLA in the nomis_ethnicity_pop_per_utla dataframe that DOES NOT match the LA of interest, and TRUE for the ONE that does match 
mask = nomis_ethnicity_pop_per_utla['local authority: county / unitary (prior to April 2015)'] == la_of_interest_user_list_selection

#use this mask to filter the nomis_ethnicity_pop_per_utla dataframe to now only contain the ONE row for the relevant LA of interest
df_nomis_ethnicity_pop_per_utla_filtered = nomis_ethnicity_pop_per_utla[mask]

# ------------------------------------------------

#because the column names (ethnicitiy labels) do not match the mapping used from the .gov.uk site, they need to be renamed. 
#To do this, first we need a dictionary that maps the original ethnicity description (key) to the one required (value)
dict_mapping_nomis_ethnicities_to_gov_uk_ethnicities = {
    'White' : 'White',
    'Mixed' : 'Mixed or multiple ethnic groups',
    'Asian' : 'Asian or Asian British',
    'Black' : 'Black, Black British, Caribbean or African',
    'Other' : 'Other ethnic group'
}

#use the dictionary to rename the columns of the df_nomis_ethnicity_pop_per_utla_filtered in place
df_nomis_ethnicity_pop_per_utla_filtered.rename(columns=dict_mapping_nomis_ethnicities_to_gov_uk_ethnicities, inplace=True)

# -------------------------------------------------

#create a list of the nomis web column names, using row and col slicing to ignore the first 2 columns of the df (LA name and total pop in all ethnic groups) as these are not required
ethnicity_col_names = list(df_nomis_ethnicity_pop_per_utla_filtered.iloc[:,2:].columns)

#create a dict for ethnicity counts - Populate a dictionary with keys (condition) and values (sub-totals per age range)
dict_condition_ethnicity_sub_totals = {}

#loop through to populate the above dictionary
for condition in list_of_conditions:
    #temp_list_pop_ethnicity_range_totals = []
    #for col_name in ethnicity_col_names:
    #    temp_list_pop_ethnicity_range_totals.append(int(df_nomis_ethnicity_pop_per_utla_filtered[col_name]))

    #This next line takes a slice from the df for just the columns containing the pop counts for each ethnicity, 
    # resets the index of the slice, then, using the squeeze method, converts this df to a pandas series, 
    # then, sorts the series by the index (ethnicity descriptions) in ascending order 
    # to ensure it matches the exact same order as the lists of service counts for each condition in the relevant dictionary, already created
    convert_df_ethnicity_to_series = df_nomis_ethnicity_pop_per_utla_filtered.loc[:, ethnicity_col_names[0]:ethnicity_col_names[-1]].reset_index(drop=True).squeeze().sort_index(ascending = True)

    #add the sorted series to the dictionary, as a value to the condition key
    dict_condition_ethnicity_sub_totals[condition] = convert_df_ethnicity_to_series

print(f"dict_condition_ethnicity_sub_totals: {dict_condition_ethnicity_sub_totals}")

# -----------------------------------------------------

#test print
ethnicity_col_names.sort()

#create a dict with the condition as key and ethnicity labels as values. All values will be identical for all keys here. 
# This is to keep a consistent approach and enable the function to produce the charts to run for ethnicity later on.
dict_condition_ethnicity_labels = {}
for condition in list_of_conditions:
    dict_condition_ethnicity_labels[condition] = ethnicity_col_names

# --------------------------------------------------------

#create dictionaries to store the percentages present in each ethnic group for service and population make-up, respectively. 
#These lists will be in the same order as the original counts which is the same order as the age ranges (because lists are ordered, this is ordering is preserved throughout)
dict_condition_ethnicity_service_percents = {}
dict_condition_ethnicity_population_percents = {}

#loop through each condition present and convert the counts to percents using the function created earlier
for condition in list_of_conditions:
    #first convert service data to percents and add as a list of float values to the relevant dictionary above
    service_ethnicity_percents = calc_list_percents_for_list_ints(list(dict_service_counts_condition_ethnicity[condition]))
    dict_condition_ethnicity_service_percents[condition] = service_ethnicity_percents

    #Then, convert population data to percents and add as a list of float values to the relevant dictionary above
    pop_ethnicity_percents = calc_list_percents_for_list_ints(list(dict_condition_ethnicity_sub_totals[condition]))
    dict_condition_ethnicity_population_percents[condition] = pop_ethnicity_percents

# --------------------------------------------------------

#create new dictionary to contain key (condition) value (list of difference between percentages in pop and service for each ethnicity)
dict_condition_diff_percents_ethnicity = {}

for condition in list_of_conditions:
    diff_between_pop_and_service_list = calc_diff_between_proportions(dict_condition_ethnicity_service_percents[condition], dict_condition_ethnicity_population_percents[condition])
    dict_condition_diff_percents_ethnicity[condition] = diff_between_pop_and_service_list

#test print - delete in final
dict_condition_diff_percents_ethnicity

# -----------------------------------------------------
 
 #because the calc_95_ci_diff_proportions function created to cycle through the number of bins and use the bin number as the index for each item in a list, this doesn't immediately port to ethnicity categories
#as, by co-incidence, there happen to be 5 categories (same as default num_bins for age ranges), but could easily be different, in which case the code would fail / crash.
#In order to still make use of the calc_95_ci_diff_proportions function without revision, we still need a num_bins value to enter into the function, though this time, in reality this is the number of ethnicity categories
#and not number of age ranges. Regardless, this still serves our purposes of providing the correct number to iterate over and use each iteration as the index value for the list(s) in the function call. 
num_bins_ethnicity = len(ethnicity_col_names)
 
# -----------------------------------------------------
 #call the function to calc the 95% CI for the diff between proportion in each ethnicity group in the service compared to the population at large
dict_condition_95_confidence_interval_ethnicity, dict_condition_95_confidence_interval_sig_bool_ethnicity = calc_95_ci_diff_proportions(list_of_conditions, num_bins_ethnicity, dict_condition_ethnicity_service_percents, dict_condition_diff_percents_ethnicity, dict_service_counts_condition_ethnicity)
# -----------------------------------------------------

#create a dictionary consiting of the df summarising the results of the statistical test between proportions, for each condition
dict_condition_df_results_ethnicity = {}
dict_temp_ethnicity_file_paths = {}

for condition in list_of_conditions:
    data_list = []
    for num in range(num_bins_ethnicity):
        temp_list = []
        temp_list.append(ethnicity_col_names[num])
        temp_list.append(service_ethnicity_percents[num])
        temp_list.append(dict_condition_95_confidence_interval_ethnicity[condition][num])
        temp_list.append(dict_condition_ethnicity_population_percents[condition][num])
        temp_list.append(dict_condition_95_confidence_interval_sig_bool_ethnicity[condition][num])
        data_list.append(temp_list)
    
    df = pd.DataFrame(data_list, columns=['ethnicity', 'service_proportion', '95%_CI', 'population_proportion', 'Significant?'])
    dict_condition_df_results_ethnicity[condition] = df
 
 # -----------------------------------------------------

#save as assets the df for each condition, showing the 95% CI, and whether or not this was significantly different to the relevant local authority's population at large
for condition in list_of_conditions:
    dict_condition_df_results_ethnicity[condition] = dict_condition_df_results_ethnicity[condition].reset_index(drop=True)
    file_prefix = number_saved_file(df_counter)
    temp_filename = f"df{file_prefix}_StatTestDiffProportionSigDiffsEthnicity-{condition}.csv"
    dict_condition_df_results_ethnicity[condition].to_csv(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}', index=False)
    dict_temp_ethnicity_file_paths[condition] = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}' # add condition (key) and filepath (value) to relevant dictionary for calling into report later
    df_counter+=1
    print("File saved.")

# --------------------------------------------------------

dict_condition_ethnicity_file_paths['dfs'] = dict_temp_ethnicity_file_paths
# --------------------------------------------------------

chart_counter, dict_condition_ethnicity_file_paths = create_bar_plot(chart_counter, 'Ethnicity make-up in service vs. population', 'EthnicityMakeUpInServiceVsPopulation', list_of_conditions, dict_condition_ethnicity_labels, dict_condition_ethnicity_service_percents, dict_condition_ethnicity_population_percents, 90, dict_condition_95_confidence_interval_ethnicity, '#41B6E6', '#768692', dict_condition_95_confidence_interval_sig_bool_ethnicity, dict_condition_ethnicity_file_paths)

# --------------------------------------------------------

#dict_condition_dataset_all_ages_in_range_mapped_gender_ethnicity_subset['GUM'].shape #males aged 16-65, this should be 55 (which it is)
#dict_condition_dataset_mapped_ethnicity['GUM'].shape #persons aged 16-65, requires UAT manual check in file

# --------------------------------------------------------
#Chi square test for ethnicity
#call the chi square function to run the chi square goodness of fit test for the ethnicity demographic
dict_chi_square_test_results_per_condition_ethnicity, dict_chi_square_test_statistic_per_condition_ethnicity, dict_chi_square_test_pvalue_per_condition_ethnicity, dict_chi_square_test_sig_at_alpha_per_condition_ethnicity = run_chi_square_test(dict_condition_ethnicity_labels, dict_service_counts_condition_ethnicity, dict_condition_ethnicity_sub_totals, dict_condition_ethnicity_population_percents)

# --------------------------------------------------------
# Ethnicity section ends

# --------------------------------------------------------
# <<< IMD SECTION STARTS >>>
# --------------------------------------------------------

imd_field_name = 'imd_decile'
print(f"The field name that will be used for the IMD decile field is: {imd_field_name}")

# --------------------------------------------------------

#populate a new dictionary, with key of condition name as usual, and values as the relevant df from the relevant dictionary according to gender selected for the analysis
#this way we have a single dictionary variable to call on for next steps.
dict_condition_imd_df_to_use = {}

for condition in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition] == 1: #persons
        dict_condition_imd_df_to_use[condition] = dict_condition_dataset[condition]

    else: #males or females
        dict_condition_imd_df_to_use[condition] = dict_condition_dataset_all_ages_in_range_mapped_gender_subset[condition]

# --------------------------------------------------------

#read in the IMD sheet from user params as a new df
df_imd_decile = pd.read_excel(filename, 'reference_lists', skiprows=20, nrows=10)
df_imd_decile = df_imd_decile.iloc[:,:1]

#convert this to a list of deciles (ints) from 1 (most deprived) to 10 (least deprived)
list_imd_decile = list(df_imd_decile.squeeze())

# --------------------------------------------------------

#now we want to create the counts of each IMD decile and assign these to a dictionary using condition name as the constant key
dict_service_counts_condition_imd = {}

for condition in list_of_conditions:
    #original code
    #count_by_imd = dict_condition_dataset[condition].value_counts(subset=[f"{imd_field_name}"]).sort_index(ascending=True)
    #dict_service_counts_condition_imd[condition] = count_by_imd
    list_count_per_decile = []

    for num in list_imd_decile:
        temp_bool_series_for_each_decile = dict_condition_imd_df_to_use[condition]['imd_decile'] == num
        count_per_decile = temp_bool_series_for_each_decile.sum()
        list_count_per_decile.append(count_per_decile)

    dict_service_counts_condition_imd[condition] = list_count_per_decile

#test print - delete in final
print("Service data counts per IMD decile:")
for condition in list_of_conditions:
    print(f"{condition}: {dict_service_counts_condition_imd[condition]}")
#print(dict_service_counts_condition_imd)

# --------------------------------------------------------

# --------------------------------------------------------

#Take a copy of the dictionary created earlier in the code (could probably just apply these changes to the original, taking a copy during development in case errors introduced!)
dict_condition_df_pop_age_range_with_imd = copy.deepcopy(dict_condition_df_pop_age_range_selected_gender)

#create a new dict which we will then populate with condition name (keys) and subset df containing lsoas in the LA of interest, imd deciles, and total pop between the min and max ages in the age ranges used (values)
dict_condition_lsoa_total_pop_imd = {}

dict_condition_to_pop_col_name = {}


#create dicts to house the min and max ages based on youngest and oldest age in the bin ranges for each condition
dict_condition_start_ages_from_bin_ranges = {}
dict_condition_end_ages_from_bin_ranges = {}

for condition in list_of_conditions:
    dict_condition_df_pop_age_range_with_imd[condition] = pd.merge(dict_condition_df_pop_age_range_with_imd[condition], df_lsoa_imd_only, left_on='LSOA Code', right_on='lsoa11cd')
    dict_condition_df_pop_age_range_with_imd

    #code to identify the first age and last age in age ranges for each condition and create a new column name made up of these first/last ages
    start_age = str(dict_condition_start_ages[condition][0])
    end_age = str(dict_condition_end_ages[condition][-2])
    col_age_total_name = f"{condition}_PopAged_{start_age}-{end_age}"
    
    dict_condition_to_pop_col_name[condition] = col_age_total_name

    dict_condition_start_ages_from_bin_ranges[condition] = int(start_age)
    dict_condition_end_ages_from_bin_ranges[condition] = int(end_age)

    #create a new df, and populate this with the lsoa code, the sum of the population in each age range in each lsoa, and include the imd decile for each lsoa
    df_subset_condition_lsoa_imd_decile = pd.DataFrame()
    
    #add lsoa code column to the new df
    df_subset_condition_lsoa_imd_decile['LSOA Code'] = dict_condition_df_pop_age_range_with_imd[condition]['LSOA Code']

    #sum the pop in each of the age range columns present, add this to the new df with a contextually relevant name for the given condition, between given min / max ages
    #df_subset_condition_lsoa_imd_decile[col_age_total_name] = dict_condition_df_pop_age_range_with_imd[condition].iloc[:,7:7+len(dict_condition_age_ranges['GUM'])].sum(axis=1) #original
    df_subset_condition_lsoa_imd_decile[col_age_total_name] = dict_condition_df_pop_age_range_with_imd[condition].iloc[:,7:7+len(dict_condition_age_ranges[condition])].sum(axis=1) #new 1/11

    #add imd decile to the new df
    df_subset_condition_lsoa_imd_decile['imd_decile'] = dict_condition_df_pop_age_range_with_imd[condition]['imd_decile']
    
    #assign to dictionary with condition as key
    dict_condition_lsoa_total_pop_imd[condition] = df_subset_condition_lsoa_imd_decile

# --------------------------------------------------------

#now we want to create the counts of patients within each imd decile and assign these to a dictionary using condition name as the constant key
dict_pop_counts_condition_imd_decile = {}

for condition in list_of_conditions:
    list_imd_decile_pop_count = []
    for num in range(10):
        decile = num+1
        decile_pop_sum = dict_condition_lsoa_total_pop_imd[condition].loc[dict_condition_lsoa_total_pop_imd[condition]['imd_decile'] == decile, dict_condition_to_pop_col_name[condition]].sum()
        list_imd_decile_pop_count.append(decile_pop_sum)
    #count_by_imd_decile = dict_condition_lsoa_total_pop_imd[condition].value_counts(subset='imd_decile').sort_index(ascending=True)
    dict_pop_counts_condition_imd_decile[condition] = list_imd_decile_pop_count

# --------------------------------------------------------

#create dictionaries to store the percentages present in each imd decile for service and population make-up, respectively. 
dict_condition_imd_service_percents = {}
dict_condition_imd_population_percents = {}

#loop through each condition present and convert the counts to percents using the function created earlier
for condition in list_of_conditions:
    #first convert service data to percents and add as a list of float values to the relevant dictionary above
    service_imd_percents = calc_list_percents_for_list_ints(dict_service_counts_condition_imd[condition])
    dict_condition_imd_service_percents[condition] = service_imd_percents

    #Then, convert population data to percents and add as a list of float values to the relevant dictionary above
    pop_imd_percents = calc_list_percents_for_list_ints(dict_pop_counts_condition_imd_decile[condition])
    dict_condition_imd_population_percents[condition] = pop_imd_percents

# --------------------------------------------------------

#create new dictionary to contain key (condition) value (list of difference between percentages in pop and service for each imd decile)
dict_condition_diff_percents_imd_decile = {}

for condition in list_of_conditions:
    diff_between_pop_and_service_list = calc_diff_between_proportions(dict_condition_imd_service_percents[condition], dict_condition_imd_population_percents[condition])
    dict_condition_diff_percents_imd_decile[condition] = diff_between_pop_and_service_list

# --------------------------------------------------------

#convert the service counts per imd decile to a series type instead of a list
#this is because the function created to calculate the confidence intervals requires a series and not a list

dict_service_counts_condition_imd_as_series = {}
for condition in list_of_conditions:
    dict_service_counts_condition_imd_as_series[condition] = pd.Series(dict_service_counts_condition_imd[condition])

# --------------------------------------------------------

#call the function to calc the 95% CI for the diff between proportion in each imd decile in the service compared to the population at large
dict_condition_95_confidence_interval_imd, dict_condition_95_confidence_interval_sig_bool_imd = calc_95_ci_diff_proportions(list_of_conditions, 10, dict_condition_imd_service_percents, dict_condition_diff_percents_imd_decile, dict_service_counts_condition_imd_as_series)

# -----------------------------------------------
#create a dictionary consiting of the df summarising the results of the statistical test between proportions, for each condition
dict_condition_df_results_imd_decile = {}

for condition in list_of_conditions:
    data_list = []
    for num in range(10):
        temp_list = []
        temp_list.append(num+1)
        temp_list.append(dict_condition_imd_service_percents[condition][num])
        temp_list.append(dict_condition_95_confidence_interval_imd[condition][num])
        temp_list.append(dict_condition_imd_population_percents[condition][num])
        temp_list.append(dict_condition_95_confidence_interval_sig_bool_imd[condition][num])
        data_list.append(temp_list)
    
    df = pd.DataFrame(data_list, columns=['IMD_decile', 'service_proportion', '95%_CI', 'population_proportion', 'Significant?'])
    dict_condition_df_results_imd_decile[condition] = df

# --------------------------------------------------------

#save as assets the df for each condition, showing the 95% CI, and whether or not this was significantly different to the relevant local authority's population at large
dict_temp_imd_file_paths = {}

for condition in list_of_conditions:
    dict_condition_df_results_imd_decile[condition] = dict_condition_df_results_imd_decile[condition].reset_index(drop=True)
    file_prefix = number_saved_file(df_counter)
    temp_filename = f"df{file_prefix}_StatTestDiffProportionSigDiffsIMD-{condition}.csv"
    dict_condition_df_results_imd_decile[condition].to_csv(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}', index=False) #dict containing actual dataframe
    dict_temp_imd_file_paths[condition] = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}' #dict containing file path to saved csv
    df_counter+=1
    print("File saved.")

dict_condition_imd_file_paths['dfs'] = dict_temp_imd_file_paths

# --------------------------------------------------------

#create a dictionary consiting of condition name (keys) and a list of ints 1 to 10 inclusive as values, representing the deciles 1 to 10
#this is used in the call to the function to produce the charts

dict_condition_imd_labels = {}

for condition in list_of_conditions:
    dict_condition_imd_labels[condition] = [num for num in range(1,11)]

# --------------------------------------------------------

chart_counter, dict_condition_imd_file_paths = create_bar_plot(chart_counter, 'IMD decile service vs. pop. proportion (1=most deprived)', 'IMDDecileMakeUpInServiceVsPopulation', list_of_conditions, dict_condition_imd_labels, dict_condition_imd_service_percents, dict_condition_imd_population_percents, 0, dict_condition_95_confidence_interval_imd, '#41B6E6', '#768692', dict_condition_95_confidence_interval_sig_bool_imd, dict_condition_imd_file_paths)

# --------------------------------------------------------
#Chi Square Test IMD
#call the chi square function to run the chi square goodness of fit test for the ethnicity demographic
dict_chi_square_test_results_per_condition_imd, dict_chi_square_test_statistic_per_condition_imd, dict_chi_square_test_pvalue_per_condition_imd, dict_chi_square_test_sig_at_alpha_per_condition_imd = run_chi_square_test(dict_condition_imd_labels, dict_service_counts_condition_imd_as_series, dict_pop_counts_condition_imd_decile, dict_condition_imd_population_percents)

# --------------------------------------------------------
#<< SUMMARISE CHI SQUARE RESULTS FOR EACH DEMOGRAPHIC INTO A RESULTS DF
# --------------------------------------------------------
#build up a list of dataframes, with each dataframe in the list a summary of the chi square results for a given demographic
master_list_of_data_frames = []

#run for age
master_list_of_data_frames = summarise_chi_square_results_into_df(
    list_of_conditions,
    dict_chi_square_test_results_per_condition_age,
    dict_chi_square_test_statistic_per_condition_age,
    dict_chi_square_test_pvalue_per_condition_age,
    dict_chi_square_test_sig_at_alpha_per_condition_age,
    master_list_of_data_frames,
    'age')

if len(list(dict_condition_gender_labels.keys())) != 0:
    #run for gender
    master_list_of_data_frames = summarise_chi_square_results_into_df(
        list_of_conditions,
        dict_chi_square_test_results_per_condition_gender,
        dict_chi_square_test_statistic_per_condition_gender,
        dict_chi_square_test_pvalue_per_condition_gender,
        dict_chi_square_test_sig_at_alpha_per_condition_gender,
        master_list_of_data_frames,
        'gender')

#run for ethnicity
master_list_of_data_frames = summarise_chi_square_results_into_df(
    list_of_conditions,
    dict_chi_square_test_results_per_condition_ethnicity,
    dict_chi_square_test_statistic_per_condition_ethnicity,
    dict_chi_square_test_pvalue_per_condition_ethnicity,
    dict_chi_square_test_sig_at_alpha_per_condition_ethnicity,
    master_list_of_data_frames,
    'ethnicity')

#run for imd
master_list_of_data_frames = summarise_chi_square_results_into_df(
    list_of_conditions,
    dict_chi_square_test_results_per_condition_imd,
    dict_chi_square_test_statistic_per_condition_imd,
    dict_chi_square_test_pvalue_per_condition_imd,
    dict_chi_square_test_sig_at_alpha_per_condition_imd,
    master_list_of_data_frames,
    'imd')

df_chi_square_results = pd.concat(master_list_of_data_frames)
df_chi_square_results = df_chi_square_results.reset_index().drop('index', axis=1)
#test print
#print(df_chi_square_results)

#save df_chi_square_results as a csv within the HEA assets folder
df_chi_square_results.to_csv(f"{hea_assets_path}/df_chi_square_results.csv", index=False)

# --------------------------------------------------------
#End of statistical test between proportion (service vs. populaton) section

# --------------------------------------------------------
# <<< Quantifying unmet need starts >>>
# --------------------------------------------------------

dict_condition_estimated_need = {}

for condition_name in list_of_conditions:
    if dict_pop_gender_for_each_condition[condition_name] == 1: #persons
        #df_all_persons_selected_age = subset_pop_estimate(df_all_persons, df_all_persons_selected_age, all_persons_fields, min_age, max_age)
        print("branch1-1") # test print to check control flow logic - delete when resolved
        df_all_persons_selected_age = subset_pop_estimate(df_all_persons)
        
        #add in the UTLA Name column
        df_all_persons_selected_age.insert(loc=6, column='UTLA19NM', value=df_all_persons['UTLA19NM'], allow_duplicates=True)

        #for condition_name in list(dict_prev_for_each_condition.keys()):
        df_all_persons_selected_age = sum_age_range(df_all_persons, df_all_persons_selected_age, str(dict_condition_start_ages_from_bin_ranges[condition_name]), str(dict_condition_end_ages_from_bin_ranges[condition_name]), condition_name)
        df_all_persons_selected_age[f"{condition_name}_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}_Estimated_Need"] = df_all_persons_selected_age[f"Total_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}_{condition_name}"] * dict_prev_for_each_condition[condition_name]
        
        #subset the df to only contain rows for LSOAs within the UTLA of interest
        mask = df_all_persons_selected_age['UTLA19NM'] == la_of_interest_user_list_selection
        subset_lsoa_df = df_all_persons_selected_age[mask]

        lower_tier_local_authorities_all = list(set(df_all_persons_selected_age["LA name (2021 boundaries)"]))
        lsoa_name_all = list(df_all_persons_selected_age["LSOA Name"])
        
        dict_condition_estimated_need[condition_name] = subset_lsoa_df
    
    elif dict_pop_gender_for_each_condition[condition_name] == 2: #Males
        print("branch2-1") # test print to check control flow logic - delete when resolved
        df_males_selected_age = subset_pop_estimate(df_males)

        #add in the UTLA Name column
        df_males_selected_age.insert(loc=6, column='UTLA19NM', value=df_males['UTLA19NM'], allow_duplicates=True)
            
        df_males_selected_age = sum_age_range(df_males, df_males_selected_age, str(dict_condition_start_ages_from_bin_ranges[condition_name]), str(dict_condition_end_ages_from_bin_ranges[condition_name]), condition_name)
        df_males_selected_age[f"{condition_name}_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}_Estimated_Need"] = df_males_selected_age[f"Total_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}_{condition_name}"] * dict_prev_for_each_condition[condition_name]
        
        #subset the df to only contain rows for LSOAs within the UTLA of interest
        mask = df_males_selected_age['UTLA19NM'] == la_of_interest_user_list_selection
        subset_lsoa_df = df_males_selected_age[mask]
        
        lower_tier_local_authorities_all = list(set(df_males_selected_age["LA name (2021 boundaries)"]))
        lsoa_name_all = list(df_males_selected_age["LSOA Name"])

        dict_condition_estimated_need[condition_name] = subset_lsoa_df
        
    else: #Females
        print("branch3-1") # test print to check control flow logic - delete when resolved
        df_females_selected_age = subset_pop_estimate(df_females)

        #add in the UTLA Name column
        df_females_selected_age.insert(loc=6, column='UTLA19NM', value=df_females['UTLA19NM'], allow_duplicates=True)
            
        df_females_selected_age = sum_age_range(df_females, df_females_selected_age, str(dict_condition_start_ages_from_bin_ranges[condition_name]), str(dict_condition_end_ages_from_bin_ranges[condition_name]), condition_name)
        df_females_selected_age[f"{condition_name}_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}_Estimated_Need"] = df_females_selected_age[f"Total_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}_{condition_name}"] * dict_prev_for_each_condition[condition_name]
        
        #subset the df to only contain rows for LSOAs within the UTLA of interest
        mask = df_females_selected_age['UTLA19NM'] == la_of_interest_user_list_selection
        subset_lsoa_df = df_females_selected_age[mask]
        
        lower_tier_local_authorities_all = list(set(df_females_selected_age["LA name (2021 boundaries)"]))
        lsoa_name_all = list(df_females_selected_age["LSOA Name"])

        dict_condition_estimated_need[condition_name] = subset_lsoa_df

# --------------------------------------------------------

#get list of all lsoa codes present in the service data - this may not be needed though if use processing emissions file 2 or so cells down !
list_lsoa_codes = list(dict_condition_estimated_need[list_of_conditions[0]]['LSOA Code'])

# --------------------------------------------------------

#dict_condition_list_attend_reasons
dict_condition_dataset = {}
dict_condition_dataset_all_ages_in_range = {}

for condition in list_of_conditions:
    dict_condition_dataset[condition] = pd.concat(nested_dict_condition_attend_reason[condition].values(), ignore_index=True)
    dict_condition_dataset_all_ages_in_range[condition] = pd.concat(nested_dict_condition_attend_reason_all_ages[condition].values(), ignore_index=True)

# --------------------------------------------------------

#EXPERIMENTAL CELL TO ACCOUNT FOR ONE TO MANY RELATIONSHIP BETWEEN CONDITION NAME AND ATTENDANCE REASONS IN THE DATA SET

#Code to create a dictionary of condition (key) to file paths for processed data for that condition (values) - to be used in subsequent cell
dict_condition_tempdf_unique_patients_seen_by_service = {}

dict_condition_file_names = {}
dict_condition_subset_list_file_names = {}

dict_condition_df_attend_reasons = {}

#This directory was created by the pre-processing file
directory = 'processed_data/1A_HEA'

for condition in list_of_conditions:
    condition_file_names = []
    condition_subset_list_file_names = []
    temp_dict_condition_attend_reason_df = {}

    for reason in dict_condition_list_attend_reasons[condition]:

        #iterate over each filename in the above directory
        for filename in os.listdir(directory):
            #split the filename string and save the last item in the resulting list to filename_condition_type. This will ALWAYS consist of '[condition type].csv'
            filename_condition_type = filename.split('_')[-1]
            #check if the above [condition type].csv matches the condition we are currently on in this loop. If it does, populate the lists with the full file path, and likewise the dictionary, using condition as the key. 
            #if it doesn't match, do nothing.
            if filename_condition_type == reason+'.csv':
                condition_file_names.append('tempdf_unique_patients_'+reason+'.csv')
            
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    #print(f)
                    condition_subset_list_file_names.append(f)
                    temp_dict_condition_attend_reason_df[reason] = pd.read_csv(f)
                    #dict_condition_tempdf_unique_patients_seen_by_service[condition] = f
    dict_condition_df_attend_reasons[condition] = temp_dict_condition_attend_reason_df

    dict_condition_file_names[condition] = condition_file_names
    dict_condition_subset_list_file_names[condition] = condition_subset_list_file_names

#list_of_file_names
#dict_condition_tempdf_unique_patients_seen_by_service
for condition in list_of_conditions:
    dict_condition_tempdf_unique_patients_seen_by_service[condition] = pd.concat(dict_condition_df_attend_reasons[condition].values(), ignore_index=True)

# --------------------------------------------------------

dict_condition_tempdf_unique_patients_seen_by_service_in_age_range = {}

for condition in list_of_conditions:
    temp_df = dict_condition_tempdf_unique_patients_seen_by_service[condition]
    
    mask = temp_df[age_with_median_field_name] >= dict_condition_start_ages_from_bin_ranges[condition]
    df_min_age_and_higher = temp_df[mask] # filter using mask to remove all rows with ages less than the min age for the condition

    mask = df_min_age_and_higher[age_with_median_field_name] <= dict_condition_end_ages_from_bin_ranges[condition]
    df_expressed_need_in_age_range = df_min_age_and_higher[mask] # filter using mask to remove all rows with ages less than the min age for the condition

    dict_condition_tempdf_unique_patients_seen_by_service_in_age_range[condition] = df_expressed_need_in_age_range

# --------------------------------------------------------

#this section is converting the individual dataframes for each condition, derived immediately above, into crosstabs (which are like excel pivot tables). 
#these provide a count of the number of patients (rows) present in the df from each lsoa in the column 'All' to the right hand side of the cross tab
#These are saved to file in the processed_date/1A_HEA directory to be read back into the file in the subsequent cell to finish the unmet need section

os.makedirs('processed_data/1A_HEA', exist_ok=True) 

#dictionary to 'temporarily' store the file paths to the created cross tabs in this section. This is so the cross tabs can easily be read into the subsequent sections of code.
dict_condition_unmet_need_temp_file_paths = {}

#create cross tabs of the above created dfs as temp files too. 
for condition in list_of_conditions:
    #create crosstab for count of expressed need per lsoa
    df_hea_reason_crosstab = pd.crosstab(dict_condition_tempdf_unique_patients_seen_by_service_in_age_range[condition]['lsoa_residence'], dict_condition_tempdf_unique_patients_seen_by_service_in_age_range[condition]['in_out_area_classifier'], margins=True)
    #drop the last row which is contains column totals we dont need
    df_hea_reason_crosstab.drop(index=df_hea_reason_crosstab.index[-1], axis=0, inplace=True)
    #save the resulting df to csv file
    df_hea_reason_crosstab.to_csv(f'processed_data/1A_HEA/CrosstabExpressedNeedAllAttendReasonsLSOA-{condition}.csv', index=True)
    #add file path of that csv file to a dictionary for later use
    dict_condition_unmet_need_temp_file_paths[condition] = f'processed_data/1A_HEA/CrosstabExpressedNeedAllAttendReasonsLSOA-{condition}.csv'
    print("File saved.")

# --------------------------------------------------------


dict_condition_df_met_unmet_need = {}

for condition in list_of_conditions:

    df_condition_expressed_need = pd.read_csv(dict_condition_unmet_need_temp_file_paths[condition])
    #test_df for GUM has 213 rows unique LSOAs, and a sum of 265 in the All column from those 213 rows. BUT, the 213 rows consist of Derby (x58 rows) AND Derbyshire (x155 rows).
    #This is what we should STILL HAVE after the next cell runs to calc expressed need.

    new_col_name = f'{condition}_{dict_condition_start_ages_from_bin_ranges[condition]}_to_{dict_condition_end_ages_from_bin_ranges[condition]}_Expressed_Need'

    df_condition_expressed_need.rename(columns = {'All': new_col_name}, inplace = True)

    list_unique_lsoa_in_df = list(set(df_condition_expressed_need["lsoa_residence"])) #not used?
    #len(list_unique_lsoa_in_df) #213 - correct 

    lsoa_to_upper_tier_lookup = pd.read_csv("raw_data/open_geography_portal_lookups/Lower_Layer_Super_Output_Area_(2011)_to_Upper_Tier_Local_Authorities_(2021)_Lookup_in_England_and_Wales_.csv")
    #lsoa_to_upper_tier_lookup

    #merged_df = pd.merge(test_df, lsoa_to_upper_tier_lookup, left_on="lsoa_residence", right_on="LSOA11CD", how="left")
    #merged_df

    merged_df_reversed = pd.merge(lsoa_to_upper_tier_lookup, df_condition_expressed_need, left_on="LSOA11CD", right_on="lsoa_residence", how="left")
    #test = merged_df_reversed['All'] >0
    #test.sum() #213 rows which is correct
    #merged_df_reversed['All'].sum() #265 patients from the 213 rows, correct.

    #Filter to just the UTLA of interest
    mask = merged_df_reversed["UTLA21NM"] == la_of_interest_user_list_selection
    selected_la_df = merged_df_reversed[mask] # filter using mask
    selected_la_df[new_col_name] = selected_la_df[new_col_name].fillna(0)

    #Merge into the df of service counts, the expected number for estimated need, using the dictionary dict_condition_estimated_need
    #selected_la_df_with_prevalence = pd.merge(selected_la_df, dict_condition_estimated_need['GUM'], left_on=("LSOA11CD"), right_on=("LSOA Code"), how="left")
    #selected_la_df_with_prevalence

    selected_la_df_with_prevalence = selected_la_df.merge(dict_condition_estimated_need[condition].iloc[:, [0,-1]], left_on="LSOA11CD", right_on="LSOA Code", how="left")
    selected_la_df_with_prevalence.drop("LSOA Code", axis=1, inplace=True)

    #test print - delete
    #selected_la_df_with_prevalence

    #assign to condition_age_range_string the string for the field name, for the current condition in this iteration of the loop
    condition_age_range_string = f"{condition}_{dict_condition_start_ages_from_bin_ranges[condition]}_to_{dict_condition_end_ages_from_bin_ranges[condition]}_Estimated_Need"

    #subset df to include LSOA code, selected age range, estimated need based on user provided prevalence
    final_df = selected_la_df_with_prevalence[["LSOA11CD", new_col_name, condition_age_range_string]].copy()
    #final_df

    #Need to use this code in the for condition in list_of_conditions for loop for estimating unmet need
    #f"estimated_need_{condition_name}_{dict_condition_start_ages_from_bin_ranges[condition_name]}_to_{dict_condition_end_ages_from_bin_ranges[condition_name]}"

    #Add modelled unmet need (minus figures suggest higher activity than prevalence would have suggested, i.e. >100% modelled prevalence seen)
    final_df[f"{condition}_{dict_condition_start_ages_from_bin_ranges[condition]}_to_{dict_condition_end_ages_from_bin_ranges[condition]}_Estimated_UNMET_Need"] = final_df[condition_age_range_string] - final_df[new_col_name]
    dict_condition_df_met_unmet_need[condition] = final_df

# --------------------------------------------------------
#Save the unmet need dfs to file and store the file paths in the relevant dictionary

# --------------------------------------------------------

#save as assets the df for each condition and the estimated need and unmet need for the condition, gender and chosen age range

for condition in list_of_conditions:
    file_prefix = number_saved_file(df_counter)
    temp_filename = f"df{file_prefix}_EstimatedMetAndUnmetNeed-{condition}.csv"
    dict_condition_df_met_unmet_need[condition].to_csv(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}', index=False) #dict containing actual dataframe
    dict_condition_unmet_need_file_paths[condition] = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}' #dict containing file path to saved csv
    df_counter+=1
    print("File saved.")

# --------------------------------------------------------
# <<< end of final code file for HEA >>> 
