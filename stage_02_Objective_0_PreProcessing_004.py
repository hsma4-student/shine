"""
What this file does:
This workbook undertakes pre-processing of the service's data set.
The output from this .py file is 1 or more 'processed' data files for each of the 
objectives in scope. 
The file checks for the presence of a directory structure, and if not present, creates this.
The file saves the processed files into the relevant folder of that created directory 
structure, as applicable for each component of the project.

Currently, the processing that takes place serves to clean up the messy data. 
At this point in time, this is done in one of 2 ways, dependant whether the field in question
is a numeric (float or integer) field or a string (text) field. If numeric, the code calculates
the median of that field and replaces (imputes) all Null (missing) values with the median. If string
the code replaces (imputes) the Null (missing) value with the word 'missing'. This is because Machine 
Learning algorithms can't handle missing values. 

It is important to note though that while the code will identify what fields are numeric 
and what string, it gives the user the option of confirming what fields are in scope for 
imputing missing values (median or word 'missing'). 

"""

# ---------------------------------------
#<<<     Importing Libraries     >>> 
# ---------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os

# ---------------------------------------
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

# ---------------------------------------
#<<<     Create branding colours     >>>
# ---------------------------------------

#Taken from https://www.england.nhs.uk/nhsidentity/identity-guidelines/colours/

"""
New code to use user-entered branding colours, taken from the user_params file.
"""

#identify the colour values to use from the Branding tab of the user parameters file

filename = "raw_data/user_and_data_parameters/user_and_data_params.xlsx"
#read-in user param for level_1_colour
level_1_colour = pd.read_excel(filename, 'Branding', index_col=None, usecols = "D", header = 1, nrows=0)
level_1_colour = list(level_1_colour)[0]

#read-in user param for level_2_colour
level_2_colour = pd.read_excel(filename, 'Branding', index_col=None, usecols = "D", header = 2, nrows=0)
level_2_colour = list(level_2_colour)[0]

#read-in user param for level_2_colour
level_3_colour = pd.read_excel(filename, 'Branding', index_col=None, usecols = "D", header = 3, nrows=0)
level_3_colour = list(level_3_colour)[0]

#read-in user param for level_2_colour
level_4_colour = pd.read_excel(filename, 'Branding', index_col=None, usecols = "D", header = 4, nrows=0)
level_4_colour = list(level_4_colour)[0]

# ---------------------------------------
#<<<     Define Functions     >>>
# ---------------------------------------

# ---------------------------------------
#<<<     Define Functions     >>>
# ---------------------------------------

#Identify any duplicate rows and allow user to decide whether to remove
def identify_duplicate_rows(df):
    #Could use stringtobool in this function - this recognises commonly used yes/no responses, converts to 1 or 0
    """
    >> from distutils.util import strtobool
    >> strtobool('yes')
    """
    
    original_df_count = df.count()
    original_df_num_rows = int(original_df_count[0])
    print(f'\nOriginal number of rows: {original_df_num_rows}')

    df_dedup = df.drop_duplicates()
    df_dedup_count = df_dedup.count()
    df_dedup_num_rows = int(df_dedup_count[0])
    print(f'\nNumber of unique rows if duplicates removed: {df_dedup_num_rows}')

    num_duplicates = original_df_num_rows - df_dedup_num_rows
    #print(f'Number of duplicates removed: {num_duplicates}')

    df_to_use = pd.DataFrame()

    #Remove Duplicates - User Decision if duplicates detected
    if num_duplicates != 0:
        print(f'\nWARNING: This data set has {num_duplicates} duplicate rows:')
        
        df_dup_rows = df.duplicated()
        print(df[df_dup_rows])

        print('\nWould you like to remove these duplicate rows?')
        while True:
            try:
                remove_dups = int((input('\nPress 1 to remove or 2 to retain >>>> ')))
                if remove_dups == 1:
                    df_to_use = df_to_use.append(df_dedup)
                    break
                elif remove_dups == 2:
                    df_to_use = df_to_use.append(df)
                    break
                else:
                    print('That is not a valid selection.')
            except:
                print('That is not a valid selection.')
        return df_to_use
    else:
        print('This data set has no duplicate rows.')
    
        return df

# ---------------------------------------
#Allow user to define what column heading / data field name is a unique identifier for patients
#such as customer ID, patient ID etc. 
#Assign to variable name of "patient_id", which is then used throughout the rest of the program
#Attempts to control for typos / name variations etc. in data sets for different use cases

def user_field_selection(field_list, field_name):
    dict_of_fields = {}
    key_counter = 1
    for value in field_list:
        dict_of_fields[key_counter] = value
        key_counter += 1

    print("\nThe data fields present in the data set are listed below")
    for key_number in dict_of_fields:
        print(str(key_number) + " : " + str(dict_of_fields[key_number]))

    print(f"\nPlease enter the number that represents the {field_name} field in the above list:\n")

    while True:
        try:
            patient_id_num = int(input('Make a selection: '))
            break

        except ValueError:
            print('Invalid Input. Please enter a number.')

    print(f"\nGreat. The entered number was: {patient_id_num}")
    print(f"\nThe {field_name} field that the program will use is: {dict_of_fields[patient_id_num]}")

    return dict_of_fields[patient_id_num]


# ---------------------------------------

def data_quality_summary(df, field_list):
    present_values = []
    missing_values = []
    percent_missing = []

    list_of_lists = [present_values, missing_values, percent_missing]
    list_names = ["Present", "Missing", "% Missing"]

    for col_name in field_list:
        col_missing = df[col_name].isna().sum()
        missing_values.append(col_missing)

        col_present = df[col_name].count()
        present_values.append(col_present)

        try:
            percent = (col_missing / (col_missing + col_present)) * 100
            percent_missing.append(round(percent,1))
        except:
            percent_missing.append(round(0.0,1))

    data_quality_summary_df = pd.DataFrame(list_of_lists, columns = field_list, index = list_names)

    #create new lists, representing counts of present / missing data by 
    #variable, and percent missing
    #To enable summary table beneath chart (subsequent cell)

    present = data_quality_summary_df.loc["Present"].values.tolist()
    missing = data_quality_summary_df.loc["Missing"].values.tolist()
    missing_pct = data_quality_summary_df.loc["% Missing"].values.tolist()
    
    return list_names, present, missing, missing_pct, data_quality_summary_df

# ---------------------------------------

#consider adding second axis plot for % missing
#This could improve interpretation (where missing count small, its invisible on the chart...)

def present_missing_stacked_bar(
        fields_list, 
        present, 
        missing, 
        present_color, 
        missing_color, 
        title, 
        filename, 
        file_location):
    
    #width = 0.35
    width = 0.5
   
    fig, ax = plt.subplots()
    
    ax.bar(fields_list, present, width, label="Present Values", color=present_color)
    ax.bar(fields_list, missing, width, bottom=present, label="Missing Values", color=missing_color)

    ax.set_ylabel("Value count")
    ax.set_title(f"Missing Values Summary: {title}")
    ax.tick_params(axis='x', labelrotation = 90)
    ax.legend(loc="lower right")

    plt.savefig(file_location+filename+".png", bbox_inches="tight")
    plt.show()

# ---------------------------------------

#https://www.pythonpool.com/matplotlib-table/
#data_quality_summary_df.loc[data_quality_summary_df['column_name'] == some_value]

def present_missing_stacked_bar_with_table(
    list_names, 
    fields_list, 
    present, 
    missing, 
    missing_pct,
    present_color,
    missing_color,
    pct_color,
    title,
    filename,
    file_location):

    plt.rcParams['figure.figsize'] = [10, 5]
    data = [present, missing, missing_pct]

    columns = fields_list
    rows = list_names #present, missing, % missing 

    colors = [present_color, missing_color, pct_color]

    n_rows = len(data) #3

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []

    for row in range(n_rows):
        #y_offset_edit = y_offset + data[row]
        cell_text.append([f'{x:.1f}' for x in data[row]])
        if "%" in rows[row]:
            continue
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom')

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel("Count of rows in the data set used")

    #plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.title(f"Missing Data Summary: {title}.")

    plt.savefig(file_location+filename+".png", bbox_inches="tight")

    plt.show()

# ---------------------------------------

#Function to visually check if missing at random or not
#Definte function to visualise matrix of missing values
#Missing data appears as white space in the chart
#Development idea: set fig size for the this (?and possibly, all) chart(s) to ensure consistency. Esp. if creating a report at the end?

def dq_matrix(df, filename, file_location):
    matrix = msno.matrix(df)
    matrix_copy = matrix.get_figure() #new line
    matrix_copy.savefig(file_location+filename+".png", bbox_inches="tight") #changed from matrix to matrix_copy
    return matrix

# ---------------------------------------

#The missingno correlation heatmap measures nullity correlation: 
#how strongly the presence or absence of one variable affects the presence of another: 
#(source: https://github.com/ResidentMario/missingno)

def heatmap(df, filename, file_location):
    print("WARNING: Results in the below chart that show '<1' or '>-1' suggest the presence of a correlation that is close to being exactly negative or positive (but is still not quite perfectly so) and cases requiring special attention.")
    heatmap = msno.heatmap(df)
    heatmap_copy = heatmap.get_figure() #newline

    heatmap_copy.savefig(file_location+filename+".png", bbox_inches="tight") #change name from heatmap
    return heatmap

# ---------------------------------------

def missing_values_field_names(fields_list, missing):
    """
    dict datafields (keys) and missing series (values)
    iterate over values in datafields dict, check if >0
    if so, create new dict of the subset where there are missing values (misisng == >0)
    then, iterate over that, and use each key/value pair, plug into impute function
    """

    keys = fields_list
    values = missing
    #dict_fieldname_missing_values = dict(zip(keys, values))
    #dict_fieldname_missing_values

    #Create dictionary of field names (keys) and county of missing values (values)
    iterator_fieldname_missing_values = zip(keys, values)
    dict_fieldname_missing_values = dict(iterator_fieldname_missing_values)

    #Use this dictionary to create a list, consisting of the subset of field names, where there are missing values
    #This will be used later in the program to identify which fields require missing values to be accounted for in some way
    missing_values_fields = []
    fields_none_missing = []

    for field in fields_list:
        if dict_fieldname_missing_values[field] >0:
            missing_values_fields.append(field)
        else:
            fields_none_missing.append(field)

    return dict_fieldname_missing_values, missing_values_fields 

# ---------------------------------------

#Function adjusted from the titanic example by Mike Allen
def impute_missing_with_median(series):
    """
    Replace missing values in a Pandas series with median,
    Returns a completed series, and a series showing which values are imputed
    Required because ML algorithms like Log regression cannot handle missing / Null values.
    By imputing missing values we are replaced all Null values in a numeric series with the 
    median of that series.
    """
    
    median = series.median(skipna = True) #ignore missing values in calculation
    missing = series.isna()
    
    # Copy the series to avoid change to the original series.
    series_copy = series.copy()
    series_copy[missing] = median
    
    return series_copy, missing

# ---------------------------------------

def group_fields_by_data_type(df, fields_list, dict_fieldname_missing_values):
    """
    Identify which fields both have missing values and are an appropriate data type to have their median calculated.
    This doesn't consider whether an int/float field is contextually appropriate to have a median calculated.
    E.g. if appt_id is an incremental numeric field, it will be logically identified and assigned to int_float_fields
    but, it is contextually meaningless to give a median appt_id. 
    Therefore, the output of this step needs some form of user validation / approval to proceed.
    This is to be handled in the next cell.
    """
    #df_with_imd_unique_pts.dtypes
    data_type_dict = dict(df.dtypes)

    int_float_fields = []
    str_fields = []
    datetime_fields = []

#original - poss error in df_with_imd_unique_pts
    """
    #identify which fields both have missing values and are int or float type. Append to list.
    for field in fields_list:
        if dict_fieldname_missing_values[field] > 0:
            if df_with_imd_unique_pts[field].dtype == "float" or df_with_imd_unique_pts[field].dtype == "int":
                int_float_fields.append(field)
            elif df_with_imd_unique_pts[field].dtype == "<M8[ns]":
                datetime_fields.append(field)
            else:
                str_fields.append(field)
    return int_float_fields, str_fields, datetime_fields, data_type_dict
    """

#experimental code attempt to correct error 

    #identify which fields both have missing values and are int or float type. Append to list.
    for field in fields_list:
        if dict_fieldname_missing_values[field] > 0:
            if df[field].dtype == "float" or df[field].dtype == "int":
                int_float_fields.append(field)
            elif df[field].dtype == "<M8[ns]":
                datetime_fields.append(field)
            else:
                str_fields.append(field)
    return int_float_fields, str_fields, datetime_fields, data_type_dict

# ---------------------------------------

"""
User confirmation these are all logically / contextually relevant 
(e.g. numeric patient ID could have median calculated but this doesnt make sense and
should be removed before the process to control for missing values takes place)

This function requires two input parameters: 
1. a field_list, and 
2. a data type as a string ("int" or "string")
"""


def identify_fields_to_impute(fields_list, data_type):
    #tell python which variables from the global scope to update
    global int_float_fields_keep
    global int_float_fields_drop
    global str_fields_keep
    global str_fields_drop
    global datetime_fields_keep
    global datetime_fields_drop
    
    #Create a dictionary of incremental numbers starting from 1 (keys) and 
    #field names identified in the last step as being int/float data type and having missing values (values)   
    dict_of_missing_fields = {}
    key_counter = 1
    for value in fields_list:
        dict_of_missing_fields[key_counter] = value
        key_counter += 1

    #Print to screen a numbered 'list' to the user, listing the fields that have been identified as int/float and missing values
    if data_type == "int":
        data_type_descriptor = "numeric (integer/decimal)"
    elif data_type == "str":
        data_type_descriptor = "string (text)"
    else:
        data_type_descriptor = "date time type"
        
    print(f"\nThe data fields with missing data that look to be {data_type_descriptor} are listed below")
    
    for key_number in dict_of_missing_fields:
        print(str(key_number) + " : " + str(dict_of_missing_fields[key_number]))

    #Get user confirmation that it is contextually appropriate to proceed with these fields for missing value control
    print("\nDo you want to impute missing data for all of these fields?")
    print("\nPlease enter 'y' to continue with all fields in scope, or 'n' to choose which fields to drop")

    while True:
        try:
            selection = input("Please make a selection (y or n): ").lower()
            if selection == "y" or selection == "n":
                break
            print("Invalid selection")
        except ValueError:
            print("\nThat is not a valid selection. Please try again (y or n).")   

    #loop to choose which fields to drop from the missing value process
    if selection == "n":
        while True:
            print("\nThe data fields currently in scope are listed below:")
            for key_number in dict_of_missing_fields:
                print(f"{key_number} : {dict_of_missing_fields[key_number]}")
            print("0 : To continue with the current selection")
        
            while True:
                try:
                    drop_selection = int(input("\nPlease enter the number of the field to remove, or 0 to continue with the current selection:"))
                    if drop_selection in range(key_counter):
                        break
                    else:
                        print("\nThat is not a correct selection.")
                except ValueError:
                    print("Value Error - try again.")

            if drop_selection == 0:
                break
            else:
                if data_type == "int":
                    int_float_fields_drop.append(dict_of_missing_fields[drop_selection]) #create new list of fields user does not want to control for missing values
                    del dict_of_missing_fields[drop_selection]
                elif data_type == "str":
                    str_fields_drop.append(dict_of_missing_fields[drop_selection]) #create new list of fields user does not want to control for missing values
                    del dict_of_missing_fields[drop_selection]
                else:
                    datetime_fields_drop.append(dict_of_missing_fields[drop_selection]) #create new list of fields user does not want to control for missing values
                    del dict_of_missing_fields[drop_selection]

        else:
            print("Please enter a valid selection.")
    
    if data_type == "int":
        int_float_fields_keep = fields_list
        for item in int_float_fields_drop:
            int_float_fields_keep.remove(item)
    elif data_type == "str":
        str_fields_keep = fields_list
        for item in str_fields_drop:
            str_fields_keep.remove(item)
    else:
        datetime_fields_keep = fields_list
        for item in datetime_fields_drop:
            datetime_fields_keep.remove(item)

    print()
    print(f"\nGreat. The following {data_type_descriptor} fields will be used when controlling for missing data:")
    
    if data_type == "int": 
        for item in int_float_fields_keep:
            print(item)
    elif data_type == "str":
        for item in str_fields_keep:
            print(item)   
    else:
        for item in datetime_fields_keep:
            print(item)   


# ---------------------------------------
"""
Decision / consideration needed:
>Need to decide whether to remove df rows based on missing values in 
int_float_fields_drop and str_fields_drop lists or exclude the df columns these fields 
relate to entirely, as ML can't handle Nulls.
Current method involves removing the rows.
"""

def remove_inappropriate_appt_statuses(df, df_field, field_name):    
    print(f"\nThe {field_name} field that the program will use is: {df_field}")
    print("\nThe program requires a binary outcome measure (e.g. attended or did not attend).")
        
    ls = list(set(df[df_field]))
    
    unique_options_attend_status = [item for item in ls if type(item) != float]
    
    dict_of_unique_options = {}
    appt_status_to_drop = []
    appt_status_to_keep = []

    if len(unique_options_attend_status) == 2:
        print(f"{df_field} is a binary measure - perfect!")
        
        unique_key_counter = 1
        for option in unique_options_attend_status:
            dict_of_unique_options[unique_key_counter] = option
            unique_key_counter += 1

    elif len(unique_options_attend_status) > 2:
        print(f"\n{df_field} has too many possible appointment statuses.")
        print("\nThe unique appointment statuses are listed below.")
        print("One by one, please enter the number for each option that represents a staus other than attended and did not attend (e.g. patient cancelled, etc.)")
        print("This will remove rows for those appointments from the data set to be used in the analysis.")
        print("Once the list only includes 2 appointment statuses, enter 0 to continue.")
              
        #dict_of_unique_options = {}
        unique_key_counter = 1
        for option in unique_options_attend_status:
            dict_of_unique_options[unique_key_counter] = option
            unique_key_counter += 1    

        #appt_status_to_drop = []
        #appt_status_to_keep = []
        
        appt_status_to_drop.append("nan")
        
        #loop to choose which fields to drop from the missing value process
        while True:
            print("\nThe appointment statuses currently in scope are listed below:")
            for key_number in dict_of_unique_options:
                print(f"{key_number} : {dict_of_unique_options[key_number]}")
            print("0 : To continue with the current selection")

            while True:
                try:
                    drop_selection = int(input("\nPlease enter the number of the appointment status to remove, or 0 to continue with the current selection:"))
                    if drop_selection in range(unique_key_counter):
                        break
                    else:
                        print("\nThat is not a correct selection.")
                except ValueError:
                    print("Value Error - try again.")

            if drop_selection != 0:
                appt_status_to_drop.append(dict_of_unique_options[drop_selection]) #create new list of appt statuses user wants to exclude
                del dict_of_unique_options[drop_selection]
            elif drop_selection == 0:
                break

#                else:
#                    appt_status_to_keep.append(dict_of_unique_options[drop_selection]) #create new list of appt statuses user wants to include
#                    del dict_of_unique_options[drop_selection]

        else:
            print("Please enter a valid selection.")
    
    for key in dict_of_unique_options:
        appt_status_to_keep.append(dict_of_unique_options[key])
    
#    else:
#        continue #WARNING possible point of failure if there is only one appointment status in the column, not handled currently
          
    return appt_status_to_drop, appt_status_to_keep

#print(patient_id) #test print statement, delete


# ---------------------------------------

#Consider how to change the terminology used here, in case the term "missing" is used in the actual data set passed
#Taken from titanic - 01. pre-processing example:
def impute_missing_with_missing_label(series):
    """Replace missing values in a Pandas series with the text 'missing'"""

    missing = series.isna()    
    
    # Copy the series to avoid change to the original series.
    series_copy = series.copy()
    series_copy[missing] = 'missing'
    
    return series_copy, missing

# ---------------------------------------
#REPLACED WITH NEW FUNCTION BELOW
'''
def impute_missing(field_list, data_type):
    
    #Tell the function to update variables that exist in the global space, and not locally within the function
    global dict_ints_imputed
    global dict_ints_imputed_missing
    global dict_str_imputed
    global dict_str_imputed_missing
    
    if data_type == "int":
        for field in field_list:
            series, missing = impute_missing_with_median(obj_1a_df_unique_pts[field])
            dict_ints_imputed[field] = series
            dict_ints_imputed_missing[field] = missing

    else:
        for field in field_list:
            series, missing = impute_missing_with_missing_label(obj_1a_df_unique_pts[field])
            dict_str_imputed[field] = series
            dict_str_imputed_missing[field] = missing
'''

def impute_missing(df, field_list, data_type):
    
    #Tell the function to update variables that exist in the global space, and not locally within the function
    global dict_ints_imputed
    global dict_ints_imputed_missing
    global dict_str_imputed
    global dict_str_imputed_missing
    
    if data_type == "int" or data_type == 'float':
        for field in field_list:
            series, missing = impute_missing_with_median(df[field])
            dict_ints_imputed[field] = series
            dict_ints_imputed_missing[field] = missing

    else:
        for field in field_list:
            series, missing = impute_missing_with_missing_label(df[field])
            dict_str_imputed[field] = series
            dict_str_imputed_missing[field] = missing

# ---------------------------------------

#duplicate function for impute missing objective 1b - error using original for 2nd time
#testing new solution with this approach
def impute_missing_1b(df, field_list, data_type):
    
    #Tell the function to update variables that exist in the global space, and not locally within the function
    global dict_ints_imputed_1b
    global dict_ints_imputed_missing_1b
    global dict_str_imputed_1b
    global dict_str_imputed_missing_1b
    
    if data_type == "int" or data_type == 'float':
        for field in field_list:
            series, missing = impute_missing_with_median(df[field])
            dict_ints_imputed_1b[field] = series
            dict_ints_imputed_missing_1b[field] = missing

    else:
        for field in field_list:
            series, missing = impute_missing_with_missing_label(df[field])
            dict_str_imputed_1b[field] = series
            dict_str_imputed_missing_1b[field] = missing
            

# ---------------------------------------
'''
#original function
#identify the fields that are labels and not numbers (as the encoding will be done on label data only)
def fields_to_encode_or_drop(df):
    final_processed_data_fields = list(df.columns)
    fields_to_not_encode = []
    fields_to_encode = []
    fields_to_remove = [] #unsure if this is needed? had thought so for bools, but now removed those via alternative method

    for field in final_processed_data_fields:
        if field != patient_id and field != appt_id:
            if df[field].dtype == "float64" or df[field].dtype == "int" or df[field].dtype == "<M8[ns]":
                fields_to_not_encode.append(field)
            #elif df[field].dtype == "<M8[ns]":
            #    fields_to_remove.append(field)
            else: 
                fields_to_encode.append(field)
    
    return fields_to_not_encode, fields_to_encode, fields_to_remove
'''

#revised function to remove columns from the df where they are all blank
#revised function 18/9
#identify the fields that are labels and not numbers (as the encoding will be done on label data only)
def fields_to_encode_or_drop(df):
    #revised to include the global variables
    global patient_id
    global appt_id
    global age
    global lsoa_residence
    
    final_processed_data_fields = list(df.columns)
    fields_to_not_encode = []
    fields_to_encode = []
    fields_to_remove = [] #unsure if this is needed? had thought so for bools, but now removed those via alternative method

    for field in final_processed_data_fields:
        #if field != patient_id and field != appt_id and field != age and field != lsoa_residence: #revised to exclude age and lsoa from encoding
        if field != patient_id and field != appt_id and field != lsoa_residence: #revised to exclude age and lsoa from encoding
            if df[field].dtype == "float64" or df[field].dtype == "int" or df[field].dtype == "<M8[ns]" or df[field].dtype == 'int64':
                fields_to_not_encode.append(field)
            
            else:
                try:
                    if df[field].value_counts()['missing'] == df.shape[0]:
                        fields_to_remove.append(field)
                except KeyError:
                    fields_to_encode.append(field)
    
    return fields_to_not_encode, fields_to_encode, fields_to_remove
# ---------------------------------------

#create function to encode all relevant fields for a given data frame and list of fields to encode, passed into the function as parameters
def encoding_dataframe(dataframe, field_list_to_encode, field_list_to_not_encode):

    unencoded_df = pd.DataFrame()
    for field in field_list_to_not_encode:
        unencoded_df = pd.concat([unencoded_df, dataframe[field]], axis=1)
    
    #Create blank dataframe that will be populated with encoded series
    encoded_dataframe = pd.DataFrame()
    
    #iterate through all fields in the dataframe passed to the function
    for field in field_list_to_encode:       
        
        #encode the current field in the loop and create a dataframe for the current fields encoding 
        field_encoded = pd.get_dummies(dataframe[field], prefix=f"encoded_{field}")
        
        #concatenate the current fields encoding with the newly create dataframe above
        encoded_dataframe = pd.concat([encoded_dataframe, field_encoded], axis=1)

    combined_df = pd.concat([unencoded_df, encoded_dataframe], axis=1)
        
    return combined_df

# ---------------------------------------

#Function to remove bool series from final produced df
def remove_booleans_from_df(df):
    final_processed_df_bools_removed = pd.DataFrame()
    
    for column in df:
        if df[column].dtype != 'bool':
            final_processed_df_bools_removed[column] = df[column]
    return final_processed_df_bools_removed

# ---------------------------------------

#function to create subset dataframes from main df, for each objective in project
#def make_df(df, dict_field_to_keep):
#    subset_df = df.copy()
#    for key in dict_field_to_keep.keys():
#        if dict_field_to_keep[key] == 'n':
#            subset_df = subset_df.drop(key, 1)
#    return subset_df

#new function to control for capitalised field names in raw source data - replace commented out function above 17/09
#function to create subset dataframes from main df, for each objective in project
def make_df(df, dict_field_to_keep):
    subset_df = df.copy()
    for key in dict_field_to_keep.keys():
        if dict_field_to_keep[key] == 'n':
            subset_df = subset_df.drop(key.lower(), 1)
    return subset_df


# ---------------------------------------

def assign_field_name_to_variable(filename, sheetname, row_num):
    """
    function to assign a given user-provided field names in the 'user_data_and_params.xlsx file, 
    to a variable name.  
    """
    field_name = pd.read_excel(filename, sheetname, index_col=None, usecols = "B", header = row_num, nrows=0)
    field_name = field_name.columns.values[0]
    try: 
        field_name = (field_name.split("."))[0]
    except:
        pass
    return(field_name)

# ---------------------------------------

def press_any_key_to_continue():
    """
    Function to pause the code from running, requiring a user to hit any key for code 
    processing to resume. 
    """
    print("")
    print("-" * 44)
    print("<<<<<    Press any key to continue    >>>>> ")
    print("-" * 44)
    input()

# ---------------------------------------

#function to provide numbered prefixes to file names for files saved as outputs from the code
def number_saved_file_preprocessing(counter):
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

# ---------------------------------------
# ---------------------------------------

# <<<     CODE STARTS!     >>>

# ---------------------------------------
# ---------------------------------------

#setting up parameters
#counters for file names
df_counter = 1
chart_counter = 1

# Read in user_and_data_params file assign to filename variable
filename = "raw_data/user_and_data_parameters/user_and_data_params.xlsx"

# ---------------------------------------

#Assign all field names from the data set to variable names, for use throughout the code
patient_id = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 1).lower()
appt_id = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 2).lower()
age = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 3).lower()
lsoa_residence = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 4).lower()
ethnicity = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 5).lower()
gender_at_birth = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 6).lower()
attend_status = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 7).lower()
appt_type = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 8).lower()
appt_date_time = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 9).lower()
gp_practice_code = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 10).lower()
religion = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 11).lower()
disability = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 12).lower()
pregnant = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 13).lower()
clinic_name = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 14).lower()
attend_reason = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 15).lower()
activity_in_van = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 16).lower()
in_out_area_classifier = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 17).lower()
imd_decile = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 18).lower()
clinic_format = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 19).lower()

# ---------------------------------------

#sense checking cell - include in final - print at the end of this section exists to print to the screen for the user to see the code has read in th right variable to the relevant string name

#sense checking cell - include in final - print at the end of this section exists to print to the screen for the user to see the code has read in th right variable to the relevant string name

list_var_names = [
    "patient_id",
    "appt_id",
    "age",
    "lsoa_residence",
    "ethnicity",
    "gender_at_birth",
    "attend_status",
    "appt_type",
    "appt_date_time",
    "gp_practice_code",
    "religion",
    "disability",
    "pregnant",
    "clinic_name",
    "attend_reason",
    "activity_in_van",
    "in_out_area_classifier",
    "imd_decile",
    "clinic_format"
]

dict_var_names_original = {
    "patient_id": patient_id,
    "appt_id": appt_id,
    "age": age,
    "lsoa_residence": lsoa_residence,
    "ethnicity": ethnicity,
    "gender_at_birth": gender_at_birth,
    "attend_status": attend_status,
    "appt_type": appt_type,
    "appt_date_time": appt_date_time,
    "gp_practice_code": gp_practice_code,
    "religion": religion,
    "disability": disability,
    "pregnant": pregnant,
    "clinic_name": clinic_name,
    "attend_reason": attend_reason,
    "activity_in_van": activity_in_van,
    "in_out_area_classifier": in_out_area_classifier,
    "imd_decile": imd_decile,
    "clinic_format": clinic_format
}


print("The following lower-case field names have been assigned to each variable (variable name : field name):")
for var_string in list_var_names:
    print(f"{var_string} : {dict_var_names_original[var_string]}")

#run press_any_key_to_continue function, to pause the code to allow the user to check the fields have aligned
press_any_key_to_continue()

# ---------------------------------------

#Read in the data set / create DataFrame
csv_files_in_folder = []

# traverse the directory
for root, dirs, files in os.walk(r'raw_data/local_service_data_test'):
    # select file name
    for file in files:
        # check the extension of files
        if file.endswith('.csv'):
            # print whole path of files
            csv_files_in_folder.append(file)

if len(csv_files_in_folder) > 1: #when there are more than 1 csv file present, have user choose which file to use
    dict_int_filename = {}
    for num in range(len(csv_files_in_folder)):
        dict_int_filename[num] = csv_files_in_folder[num]
    for num in range(len(csv_files_in_folder)):
        print(f"{num}: {csv_files_in_folder[num]}")
    
    user_file_selection = int(input("Please enter the number associated with the data file you wish to use, from the printed list."))
    df = pd.read_csv(f'raw_data/local_service_data_test/{csv_files_in_folder[user_file_selection]}') #extended dataset
    df.columns = map(str.lower, df.columns) #18/09 addition. zxcv

else: #when there is only 1 file in the list, use this file
    df = pd.read_csv(f'raw_data/local_service_data_test/{csv_files_in_folder[0]}') #extended dataset
    df.columns = map(str.lower, df.columns) #18/09 addition. zxcv
#print(df.head())

# ---------------------------------------
#code to rename all columns to expected col names.
copy_df_rename_cols = df.copy(deep=True)

for col in list_var_names:
    copy_df_rename_cols.rename(columns={dict_var_names_original[col]: col}, inplace=True)

#replace df variable with the copy which has expected naming convention for the columns
df = copy_df_rename_cols

# ---------------------------------------
#create new variables for each var containing the standardised col names
patient_id = "patient_id"
appt_id = "appt_id"
age = "age"
lsoa_residence = "lsoa_residence"
ethnicity = "ethnicity"
gender_at_birth = "gender_at_birth"
attend_status = "attend_status"
appt_type = "appt_type"
appt_date_time = "appt_date_time"
gp_practice_code = "gp_practice_code"
religion = "religion"
disability = "disability"
pregnant = "pregnant"
clinic_name = "clinic_name"
attend_reason = "attend_reason"
activity_in_van = "activity_in_van"
in_out_area_classifier = "in_out_area_classifier"
imd_decile = "imd_decile"
clinic_format = "clinic_format"
# ---------------------------------------
#now recreate the dict_var_names variable consisting of the standardised col names
dict_var_names = {
    "patient_id": patient_id,
    "appt_id": appt_id,
    "age": age,
    "lsoa_residence": lsoa_residence,
    "ethnicity": ethnicity,
    "gender_at_birth": gender_at_birth,
    "attend_status": attend_status,
    "appt_type": appt_type,
    "appt_date_time": appt_date_time,
    "gp_practice_code": gp_practice_code,
    "religion": religion,
    "disability": disability,
    "pregnant": pregnant,
    "clinic_name": clinic_name,
    "attend_reason": attend_reason,
    "activity_in_van": activity_in_van,
    "in_out_area_classifier": in_out_area_classifier,
    "imd_decile": imd_decile,
    "clinic_format": clinic_format
}
# ---------------------------------------

#Identify duplicate rows using pre-determined function
df_to_use = identify_duplicate_rows(df)

# ---------------------------------------

imd_present = input("Does your data set have an IMD field present? (y/n)").lower()
if imd_present == 'n':
    
    #Library called requests (pip install needed) - commonly used for http request
    #requests.get(url) - this could get the data from a file on a given website, 
    #store this in dataframe (or save it)

    #Current non-API approach referencing downloaded full LSOA to IMD decile csv 
    # from https://opendatacommunities.org/

    #fine for pilot / project, beyond that would need to be scalable
    df_lsoa_imd_decile = pd.read_csv('raw_data/open_data_communities/imd2019lsoa_decile.csv') 
    df_lsoa_imd_decile = df_lsoa_imd_decile[['FeatureCode', 'Value']]
    
    #merge IMD decile value into service data - result is new dataframe
    df_with_imd = df_to_use.merge(df_lsoa_imd_decile, left_on='lsoa_residence', right_on='FeatureCode', how='left').drop(columns='FeatureCode', axis=1) #was left_on = 'LSOA_residence' hard coded. 
    df_with_imd.columns = df_with_imd.columns.str.lower()
    df_with_imd = df_with_imd.rename(columns={"value": "IMD_decile", "attendance_status": "attend_status"})
    
    #format all column heads to have no trailing whitespace and lowercase
    df_with_imd.columns= df_with_imd.columns.str.strip().str.lower()

else:
    df_with_imd = df_to_use
    df_with_imd = df_with_imd.columns.str.lower()

# ---------------------------------------
#<<< DQ section start for the whole df >>>
# ---------------------------------------
df_with_imd_fields = list(df_with_imd.columns)

#Data quality description section
list_names, present_values, missing_values, missing_pct, data_quality_summary_df = data_quality_summary(df_with_imd, df_with_imd_fields)

#Test print statement for above function - can be used in final report
#print(data_quality_summary_df)
file_number = number_saved_file_preprocessing(df_counter)
df_counter+=1

data_quality_summary_df.to_excel(f"{preprocessing_assets_path}/df{file_number}_Data_Quality_Summary_Df_whole_dataset.xlsx")


#Produce a stacked bar chart
#save the chart to disk, for later use as an asset in a compiled report

file_number = number_saved_file_preprocessing(chart_counter)
chart_counter+=1

temp_file_name = f"/chart{file_number}_stacked_bar_missing_values_per_field"

#create stack bar chart showing counts of present or missing values per column in the df
stacked_bar_1a = present_missing_stacked_bar(
    df_with_imd_fields, 
    present_values, 
    missing_values, 
    level_3_colour, 
    level_4_colour, 
    "raw data set",
    temp_file_name,
    preprocessing_assets_path
    )

# ---------------------------------------

#Call the matrix function to display DQ matrix
#Consider how to save to file, for use as an asset in a to be compiled report
file_number = number_saved_file_preprocessing(chart_counter)
chart_counter+=1

temp_file_name = f"/chart{file_number}_whole_dataset_dq_matrix"

matrix_1a = dq_matrix(df_with_imd, temp_file_name, preprocessing_assets_path)

# ---------------------------------------

#produce dq heat map for the raw data set
#Consider how to save to file, for use as an asset in a to be compiled report
file_number = number_saved_file_preprocessing(chart_counter)
chart_counter+=1

temp_file_name = f"/chart{file_number}_whole_dataset_dq_heatmap"

heatmap_1a = heatmap(df_with_imd, temp_file_name, preprocessing_assets_path)

# ---------------------------------------
#19/09 new code to remove rows where lsoa of residence or imd decile are blank (NaN). 
print(df_with_imd.shape)
df_with_imd.dropna(subset=['lsoa_residence','imd_decile'], inplace=True)
print(df_with_imd.shape)
# ---------------------------------------

#Identify the fields (columns) present in the data set
data_fields = list(df_with_imd.columns)

#Test print for user to visually check the joining of the datasets has worked
#print(df_with_imd.head())

# ---------------------------------------

#Create dictionaries to "chunk up" the base data into the 3 different dataframes
#required for each objective in the project

#Create a dictionary for each objectives' data set to consist of field names (keys)
#and boolean y/n for whether that field is to included in that objectives df
#need to add dummy data for each of the fields in the dicts below then
#using the user_field_selection function to assign ACTUAL field name for each
#field to the variables (key descriptors) used in the dicts
#This will enable us to work through building up / processing each of the 3
#DataFrames in turn, and dropping/keeping the relevant columns for each
#THEN, using getdummies, to one hot encode those that are appropriate and retained?

#dictionary to ID which fields to include in data set 1a
#some fields included at this point to enable further processing, will be 
#removed before final df saved, e.g. 
#patient_id : use to retain unique patients only, then remove field
#attend_status : use to retain attended appts only, then remove field
#appt_date_time : use to sort df ascending, to retain most recent appts only, then remove
#in_out_area_classifier : retain in area only, then remove

dict_obj_1a_HEA_UML_fields_to_inc = {
    patient_id: 'y',
    appt_id: 'n',
    age: 'y', 
    lsoa_residence: 'y',
    ethnicity: 'y',
    gender_at_birth: 'y',
    attend_status: 'y',
    appt_type: 'n',
    appt_date_time: 'y',
    gp_practice_code: 'n',
    religion: 'y',
    disability: 'y',
    pregnant: 'y', 
    clinic_name: 'n', 
    attend_reason: 'y', 
    activity_in_van: 'n', 
    in_out_area_classifier: 'y',
    imd_decile: 'y',
    clinic_format: 'n'
    }

#dictionary to ID which fields to include in data set 1b
#some fields included at this point to enable further processing, will be 
#removed before final df saved, e.g. 
#appt_id : use to retain unique appts only, then remove
#attend_status : use to retain attended / DNA appts only, then remove
#appt_type : use to retain face to face appts only, then remove
#in_out_area_classifier : retain in area only, then remove
dict_obj_1b_DNA_SML_fields_to_inc = {
    patient_id: 'n',
    appt_id: 'y',
    age: 'y', 
    lsoa_residence: 'n',
    ethnicity: 'y',
    gender_at_birth: 'y',
    attend_status: 'y',
    appt_type: 'y',
    appt_date_time: 'n',
    gp_practice_code: 'n',
    religion: 'y',
    disability: 'y',
    pregnant: 'y', 
    clinic_name: 'y', 
    attend_reason: 'y', 
    activity_in_van: 'n', 
    in_out_area_classifier: 'y',
    imd_decile: 'y',
    clinic_format: 'y'
    }

#dictionary to ID which fields to include in data set 2
#some fields included at this point to enable further processing, will be 
#removed before final df saved, e.g. 
#appt_id : use to retain unique appts only, then remove
#attend_status : use to retain attended appts only, then remove
#appt_type : use to retain face to face appts only, then remove
#in_out_area_classifier : retain in area only, then remove
#activity_in_van included to set up cross tabs, then removed
#attend_reason included to set up cross tabs, then removed

#prev inc: age, gender_at_birth, gp_practice_code, in_out_area_classifier, imd_decile
dict_obj_2_carbon_emissions_fields_to_inc = {
    patient_id: 'n',
    appt_id: 'y',
    age: 'n',   
    lsoa_residence: 'y',
    ethnicity: 'n',
    gender_at_birth: 'n',
    attend_status: 'y',
    appt_type: 'n',
    appt_date_time: 'n',
    gp_practice_code: 'n',
    religion: 'n',
    disability: 'n',
    pregnant: 'n', 
    clinic_name: 'y', 
    attend_reason: 'y', 
    activity_in_van: 'y', 
    in_out_area_classifier: 'y',
    imd_decile: 'y',
    clinic_format: 'y'
    }

# ---------------------------------------
#dict to check for cols where all value are missing, and where these should be string, auto replace all to the word missing
dict_obj1a_cols_to_dtype = {
    patient_id: str,
    age: 'int64',
    lsoa_residence: str,
    ethnicity: str,
    gender_at_birth: str,
    attend_status: str,
    appt_type: str,
    gp_practice_code: str,
    religion: str,
    disability: str,
    pregnant: str,
    clinic_name: str,
    attend_reason: str,
    activity_in_van: str,
    in_out_area_classifier: str,
    clinic_format: str,
    imd_decile: 'int64'
}

# ---------------------------------------

#create a dict of dicts, consisting of the above 3 dictionaries created - this will be used in the final report to illustrate what fields were in / out of scope at the outset of each stage
dict_of_dicts_of_fields_in_each_section = {
    '01. HEA': dict_obj_1a_HEA_UML_fields_to_inc,
    '02. DNA Profile': dict_obj_1b_DNA_SML_fields_to_inc,
    '03. Est. Pt. Travel Carbon Emissions': dict_obj_2_carbon_emissions_fields_to_inc
    }

#turn the dict of dicts created above into a data frame for later use in the report
df_summary_of_fields_used_in_each_section = pd.DataFrame.from_dict(dict_of_dicts_of_fields_in_each_section)
df_summary_of_fields_used_in_each_section = df_summary_of_fields_used_in_each_section.reset_index()
df_summary_of_fields_used_in_each_section.rename(columns = {'index' : 'Field Name'}, inplace=True)
df_summary_of_fields_used_in_each_section = df_summary_of_fields_used_in_each_section.replace('y', 'Included') #change from y or n to 'Included' or empty string
df_summary_of_fields_used_in_each_section = df_summary_of_fields_used_in_each_section.replace('n', '')

# ---------------------------------------

"""
Produce a raw DataFrame (df) for each of the 3 objectives in scope for the work/code. 
Each is made up from the original raw data, but using only the columns pertinent to that 
specific objective, as identified in the preceding dictionaries
"""

#call make_df function to create dataframe with fields relevant to Obj 1A (HEA)
obj_1a_df = make_df(df_with_imd, dict_obj_1a_HEA_UML_fields_to_inc)
#call make_df function to create dataframe with fields relevant to Obj 1B (DNAs)
obj_1b_df = make_df(df_with_imd, dict_obj_1b_DNA_SML_fields_to_inc)
#call make_df function to create dataframe with fields relevant to Obj 2 (emissions)
obj_2_df = make_df(df_with_imd, dict_obj_2_carbon_emissions_fields_to_inc)


# ------------------------------------------------------
# ------------------------------------------------------
#<<< Creating processed data for objective 1 (HEA) >>>
# ------------------------------------------------------
# ------------------------------------------------------

#Creating Processed Data for Objective 1A (HEA)
count_row_before = obj_1a_df.shape[0]

#Create DataFrame for Objective 1A (HEA / profile):
#keep="last" as df has already been sorted ascending in earlier code 
#Assumption is because specific demographics can change, the latest known contact should be the most accurate to use, hence retain the latest attendance only
#obj_1a_df = obj_1a_df.sort_values(appt_date_time, ascending=True) #22/6

# --------------------------------------------------------
# new code block to fix error in above sort added 23/6
#list old dates to check for validation
list_old_dates = list(obj_1a_df.appt_date_time)

#convert to date time and populate a list
list_new_dates = [pd.to_datetime(row.appt_date_time, format="%d/%m/%Y %H:%M") for row in obj_1a_df.itertuples()]

#add list of datetime objects as new column / series to dataframe
obj_1a_df.insert(6, 'DateEvent_datetime', list_new_dates)

#delete the original date time field 
obj_1a_df = obj_1a_df.drop(appt_date_time, axis=1)

#rename new date time field to the same name as the old one that was just dropped
obj_1a_df.rename(columns={'DateEvent_datetime': appt_date_time}, inplace=True)

#sort the df based on the date time in ascending order
obj_1a_df = obj_1a_df.sort_values(appt_date_time, ascending=True) #current line 1336

#filter to in-area only
obj_1a_df = obj_1a_df[obj_1a_df[in_out_area_classifier] == 'in-area']
# --------------------------------------------------------

obj_1a_df_unique_pts = obj_1a_df.drop_duplicates(subset=patient_id.lower(), keep="last") 
count_row_after = obj_1a_df_unique_pts.shape[0]
#obj_1a_df_unique_pts.shape

#Test print statements to show change in size of the dataset once duplicates are removed
print(f"Total rows with duplicate patients: {count_row_before}")
print(f"\nTotal rows with unique patients only: {count_row_after}")

# ---------------------------------------

#identify data fields present in each newly created subset df
obj_1a_data_fields = list(obj_1a_df.columns)
obj_1b_data_fields = list(obj_1b_df.columns)
obj_2_data_fields = list(obj_2_df.columns)

# ---------------------------------------

#new code 19/09 remove out of area patients from this objectives' data frame - now being handled in above new code block
#mask = obj_1a_df_unique_pts[in_out_area_classifier] == 'in-area' #in-area required in real data set
#obj_1a_df_unique_pts = obj_1a_df_unique_pts[mask]

# ---------------------------------------

#Data quality description section
list_names, present_values, missing_values, missing_pct, data_quality_summary_df = data_quality_summary(obj_1a_df_unique_pts, obj_1a_data_fields)

#Test print statement for above function - can be used in final report
#print(data_quality_summary_df)
file_number = number_saved_file_preprocessing(df_counter)
df_counter+=1

data_quality_summary_df.to_excel(f"{preprocessing_assets_path}/df{file_number}_Data_Quality_Summary_Df_HEA.xlsx")

# ---------------------------------------

#loop to check for any columsn in obj_1a_df_unique_pts for which all rows are blank (so whole col is blank) AND where that column SHOULD be a string, if so, replace all cells with the word 'missing'
for col in obj_1a_df_unique_pts.columns:
    if obj_1a_df_unique_pts[col].isnull().sum() == obj_1a_df_unique_pts.shape[0]:
        if dict_obj1a_cols_to_dtype[col] == str:
            obj_1a_df_unique_pts[col] = obj_1a_df_unique_pts[col].fillna('missing')

# ---------------------------------------

#new code block 19/09: repeat data quality summmary function from above. The above call to this function produces the true DQ summary for the fields in scope. We're now calling it again because we have since controlled for IMD decile and lsoa
#by removing the rows will NaNs in those fields. As a result, at this stage the obj_1a_df_unique_pts df has 100% present values in those fields, but they are currently still included in the list of missing fields
#we call the data_quality_summary function again here to update the variables it produces to reflect this 'manual' adjustment.

#Data quality description section
list_names, present_values, missing_values, missing_pct, data_quality_summary_df = data_quality_summary(obj_1a_df_unique_pts, obj_1a_data_fields)

# ---------------------------------------

#Create dictionary of field names (keys) to count of missing values (values)
dict_fieldname_missing_values, missing_values_fields = missing_values_field_names(obj_1a_data_fields, missing_values)

# ---------------------------------------

#Call the function to create list of field names with missing values of 
#int/float type and string type respectively
int_float_fields, str_fields, datetime_fields, data_type_dict = group_fields_by_data_type(obj_1a_df_unique_pts, obj_1a_data_fields, dict_fieldname_missing_values)

# ---------------------------------------
#List to populate with the fields the user decides to remove from / exclude from median (or other) missing values imputing
int_float_fields_keep = [] #fields to impute missing values for
int_float_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
str_fields_keep = [] #fields to impute missing values for
str_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
datetime_fields_keep = []
datetime_fields_drop = []

# ---------------------------------------

#CHECK REQUIRED - NEEDS TO CREATE NULLS / MISSING DATETIMES IN THE DUMMY 
#DATA SET TO CHECK THIS CELL FUNCTIONS AS EXPECTED - STILL TO DO..!

#allow user to decide whether to control for missing appt date times or exclude rows where this is empty
#Call function for user to confirm integer fields to retain - where there are integer fields with missing values only
if len(datetime_fields) > 0:
    identify_fields_to_impute(datetime_fields, "datetime")

if len(int_float_fields) > 0:
    identify_fields_to_impute(int_float_fields, "int")
    
if len(str_fields) > 0:
    identify_fields_to_impute(str_fields, "str")

# ---------------------------------------
#zxcv - new code 10/9 AM - these should be populated with the content of the above lists. 
#they are created here as the code for keep / drop lists is repeated for subsequent stages, meaning the lists are overwritten each time
#so, we need a record of each stage's decision - hence the copy made here assigned to new variable names
hea_for_report_int_float_fields_keep = int_float_fields_keep
hea_for_report_int_float_fields_drop = int_float_fields_drop
hea_for_report_str_fields_keep = str_fields_keep
hea_for_report_str_fields_drop = str_fields_drop
hea_for_report_datetime_fields_keep = datetime_fields_keep
hea_for_report_datetime_fields_drop  = datetime_fields_drop

# ---------------------------------------

#Global dicts for use when impute_missing function is called
dict_ints_imputed = {}
dict_ints_imputed_missing = {}
dict_str_imputed = {}
dict_str_imputed_missing = {}

# ---------------------------------------

#Call the function to impute missing int/float fields using the median
impute_missing(obj_1a_df_unique_pts, int_float_fields_keep, "int")

#Call the function to impute missing string fields with the word "missing"
impute_missing(obj_1a_df_unique_pts, str_fields_keep, "string")

"""
DECISION NEEDED: do we call impute_missing function for missing appointment date times
or remove entirely or assume always present / handled outside the code???
currently assumes all present / handled outside the code. Further development area.
"""

# ---------------------------------------

#Create a final df consisting of all the imputed fields, the fields for which 
#no imputing needed, and excluding all dropped fields
final_processed_df = pd.DataFrame()

data_fields_copy = obj_1a_data_fields.copy()

for field in obj_1a_data_fields:
    if field in int_float_fields_drop:
        data_fields_copy.remove(field)
    elif field in int_float_fields_keep:
        data_fields_copy.remove(field)
    elif field in str_fields_drop:
        data_fields_copy.remove(field)
    elif field in str_fields_keep:
        data_fields_copy.remove(field)

for field in data_fields_copy:
    final_processed_df[field] = obj_1a_df_unique_pts[field]

# ---------------------------------------

#For all fields with missing data that the user chose to keep/impute values for
#work through each included field, and where the missing row count for that field
#is >0, add a new series to the df for the series including imputed values
#and a boolean series indicating which was an imputed value

#The final processed df:
#1. includes fields that originally had no missing data
#2. excludes fields that had missing data, but the user decided to drop / not impute
#3. includes fields that originally had missing data, and the user imputed data for 
# missing values

if len(int_float_fields_keep) > 0:
    for field in int_float_fields_keep:
        #final_processed_df[field] = (obj_1a_df_unique_pts[field]) #Exclue original field
        #  with nulls
        final_processed_df[field+'_with_median'] = dict_ints_imputed[field]
        final_processed_df[field+'_Null_replaced'] = dict_ints_imputed_missing[field]
        
        #if field in list(final_processed_df.columns.values):
        #    final_processed_df = final_processed_df.drop(field, 1)
        
if len(str_fields_keep) > 0:
    for field in str_fields_keep:
        #final_processed_df[field] = (obj_1a_df_unique_pts[field]) #Exclue original field 
        # with nulls
        final_processed_df[field+'_with_missing'] = dict_str_imputed[field]
        final_processed_df[field+'_Null_replaced'] = dict_str_imputed_missing[field]
        
        #if field in list(final_processed_df.columns.values):
        #    final_processed_df = final_processed_df.drop(field, 1)

# ---------------------------------------

#call function to remove bool fields from the DataFrame
final_processed_df_bools_removed = remove_booleans_from_df(final_processed_df)

# ---------------------------------------

final_processed_df_bools_removed_standard_col_names = final_processed_df_bools_removed.copy()
#check through the dict of imputed string fields and replaced col name back to original name
for col in dict_str_imputed_missing.keys():
    if col+"_with_missing" in final_processed_df_bools_removed_standard_col_names.columns:
        final_processed_df_bools_removed_standard_col_names.rename(columns={col+"_with_missing" : col}, inplace=True)

# ---------------------------------------

#check through the dict of imputed int fields and replaced col name back to original name
for col in dict_ints_imputed_missing.keys():
    if col+"_with_median" in final_processed_df_bools_removed_standard_col_names.columns:
        final_processed_df_bools_removed_standard_col_names.rename(columns={col+"_with_median" : col}, inplace=True)

# ---------------------------------------

"""
1) One for traditional (non-ML) HEA Objective 1a: DONE
    unique pts (most recent appt), 
    all appt statuses, 
    inc demographics but NOT one hot encoded (save to file before encoding)
    in area only
"""

os.makedirs('processed_data/1A_HEA', exist_ok=True)  
final_processed_df_bools_removed.to_csv('processed_data/1A_HEA/processed_data_1A_HEA_not_encoded.csv', index=False)
print("File saved.")
final_processed_df_bools_removed_standard_col_names.to_csv('processed_data/1A_HEA/processed_data_1A_HEA_not_encoded_standard_col_names.csv', index=False)
print("File saved.")

# ---------------------------------------

#create and populate dictionary of filtered cross tabs, for each attendance type (condition) 
# to produce df per condition based on the HEA (unique patients) df, which DOES include age, 
# and will save those dfs as temp_dfs into the relevant processed folder. 
#Then, the code will read those dfs into the HEA file when that runs, to enable subsetting 
# the dfs based on condition age range (Which isn't yet known in this processing file, 
# so can't be done here) and then, finally, re use the cross tab for loop below in the 
# HEA file, to produce the required crosstabs for unmet need per condition. 

dict_df_condition_unique_pts_expressed_need = {}
dict_crosstab_condition_unique_pts_expressed_need = {} #not used in this code file, to be copied/pasted/repurposed in the HEA code file once dfs created here, called into HEA file, are filtered by age.
dict_condition_unique_pts_expressed_need_filenames = {}

attend_reason_list = list(set(final_processed_df_bools_removed[attend_reason]))

for reason in attend_reason_list:
    mask = final_processed_df_bools_removed[attend_reason] == reason
    final_processed_df_bools_removed_mask = final_processed_df_bools_removed[mask]
    
    #Add the filtered df to the dictionary created above
    dict_df_condition_unique_pts_expressed_need[reason] = final_processed_df_bools_removed_mask


#start
for reason in attend_reason_list:
    #check through the dict of imputed string fields and replaced col name back to original name
    for col in dict_str_imputed_missing.keys():
        if col+"_with_missing" in dict_df_condition_unique_pts_expressed_need[reason].columns:
            dict_df_condition_unique_pts_expressed_need[reason].rename(columns={col+"_with_missing" : col}, inplace=True)

# ---------------------------------------

#check through the dict of imputed int fields and replaced col name back to original name
    for col in dict_ints_imputed_missing.keys():
        if col+"_with_median" in dict_df_condition_unique_pts_expressed_need[reason].columns:
            dict_df_condition_unique_pts_expressed_need[reason].rename(columns={col+"_with_median" : col}, inplace=True)
#end


#save all dfs created for later unmet need use. In the HEA file, these will be converted 
os.makedirs('processed_data/1A_HEA', exist_ok=True) 

for reason in attend_reason_list:
    dict_df_condition_unique_pts_expressed_need[reason].to_csv(f'processed_data/1A_HEA/tempdf_unique_patients_{reason}.csv')
    print(f"{reason} dataframe file saved.")

# ---------------------------------------

#new code 18/9 to locate the text for the lsoa field and control for scenario of lsoa field having missing values and therefore having the _with_missing suffix
if lsoa_residence in final_processed_df_bools_removed.columns:
    lsoa_field_name = lsoa_residence
else:
    lsoa_field_name = lsoa_residence+"_with_missing"

#new code 18/9 to locate the text for the lsoa field and control for scenario of lsoa field having missing values and therefore having the _with_missing suffix
if in_out_area_classifier in final_processed_df_bools_removed.columns:
    in_out_area_classifier = in_out_area_classifier
else:
    in_out_area_classifier = in_out_area_classifier+"_with_missing"

# ---------------------------------------


#create cross tabs of the above created dfs as temp files too. 
for reason in attend_reason_list:
    #Create crosstab and add to the dict created above
    #df_crosstab = df_copy_mask.groupby([lsoa_residence, clinic_name])[clinic_name].count().unstack().fillna(0)
    df_hea_reason_crosstab = pd.crosstab(dict_df_condition_unique_pts_expressed_need[reason][lsoa_field_name], dict_df_condition_unique_pts_expressed_need[reason][in_out_area_classifier], margins=True)
    dict_crosstab_condition_unique_pts_expressed_need[reason] = df_hea_reason_crosstab


#os.makedirs('processed_data/2_Carbon_Emissions', exist_ok=True)  
#for reason in attend_reasons:
#    dict_condition_clinic_crosstabs[reason].to_csv(f'processed_data/2_Carbon_Emissions/crosstab_attendances_lsoa_to_clinic_{reason}.csv')  
#    print("File saved.")

# ---------------------------------------

#call function to create list of fields to omit from imputing function and 
#list of fields to inc in impute function

fields_to_not_encode, fields_to_encode, fields_to_remove = fields_to_encode_or_drop(final_processed_df_bools_removed)

# ---------------------------------------
#THIS IS SKIPPED AT PRESENT AS IT IS NOT USED CURRENTLY IN THE PROGRAM.
#THIS COULD BE UTILISED IN FUTURE IF UNSUPERVISED LEARNING MODELS ARE REQUIRED / DEVELOPED
#call the encoding_dataframe function
#encoded_dataframe = encoding_dataframe(final_processed_df_bools_removed, fields_to_encode, fields_to_not_encode)

# ---------------------------------------

#All done, save pre-processed file
"""
2) One for unsupervised HEA Objective 1a: DONE
    unique pts (most recent appt), 
    all appt statuses retained, 
    inc demographics, 
    and one hot encoded
    in area only
"""
#AS ABOVE - THIS ENCODED HEA FILE IS NOT MADE CURRENTLY DUE TO THE HASHED OUT CODE ABOVE
#IF UNSUPERVISED ML MODELS ARE DEVELOPED OR NEEDED FOR HEA, JUST UNCOMMENT THE ABOVE CODE 
#AND THE LINES IMMEDIATELY BELOW TO SAVE THIS CREATED ENCODED FILE.

#os.makedirs('processed_data/1A_HEA', exist_ok=True)  
#encoded_dataframe.to_csv('processed_data/1A_HEA/processed_data_1A_HEA_encoded_for_ML.csv', index=False)  
#print("File saved.")

# ---------------------------------------

#pause after the first section has completed

print()
print("--------------------------------------------")
print("<<< Processing for objective 1a complete >>>")
print("--------------------------------------------")

press_any_key_to_continue()


# -------------------------------------------------------------------
# -------------------------------------------------------------------
#<<<     Creating processed data for objective 1b: DNA profiling >>>
# -------------------------------------------------------------------
# -------------------------------------------------------------------

#removing duplicates based on user-instructed appt_id for objective 1B
#keep="last" as df earlier sorted ascending - consistent with approach to 1A
#appt ID (dup patients fine to inc in obj 2) but have all rows removed that 
#have an attendance status that isnt attended or DNA

"""
dev note for renaming variables in code: remove if all working!!
original_df RENAMED AS obj_1b_df !!
df_with_imd_unique_appts_1b RENAMED AS obj_1b_df_unique_appts !!
df_with_imd_unique_appts_1b_dna_attend RENAMED AS obj_1b_df_unique_appts_dna_attend
"""

obj_1b_df_unique_appts = obj_1b_df.drop_duplicates(subset=appt_id, keep="last")

mask = obj_1b_df_unique_appts[appt_type] == 'Booked' #test new code 13/09 zxcv
obj_1b_df_unique_appts = obj_1b_df_unique_appts[mask] #test new code 13/09
obj_1b_df_unique_appts.drop(appt_type, axis=1, inplace=True) #Once confirmed the above filter to booked only is working, remove appt type field

#end test new code
count_row_after_unique_appt_id = obj_1b_df_unique_appts.shape[0]

count_row_before = df_with_imd.shape[0]

print(f"Total rows with duplicate appt IDs: {count_row_before}")
print(f"\nTotal rows with unique appt IDs only: {count_row_after_unique_appt_id}")

# ---------------------------------------

#locate any column consisting of ONLY missing Null (NaN) values by checking the sum of thoes columns is equal to the size of the df. 
#where this is true, then check the data type expected in that cell using the reference dict
#if this is str, replace all rows of that column with the word 'missing'
for col in obj_1b_df_unique_appts.columns:
    if obj_1b_df_unique_appts[col].isnull().sum() == obj_1b_df_unique_appts.shape[0]:
        #print(f"{col} : all empty")
        if dict_obj1a_cols_to_dtype[col] == str:
            #print(f"{col} all blank and string type")
            obj_1b_df_unique_appts[col] = obj_1b_df_unique_appts[col].fillna('missing')

# ---------------------------------------

#RELEVANT TO 1B ONLY - DNA PROFILING
#once tested and working for 1B processed data set production, remove the 
#doc stringed out function in function section
appt_status_to_drop, appt_status_to_keep = remove_inappropriate_appt_statuses(obj_1b_df_unique_appts, attend_status, "attendance status")  #field_list, df, field_name

#Test print statements - delete when final
#print(appt_status_to_drop) 
#print(appt_status_to_keep)

# ---------------------------------------

"""
DECISION NEEDED: do we also dropna's from the attendance status column at this point
(see "DECISION NEEDED" line in code block below). Code done to do this (hashed out below)
but would just need validating / UAT if introduced.
"""

#RELEVANT TO 1B ONLY - DNA PROFILING
#Code to remove rows where appointment statuses are those user indicated to drop 
#this is on a test copied df for now. once confirmed working, repoint to 
#df_with_imd_unique_appts
#Delete Rows by Checking Conditions: any row in attend_status that is equal 
#to values in appt_status_to_drop to be removed from the df

obj_1b_df_unique_appts_dna_attend = obj_1b_df_unique_appts.copy()

for status in appt_status_to_drop:
    indexNames = obj_1b_df_unique_appts_dna_attend[obj_1b_df_unique_appts_dna_attend[attend_status] == status ].index
    obj_1b_df_unique_appts_dna_attend.drop(indexNames , inplace=True)

print(f"Original number of unique appointments: {obj_1b_df_unique_appts.shape[0]}")
print(f"Number of unique appointments with a status of attended or DNA inc Nulls: {obj_1b_df_unique_appts_dna_attend.shape[0]}")


# ---------------------------------------

obj_1b_df_unique_appts_dna_attend = obj_1b_df_unique_appts_dna_attend.dropna(subset=[attend_status])
print(f"Number of unique appointments with a status of attended or DNA exc. Nulls: {obj_1b_df_unique_appts_dna_attend.shape[0]}")

# ---------------------------------------
obj_1b_data_fields = list(obj_1b_df_unique_appts_dna_attend.columns)

list_names_1b, present_values_1b, missing_values_1b, missing_pct_1b, data_quality_summary_df_1b = data_quality_summary(obj_1b_df_unique_appts_dna_attend, obj_1b_data_fields)

#-----------------------------------------------------

#create dictionary of field names (keys) to missing value counts (values)
dict_fieldname_missing_values_1b, missing_values_fields_1b = missing_values_field_names(obj_1b_data_fields, missing_values_1b)

#-----------------------------------------------------

#Call the function to create list of field names with missing values of int/float type and string type respectively
int_float_fields_1b, str_fields_1b, datetime_fields_1b, data_type_dict_1b = group_fields_by_data_type(obj_1b_df_unique_appts_dna_attend, obj_1b_data_fields, dict_fieldname_missing_values_1b)

file_number = number_saved_file_preprocessing(df_counter)
df_counter+=1

data_quality_summary_df_1b.to_excel(f"{preprocessing_assets_path}/df{file_number}_Data_Quality_Summary_Df_DNA.xlsx")

# ---------------------------------------

#List to populate with the fields the user decides to remove from / exclude from median (or other) missing values imputing
int_float_fields_keep = [] #fields to impute missing values for
int_float_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
str_fields_keep = [] #fields to impute missing values for
str_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
datetime_fields_keep = [] #fields to impute missing values for
datetime_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm

# ---------------------------------------

"""
CHECK REQUIRED - NEEDS TO CREATE NULLS / MISSING DATETIMES IN THE DUMMY DATA SET TO CHECK 
THIS CELL FUNCTIONS AS EXPECTED - STILL TO DO..!
"""

#allow user to decide whether to control for missing appt date times or 
#exclude rows where this is empty
if len(datetime_fields_1b) > 0:
    identify_fields_to_impute(datetime_fields_1b, "datetime")

#test print only
print(datetime_fields_1b)

# ---------------------------------------

#Call function for user to confirm integer fields to retain - where there are 
#integer fields with missing values only
if len(int_float_fields_1b) > 0:
    identify_fields_to_impute(int_float_fields_1b, "int")

# ---------------------------------------

#Call function for user to confirm string fields to retain - where there are 
#string fields with missing values only
if len(str_fields_1b) > 0:
    identify_fields_to_impute(str_fields_1b, "str")

#---------------------------------------
#zxcv - new code 10/9 AM - these should be populated with the content of the above lists. 
#they are created here as the code for keep / drop lists is repeated for subsequent stages, meaning the lists are overwritten each time
#so, we need a record of each stage's decision - hence the copy made here assigned to new variable names
dna_for_report_int_float_fields_keep = int_float_fields_keep
dna_for_report_int_float_fields_drop = int_float_fields_drop
dna_for_report_str_fields_keep = str_fields_keep
dna_for_report_str_fields_drop = str_fields_drop
dna_for_report_datetime_fields_keep = datetime_fields_keep
dna_for_report_datetime_fields_drop  = datetime_fields_drop

# ---------------------------------------

#Global dicts for use when impute_missing function is called
dict_ints_imputed_1b = {}
dict_ints_imputed_missing_1b = {}
dict_str_imputed_1b = {}
dict_str_imputed_missing_1b = {}

# ---------------------------------------

#Call function to impute missing values using median for int/float
impute_missing_1b(obj_1b_df_unique_appts_dna_attend, int_float_fields_keep, "int")

# ---------------------------------------
"""
DECISION NEEDED: Do we also imput missing appointment date/times or remove entirely?
"""
#Call function to impute missing vaues for string fields using text "missing"
impute_missing_1b(obj_1b_df_unique_appts_dna_attend, str_fields_keep, "str")

# ---------------------------------------

#Create a final df consisting of all the imputed fields, the fields for 
#which no imputing needed, and excluding all dropped fields

final_processed_df_1b = pd.DataFrame()

data_fields_temp = obj_1b_data_fields.copy()
data_fields_copy = obj_1b_data_fields.copy()


for field in data_fields_copy:
    if field in int_float_fields_drop:
        data_fields_temp.remove(field)
    elif field in int_float_fields_keep:
        data_fields_temp.remove(field)
    elif field in str_fields_drop:
        data_fields_temp.remove(field)
    elif field in str_fields_keep:
        data_fields_temp.remove(field)

for field in data_fields_temp:
    final_processed_df_1b[field] = obj_1b_df_unique_appts_dna_attend[field]

# ---------------------------------------

# ---------------------------------------

#For all fields with missing data that the user chose to keep/impute values for
#work through each included field, and where the missing row count for that field
#is >0, add a new series to the df for the series including imputed values
#and a boolean series indicating which was an imputed value

#The final processed df:
#1. includes fields that originally had no missing data
#2. excludes fields that had missing data, but the user decided to drop / not impute
#3. includes fields that originally had missing data, and the user imputed data for missing values

if len(int_float_fields_keep) > 0:
    for field in int_float_fields_keep:
        #final_processed_df[field] = (obj_1a_df_unique_pts[field]) #Exclue original field with nulls
        final_processed_df_1b[field+'_with_median'] = dict_ints_imputed_1b[field]
        final_processed_df_1b[field+'_Null_replaced'] = dict_ints_imputed_missing_1b[field]
        
        #Couldn't locate why gender field remained, but other string replaced fields removed. 
        #This for loop looks to solve the issue - for each field user says to keep, bring back the imputed values
        #Then, if that field is still in the df (for reasons not yet identified!) remove it
        #and only keep the new field with imputed values for blanks
        
        if field in list(final_processed_df_1b.columns.values):
            final_processed_df_1b = final_processed_df_1b.drop(field, 1)

if len(str_fields_keep) > 0:
    for field in str_fields_keep:
        #final_processed_df[field] = (obj_1a_df_unique_pts[field]) #Exclue original field with nulls
        final_processed_df_1b[field+'_with_missing'] = dict_str_imputed_1b[field]
        final_processed_df_1b[field+'_Null_replaced'] = dict_str_imputed_missing_1b[field]
        
        if field in list(final_processed_df_1b.columns.values):
            final_processed_df_1b = final_processed_df_1b.drop(field, 1)


# ---------------------------------------

#Call function to remove bool series from df to encode
final_processed_df_bools_removed_1b = remove_booleans_from_df(final_processed_df_1b)

"""
#Need to think of approach to make this generalisable, as attend statuses may vary
final_df_attended_only_obj_2 = final_processed_df_bools_removed_1b.loc[final_processed_df_bools_removed_1b['attend_status_with_missing']=='Attended']

os.makedirs('processed_data', exist_ok=True)  
final_df_attended_only_obj_2.to_csv('processed_data/processed_data_2_carbon_emissions.csv', index=False)  
print("File saved.")
#encoded_dataframe.to_csv('./data/processed_data.csv', index=False)
"""

#Test print - delete 
#final_processed_df_bools_removed_1b.loc[[69, 68, 42, 29, 67, 81]]


# ---------------------------------------

#call function to create list of fields to omit from imputing function and 
#list of fields to inc in impute function
fields_to_not_encode_1b, fields_to_encode_1b, fields_to_remove_1b = fields_to_encode_or_drop(final_processed_df_bools_removed_1b)

# ---------------------------------------

#new code 13/09 - added to filter out walk-in appointments and retain only those that were Booked
#mask = final_processed_df_bools_removed_1b[appt_type] == 'Booked'
#final_processed_df_bools_removed_1b = final_processed_df_bools_removed_1b[mask]
#count_row_after_unique_appt_id_booked =  final_processed_df_bools_removed_1b.shape[0]
#print(f"\nTotal rows booked appointments only: {count_row_after_unique_appt_id_booked}")
#new code end
# ---------------------------------------

#call the encoding_dataframe function
encoded_dataframe_1b = encoding_dataframe(final_processed_df_bools_removed_1b, fields_to_encode_1b, fields_to_not_encode_1b)

# ---------------------------------------

#dev note:
#hard-coded (ish) approach to remove DNA and missing from one hot encoded fields
#not ideal, as not generalisable, needs function to scale to any data set
#should now be solved with standardised mapping of col names to actuals in the df
if f"encoded_"+attend_status+"_with_missing_DNA" in list(encoded_dataframe_1b.columns):
    encoded_dataframe_1b = encoded_dataframe_1b.drop("encoded_"+attend_status+"_with_missing_DNA", 1)
if f"encoded_"+attend_status+"_with_missing_missing" in list(encoded_dataframe_1b.columns):
    encoded_dataframe_1b = encoded_dataframe_1b.drop("encoded_"+attend_status+"_with_missing_missing", 1)

# ---------------------------------------

#Save the file
"""
3) One for supervised DNA Objective 1b: DONE
    unique appts (NOT patients), 
    inc demographics, 
    one hot encoded, 
    only attend/DNA appt status (any other status removed)
    in-area only
"""

os.makedirs('processed_data/1B_DNA_profile', exist_ok=True)  
encoded_dataframe_1b.to_csv('processed_data/1B_DNA_profile/processed_data_1B_DNA_Profile_new_test.csv', index=False)  
print("File saved.")
#encoded_dataframe.to_csv('./data/processed_data.csv', index=False)

# ---------------------------------------

#pause after the section has completed

print()
print("--------------------------------------------")
print("<<< Processing for objective 1b complete >>>")
print("--------------------------------------------")

press_any_key_to_continue()

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
#<<< Create processed data for objective 2: geomodelling and carbon emissions >>>
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
#Plan:
#appt_id : use to retain unique appts only, then remove
#attend_status : use to retain attended appts only, then remove
#appt_type : use to retain face to face appts only, then remove
#in_out_area_classifier : retain in area only, then remove

obj_2_df_unique_appts = obj_2_df.drop_duplicates(subset=appt_id, keep="last")
obj_2_df_unique_appts_attended = obj_2_df_unique_appts.loc[obj_2_df_unique_appts[attend_status] == "Attended"]
#obj_2_df_unique_appts_attended_f2f = obj_2_df_unique_appts_attended.loc[obj_2_df_unique_appts[appt_type] == "Booked"]


# ---------------------------------------

#locate any column consisting of ONLY missing Null (NaN) values by checking the sum of thoes columns is equal to the size of the df. 
#where this is true, then check the data type expexted in that cell using the reference dict
#if this is str, replace all rows of that column with the word 'missing'
for col in obj_2_df_unique_appts_attended.columns:
    if obj_2_df_unique_appts_attended[col].isnull().sum() == obj_2_df_unique_appts_attended.shape[0]:
        #print(f"{col} : all empty")
        if dict_obj1a_cols_to_dtype[col] == str:
            #print(f"{col} all blank and string type")
            obj_2_df_unique_appts_attended[col] = obj_2_df_unique_appts_attended[col].fillna('missing')

# ---------------------------------------

obj_2_df_unique_appts_attended_inarea = obj_2_df_unique_appts_attended.loc[obj_2_df_unique_appts_attended[in_out_area_classifier] == "in-area"] #fine
obj_2_df_unique_appts_attended_inarea_f2f = obj_2_df_unique_appts_attended_inarea.loc[obj_2_df_unique_appts_attended_inarea[clinic_format] == "Clinic Face to Face"] #fine

#drop all fields no longer required as filtered onto unique appointment ID's, in-area, attended appointments only
obj_2_df_unique_appts_attended_inarea_f2f.drop([appt_id, attend_status, in_out_area_classifier, clinic_format], axis=1, inplace=True)

#create list of remaining fields
obj_2_data_fields = list(obj_2_df_unique_appts_attended_inarea_f2f.columns)

count_row_after_unique_appt_attend_booked_in_area = obj_2_df_unique_appts_attended_inarea_f2f.shape[0]
count_row_before = df_with_imd.shape[0]

print(f"Total rows with duplicate appt IDs: {count_row_before}")
print(f"\nTotal rows with unique appt IDs only: {count_row_after_unique_appt_attend_booked_in_area}")

# ---------------------------------------

#create variables to quantify volume of missing data, by each field included in the df
list_names_2, present_values_2, missing_values_2, missing_pct_2, data_quality_summary_df_2 = data_quality_summary(obj_2_df_unique_appts_attended_inarea_f2f, obj_2_data_fields)

# ---------------------------------------

#create dictionary of field names (keys) to missing value counts (values)
dict_fieldname_missing_values_2, missing_values_fields_2 = missing_values_field_names(obj_2_data_fields, missing_values_2)

# ---------------------------------------

#Call the function to create list of field names with missing values of int/float type and string type respectively
int_float_fields_2, str_fields_2, datetime_fields_2, data_type_dict_2 = group_fields_by_data_type(obj_2_df_unique_appts_attended_inarea_f2f, obj_2_data_fields, dict_fieldname_missing_values_2)

# ---------------------------------------

#new cell 17/8/22
file_number = number_saved_file_preprocessing(df_counter)
df_counter+=1

#save file
data_quality_summary_df_2.to_excel(f"{preprocessing_assets_path}/df{file_number}_Data_Quality_Summary_Df_Geomodelling_Emissions.xlsx")

# ---------------------------------------

#This code is new as of 18/3 and intended to create the required summary dfs for obj 2
obj_2_df_unique_appts_attended_f2f_inarea = obj_2_df_unique_appts_attended_inarea_f2f.dropna()
df_copy = obj_2_df_unique_appts_attended_f2f_inarea.copy()

# ---------------------------------------

#Create list of unique possible attendance reasons in the data set
attend_reasons = list(set(obj_2_df_unique_appts_attended_f2f_inarea[attend_reason]))

# ---------------------------------------
#this variable called van_activity is so called because the dataset used in the pilot 
#around which this code was written involved consideration of where to position a mobile clinic
#in a van for a hypothetical new clinic location. But, clearly, it doesn't need to be a mobile
#clinic, it could just as well be a bricks and mortar clinic building. 
#So, when reviewing this code, consider 'activity in van' as 'activity that can be seen at the 
#potential new location'.

#Create list of unique possible "activity in the van" field options in the data set
van_activity = list(set(obj_2_df_unique_appts_attended_f2f_inarea[activity_in_van]))

# ---------------------------------------

#create and populate dictionary of filtered cross tabs
dict_of_frequency_counts = {}
for activity_type in van_activity:
    for reason  in attend_reasons:
        mask = ((
            obj_2_df_unique_appts_attended_f2f_inarea[attend_reason] == reason) & (
            obj_2_df_unique_appts_attended_f2f_inarea[activity_in_van] == activity_type
             ))
                
        df_copy_mask = df_copy[mask]
        #df_crosstab = df_copy_mask.groupby([lsoa_residence, clinic_name])[clinic_name].count().unstack().fillna(0)
        df_crosstab = pd.crosstab(df_copy_mask[lsoa_field_name], df_copy_mask[clinic_name], margins=True)
        dict_of_frequency_counts[f"crosstab_attendances_lsoa_to_clinic_van_{activity_type}-{reason}"] = df_crosstab

# ---------------------------------------

"""
TO DO:
Need to add code in here (or within the nested for loop above, after creating the crosstab
and before assigning that to the dictionary, to ensure all matrixes include all LSOAs with 0 
values where we haven't had anyone from there yet.
Alternatively, do this in SB's code file and / or the distance calc file.
Done.
"""

# ---------------------------------------

#Save all created crosstabs - based on activity can be done in the van (Yes/No) and Attendance Type (GU / Con / Both / Neither))
"""
separate output files for carbon emissions, each a cross tab for each combination of in/not in van and attendance type (con / GU / both), drop appt type of neither:
"""

for key in dict_of_frequency_counts.keys():
    os.makedirs('processed_data/2_Carbon_Emissions', exist_ok=True)
    dict_of_frequency_counts[key].to_csv("processed_data/2_Carbon_Emissions/" + key + ".csv", index=True)  
    print("File saved.")

# ---------------------------------------

#List to populate with the fields the user decides to remove from / exclude from median (or other) missing values imputing
int_float_fields_keep = [] #fields to impute missing values for
int_float_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
str_fields_keep = [] #fields to impute missing values for
str_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm
datetime_fields_keep = [] #fields to impute missing values for
datetime_fields_drop = [] #fields to exclude from imputing process - so either drop rows with any Nulls in these fields or exclude in ML algorithm

# ---------------------------------------

"""
CHECK REQUIRED - NEEDS TO CREATE NULLS / MISSING DATETIMES IN THE DUMMY DATA SET TO CHECK 
THIS CELL FUNCTIONS AS EXPECTED - STILL TO DO..!
"""

#allow user to decide whether to control for missing appt date times or 
#exclude rows where this is empty
if len(datetime_fields_2) > 0:
    identify_fields_to_impute(datetime_fields_2, "datetime")

# ---------------------------------------

#Call function for user to confirm integer fields to retain - where there are 
#integer fields with missing values only
if len(int_float_fields_2) > 0:
    identify_fields_to_impute(int_float_fields_2, "int")

# ---------------------------------------

#-----------------------------------------------------   
#Call function for user to confirm string fields to retain - where there are 
#string fields with missing values only
if len(str_fields_2) > 0:
    identify_fields_to_impute(str_fields_2, "str")


#---------------------------------------
#zxcv - new code 10/9 AM - these should be populated with the content of the above lists. 
#they are created here as the code for keep / drop lists is repeated for subsequent stages, meaning the lists are overwritten each time
#so, we need a record of each stage's decision - hence the copy made here assigned to new variable names
emissions_for_report_int_float_fields_keep = int_float_fields_keep
emissions_for_report_int_float_fields_drop = int_float_fields_drop
emissions_for_report_str_fields_keep = str_fields_keep
emissions_for_report_str_fields_drop = str_fields_drop
emissions_for_report_datetime_fields_keep = datetime_fields_keep
emissions_for_report_datetime_fields_drop  = datetime_fields_drop

#-----------------------------------------------------
#13/6/23 revision - to control for when the source data has Nulls in the clinic_name field
if clinic_name in emissions_for_report_str_fields_keep:
    clinic_name = f"{clinic_name}_with_missing"

#-----------------------------------------------------
#Global dicts for use when impute_missing function is called
dict_ints_imputed_1b = {}
dict_ints_imputed_missing_1b = {}
dict_str_imputed_1b = {}
dict_str_imputed_missing_1b = {}

# ---------------------------------------

#Call function to impute missing values using median for int/float
#DEV NOTE: CHECK FUNCTION 1B IS WOKRING FOR OBJECTIVE 2 
impute_missing_1b(obj_2_df_unique_appts_attended_f2f_inarea, int_float_fields_keep, "int")

#Call function to impute missing vaues for string fields using text "missing"
impute_missing_1b(obj_2_df_unique_appts_attended_f2f_inarea, str_fields_keep, "str")

# ---------------------------------------

#Create a final df consisting of all the imputed fields, the fields for 
#which no imputing needed, and excluding all dropped fields

final_processed_df_2 = pd.DataFrame()

data_fields_copy = obj_2_data_fields.copy()

for field in data_fields_copy:
    if field in int_float_fields_drop:
        data_fields_copy.remove(field)
    elif field in int_float_fields_keep:
        data_fields_copy.remove(field)
    elif field in str_fields_drop:
        data_fields_copy.remove(field)
    elif field in str_fields_keep:
        data_fields_copy.remove(field)

for field in data_fields_copy:
    final_processed_df_2[field] = obj_2_df_unique_appts_attended_f2f_inarea[field]

# ---------------------------------------

#For all fields with missing data that the user chose to keep/impute values for
#work through each included field, and where the missing row count for that field
#is >0, add a new series to the df for the series including imputed values
#and a boolean series indicating which was an imputed value

#The final processed df:
#1. includes fields that originally had no missing data
#2. excludes fields that had missing data, but the user decided to drop / not impute
#3. includes fields that originally had missing data, and the user imputed data for missing values

if len(int_float_fields_keep) > 0:
    for field in int_float_fields_keep:
        #final_processed_df[field] = (obj_1a_df_unique_pts[field]) #Exclue original field with nulls
        final_processed_df_2[field+'_with_median'] = dict_ints_imputed_1b[field]
        final_processed_df_2[field+'_Null_replaced'] = dict_ints_imputed_missing_1b[field]
        
        #Couldn't locate why gender field remained, but other string replaced fields removed. 
        #This for loop looks to solve the issue - for each field user says to keep, bring back the imputed values
        #Then, if that field is still in the df (for reasons not yet identified!) remove it
        #and only keep the new field with imputed values for blanks
        
        if field in list(final_processed_df_2.columns.values):
            final_processed_df_2 = final_processed_df_2.drop(field, 1)

if len(str_fields_keep) > 0:
    for field in str_fields_keep:
        #final_processed_df[field] = (obj_1a_df_unique_pts[field]) #Exclue original field with nulls
        final_processed_df_2[field+'_with_missing'] = dict_str_imputed_1b[field]
        final_processed_df_2[field+'_Null_replaced'] = dict_str_imputed_missing_1b[field]
        
        if field in list(final_processed_df_2.columns.values):
            final_processed_df_2 = final_processed_df_2.drop(field, 1)

# ---------------------------------------

#Check if this df has nulls replaced as expected - delete when confirmed
#final_processed_df_1b[final_processed_df_1b['age_Null_replaced'] == True]

# ---------------------------------------

#Call function to remove bool series from df to encode
final_processed_df_bools_removed_2 = remove_booleans_from_df(final_processed_df_2)

df_copy = final_processed_df_bools_removed_2.copy()

#create dictionary to fill with filterered df values
dict_of_dfs={}
#create and populate dictionary of filtered cross tabs
for reason  in attend_reasons:
    mask = final_processed_df_bools_removed_2[attend_reason] == reason
            
    df_copy_mask = df_copy[mask]
    df_copy_mask = df_copy_mask.drop(columns=[attend_reason, activity_in_van], axis=1)    
    dict_of_dfs[f"{reason}"] = df_copy_mask

# ---------------------------------------

#Save all created crosstabs - based on activity can be done in the van (Yes/No) and Attendance Type (GU / Con / Both / Neither))
"""
separate output files to filter the all attended activity df by each attend reason
"""

for key in dict_of_dfs.keys():
    os.makedirs('processed_data/2_Carbon_Emissions', exist_ok=True)
    dict_of_dfs[key].to_csv("processed_data/2_Carbon_Emissions/processed_data_2_carbon_emissions_" + key + ".csv", index=True)  
    print("File saved.")

# ---------------------------------------

#Code to drop activity in van and attend reason columns manually 
final_processed_df_bools_removed_2_dropped_cols = final_processed_df_bools_removed_2.drop(columns=[attend_reason, activity_in_van], axis=1)

#Save the file
""" 
4) One for carbon emission patient travel modelling: DONE
    df of lsoa_residence, clinics attended, imd_residence, made up of:
    unique appts, 
    attended appt only,
    booked and walk-in appts,
    in-area appts only,
"""

os.makedirs('processed_data/2_Carbon_Emissions', exist_ok=True)  
final_processed_df_bools_removed_2_dropped_cols.to_csv('processed_data/2_Carbon_Emissions/processed_data_2_carbon_emissions_total.csv', index=False)  
print("File saved.")

# ---------------------------------------

#create cross tab of all attended activity (similar to pivot table of lsoa_residence and clinic_name)
df_crosstab = pd.crosstab(final_processed_df_bools_removed_2_dropped_cols[lsoa_residence], final_processed_df_bools_removed_2_dropped_cols[clinic_name], margins=True)

#Save the cross tab file for all in-area activity
""" 
5) One for carbon emission patient travel modelling: DONE
    cross tab of lsoa_residence and clinics attended, 
    unique appts, 
    attended appt only,
    booked and walk-in appts,
    in-area appts only,
"""
os.makedirs('processed_data/2_Carbon_Emissions', exist_ok=True)  
df_crosstab.to_csv('processed_data/2_Carbon_Emissions/crosstab_total_activity_all_attend_reasons_lsoa_clinic_location.csv', index=True)  
print("File saved.")

# ----------------------------------------
# ----------------------------------------
#<<<          END OF THE FILE          >>>
# ----------------------------------------
# ----------------------------------------
