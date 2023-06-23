"""
Authors:
Matt Eves, Anya Gopfert, Sally Brown with input from Drew.

Acknowledgements:
Open Route Service for the use of their API.
Drew from CodeWith for his help with the use of the ORS API.

Purpose:
To produce distance matrixes for the road distance between each LSOA and each clinic for the
given service the code is being used for. The units for the calculated distances is MILES.

The file also produces an identically structured df of the count of patients from each LSOA seen
at each clinic. 

Both will later be converted to numpy arrays and used to estimate carbon emissions for patient travel. 

"""

#Import packages


import requests
import json
import os
import pandas as pd
import pyproj
import geopandas as gpd
import openpyxl
import numpy as np

#import bng-latlon library to convert x y to lat long
from bng_latlon import OSGB36toWGS84

import time

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
else:
    print("Existing 'Assets_produced_by_code' directory located.")

# ---------------------------------------

#Read in parameters

#read-in la of interest from user params file
filename = "raw_data/user_and_data_parameters/user_and_data_params.xlsx"
la_of_interest = pd.read_excel(filename, 'HEA_parameters', index_col=None, usecols = "C", header = 3, nrows=0)
la_of_interest = list(la_of_interest)[0]

#read in user's api Key from user params file
apiKey = pd.read_excel(filename, 'Carbon_emissions', index_col=None, usecols = "B", header = 1, nrows=0)
apiKey = list(apiKey)[0]
#apiKey

#set df counter for output data frames
df_counter = 1

# ----------------------------------------
#Define functions
# ----------------------------------------

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

# ----------------------------------------
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

# ----------------------------------------

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

# ----------------------------------------
#Define Classes
# ----------------------------------------

#Create classes for LSOA and Clinic
class Lsoa:
    def __init__(self, lsoa_id, long, lat):
        self.lsoa_id = lsoa_id
        self.long = long
        self.lat = lat

class Clinic:
    def __init__(self, clinic_name, postcode, long, lat):
        self.clinic_name = clinic_name
        self.postcode = postcode
        self.long = long
        self.lat = lat

# ----------------------------------------
# ----------------------------------------
# <<<           Code begins!           >>>
# ----------------------------------------
# ----------------------------------------

"""
Once openpyxl installed and working, and the above code to create la_of_interest from
user params file is working, this code block can be deleted 
"""
#Read-in a .csv file of the clinic locations (addresses) - postcode needs to be an 
# isolated field called "postcode" (all lower case)

#revised 2/9 to inc clinic names in the user params file and read in from there. 
# skiprows ignores the first row and usecols specifies which columns to use. 
# Combined, this ignores the instructions in the excel file.
df_clinics = pd.read_excel(filename, 'Clinic_details', skiprows=1, usecols=[0,1,2]) 

#subset df to just the chosen LA (update with .py's created to use the pre-process LA)
list_la = [la for la in set((df_clinics['LocalAuthority']))]

clinic_postcode_list = []

if len(list_la) > 1:
    #code for the user to select the specific LA of all present LA's in which clinics are based
    dict_las = {}
    counter = 0
    for item in list_la:
        dict_las[counter] = item
        counter+=1

    print("\nThe following Local Authorities have been identified in the list of clinics:")
    for num in range(len(list(dict_las.keys()))):
        print(f"{num}: {dict_las[num]}")
    selected_clinic_la = int(input("Please input the number for the Local Authority's clinics you wish to use in the model >> "))
    
    #subset the df of clinics to just those with an associated LocalAuthority value matching that the user has input above
    df_clinics_selected_LA_only = df_clinics[df_clinics['LocalAuthority'] == dict_las[selected_clinic_la]]

else:
    print(f"The clinics provided are all present in the following Local Authority: {list_la[0]}.")
    df_clinics_selected_LA_only = df_clinics.copy()
    
for idx, row in df_clinics_selected_LA_only.iterrows():
    clinic_postcode_list.append(row['postcode'])
# ---------------------------------------
"""
Create a list of the names of the clinics from the chosen LA
"""
list_of_clinics = list(df_clinics_selected_LA_only.clinic_name)

# ---------------------------------------

"""
Create temp copy of the df. Reset the index and remove the index and LocalAuthority colums. 
Convert this df to a dictionary, where the key is the df index, and the values are nested 
dictionaries, with keys as column headers and values as the value in the column for the 
given row/record.
"""

#create a dictionary consisting of index (keys) and clinic postcodes & postcode(values) - 
# technically, assumes all clinics have unique postcodes (though if this isnt true # this 
# error wont be flagged)

df_temp_copy = df_clinics_selected_LA_only.copy()
df_temp_copy.reset_index(inplace=True)#.drop('index', axis=1)
df_temp_culled = df_temp_copy.drop(columns=['index', 'LocalAuthority']) 
dict_idx_clinicname_pc = df_temp_culled.transpose().to_dict()
#print(dict_idx_clinicname_pc[0]) #test print

# ---------------------------------------

"""
Using the postcode of the selected Local Authority's clinics, access the Open Route 
Service API and identify the centroid lat and long coordinates for each individual postcode, 
to geo-code the clinics by postcode. 
"""

##Call API to geocode the clinic locations based on their postcode
#create a Class for each clinic, and append each to a list (clinics) of Classes
headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
}

#original code in this doc string, rmeove once rest working
"""for postcode in clinic_postcode_list:
    call = requests.get(f"https://api.openrouteservice.org/geocode/search?api_key={apiKey}&text={postcode}&size=1", headers=headers)
    json_object = json.loads(call.text)
    print(type(json_object))
    clinic_postcode_list_new.append(postcode)
    clinic_long_list.append(json_object['features'][0]['geometry']['coordinates'][0])
    clinic_lat_list.append(json_object['features'][0]['geometry']['coordinates'][1])"""

clinics = []
print()
print('Open Route Service API now running to geo-code all clinic sites in scope for this analysis.')
for num in range(len(dict_idx_clinicname_pc.keys())):
    postal_code = dict_idx_clinicname_pc[num]['postcode']
    clinic_name = dict_idx_clinicname_pc[num]['clinic_name']

    call = requests.get(f"https://api.openrouteservice.org/geocode/search?api_key={apiKey}&text={postal_code}&size=1", headers=headers)
    json_object = json.loads(call.text)
    print(type(json_object))
    long, lat = json_object['features'][0]['geometry']['coordinates']
    new_clinic = Clinic(clinic_name, postal_code, long, lat)
    clinics.append(new_clinic)

    print(f'Clinic {num+1} of {len(dict_idx_clinicname_pc.keys())} geocoding complete. Forcing 2 second delay for geocode API to run')
    time.sleep(2)

clinic_coords = [[l.long, l.lat] for l in clinics]
clinics_length = len(clinic_coords)

#------------------------------------------
#new code / work in progress
list_of_required_clinic_details = []
for num in range(len(dict_idx_clinicname_pc.keys())):
    temp_list = []
    temp_list.append(clinics[num].clinic_name)
    temp_list.append(clinics[num].lat)
    temp_list.append(clinics[num].long)
    list_of_required_clinic_details.append(temp_list)

df_clinics_lat_long_name = pd.DataFrame(data = list_of_required_clinic_details, columns=['ClinicName', 'Lat', 'Long'])
df_clinics_lat_long_name.to_csv(f'{carbon_emissions_assets_path}/OutputFromAPI.csv', index=False)

# ---------------------------------------

"""
Read-in the .gov.uk's .csv file of population weight centroids for all 34,000+ LSOAs 
in the country. This includes their x and y coordinates, but excludes the Upper Tier Local 
Authority (UTLA) in which they reside. Therefore, also read-in the .csv reference file of 
LSOAs-to-UTLA and join this to the first dataframe. Rename columns to lower-case for 
consistency.
"""

#create df of lsoa pop weight centroids csv
df_lsoas_file_csv = pd.read_csv('raw_data\govuk\\LSOA_Dec2011_Pop_Weighted_Centroids.csv')

#convert df col headers to lowercase
df_lsoas_file_csv = df_lsoas_file_csv.rename(columns=str.lower)

#create df of lsoa to upper tier local authority csv
df_lsoa_to_utla = pd.read_csv('raw_data\open_geography_portal_lookups\Lower_Layer_Super_Output_Area_(2011)_to_Upper_Tier_Local_Authorities_(2021)_Lookup_in_England_and_Wales_.csv')

#convert df col headers to lower case
df_lsoa_to_utla = df_lsoa_to_utla.rename(columns=str.lower)

#create merged df, consisting of both x and y coords, for each pop weight centroid of each lsoa, as well the upper tier LA in which that LSOA is located
df_merged = df_lsoa_to_utla.merge(df_lsoas_file_csv, on='lsoa11cd', how='left') 
#df_merged

# ---------------------------------------

#find the UTLAs that match the chosen LA of interest (as identified in the params file)
#subset the df, and store this in the relevant subfolder in the assets_produced_by_code folder 
#this df is used in the code, but the file saved here isnt actually called back into the file. Could be used for local testing etc.

#https://towardsdatascience.com/8-ways-to-filter-a-pandas-dataframe-by-a-partial-string-or-pattern-49f43279c50f
#mask = df_merged['utla21nm'].str.contains(la_of_interest, case=False, na=False) #original code, but prone to error where an LA appears in another LAs name (e.g. Derby appears in the word Derbyshire)
mask = df_merged['utla21nm'] == la_of_interest #replaces above line
df_merged_filtered = df_merged[mask]
#df_merged_filtered.to_csv()

#create new column containing a tuple of the x and y columns for each row - possibly not required
df_merged_filtered['tuple_x_y'] = list(zip(df_merged_filtered.x, df_merged_filtered.y))

#create empty list and populate this with the lat_long tuple for each x y coordinate pair from the df
lat_long = []
for index, row in df_merged_filtered.iterrows():
    lat_long.append(
        OSGB36toWGS84(row['x'], row['y'])
        )

#add this lat long tuple into the df
df_merged_filtered['lat_long'] = lat_long

#split the lat_long tuple into 2 new columns in the df, lat and long
df_merged_filtered[['lat', 'long']] = pd.DataFrame(df_merged_filtered['lat_long'].tolist(), index=df_merged_filtered.index)

#create new directory to temporarily store the df in, if it doesnt exist yet
file_name = "UTLA_of_choice_lsoas_with_coordinates"

file_path = f"{carbon_emissions_assets_path}\{file_name}.csv"

#save the subset df into the above temp folder location
df_merged_filtered.to_csv(file_path)

#test print
#print(df_merged_filtered)

# ---------------------------------------

"""
Now we have lat and long coordinates in the standard DataFrame, this is converted to 
a GeoPandas GeoDataFrame
"""

#convert the subset df to a geodata frame, and pass in which columns in the df are the long and lat respectively
gdf = gpd.GeoDataFrame(df_merged_filtered, geometry=gpd.points_from_xy(df_merged_filtered.long, df_merged_filtered.lat))

# ---------------------------------------
print()

lsoas = []
for index, row in df_merged_filtered.iterrows():
    lsoa_id = row['lsoa11cd'],
    #long = row['long'], #delete in final if this revised code works
    #lat = row['lat'] #delete in final if this revised code works
    long = row['lat_long'][1]
    lat = row['lat_long'][0]

    #create a new Lsoa Class for each lsoa in the data frame, with properties lsoa_id, long and lat). Append each to a temp list (lsoas) which is a list of Class objects
    new_lsoa = Lsoa(lsoa_id, long, lat)
    lsoas.append(new_lsoa)

#Create a new list (lsoa_coords) using a list comprehension, consiting of the 
lsoa_coords = [[l.long, l.lat] for l in lsoas]
lsoas_length = len(lsoa_coords)

#print(lsoas_length) # test print (delete in final)


# ---------------------------------------

#create an empty list, and then populate this with the lsoa coordinates, and then the clinic coordinates
combined_coords = []
combined_coords += lsoa_coords

length_combined_coords = len(combined_coords)
print(f"\nThe length of combined_coords with just LSOA coordinates is:  {length_combined_coords}")

combined_coords += clinic_coords

num_clinics = len(clinic_coords)

length_combined_coords = len(combined_coords)
print(f"The length of combined_coords with LSOA coordinates and clinic coordinates is:  {length_combined_coords}")
print(f"There are {num_clinics} clinics included in this analysis.")

# ---------------------------------------

#run press_any_key_to_continue function, to pause the code execution tp
#allow user to sense check the above output
press_any_key_to_continue()

# ---------------------------------------

#test cell - delete in final
#list(range (len(combined_coords) - num_clinics, len(combined_coords)))

# ---------------------------------------
#ORS API has a limit of 3500 routes. Lowered to cap at 3000. 
#the size of the matrix is calculated below. This is equivalent to the number of routes 
#for which distances are to be calculated. 
# if this number of routes is greater than the API cap of 3500 chunk_up is set to True
# If chunk_up is True, we need to loop the API to avoid an error being caused by exceeding 
#the 3500 route limit.

matrix_size = lsoas_length * clinics_length
chunk_up = matrix_size > 3500 

# ---------------------------------------
print(f"\nThe Open Route Service API is now running to calculate the distance between every single individual LSOA and geo-coded clinic location, to produce a 'distance matrix' consisting of all possible combinations / routes (total = {matrix_size}).\nFor more information, please visit: https://openrouteservice.org/ \n")
#work in progress to get up and running with the long list of all lsoas and clinics
apiUrl = "https://api.openrouteservice.org/v2/matrix/driving-car"

request_headers = {
    'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
    'Authorization': apiKey,
    'Content-Type': 'application/json; charset=utf-8'
}

#this step line ensures the API runs, with complete matrices returned, for any length of clinics_length
step = 3500 // clinics_length

distances = []

for window_start in range(0, len(lsoa_coords), step):
    window_end = window_start + step

    if window_end > len(lsoa_coords):
        window_end = len(lsoa_coords)

    request_body = {
        "locations": combined_coords,
        #destinations explanation: destinations are at end of the list, so pass in the position in the list from the end of the lsoas to the end of the list
        "destinations": list(range (len(combined_coords) - clinics_length, len(combined_coords))),
        "metrics":["distance"],
        "units":"mi",
        "sources": list(range(window_start, window_end))
    }

    response = requests.post(apiUrl, json=request_body, headers=request_headers)

    print(response.status_code, response.reason)
    print(response.text)

    json_object = response.json()
    distances += json_object["distances"]
    #just_clinic_to_lsoa_distances = distances[0:lsoas_length]
    #distances += just_clinic_to_lsoa_distances

"""
    if call.status_code == 200:
        json_object = response.json()
        distances = json_object["distances"]
        just_clinic_to_lsoa_distances = distances[0:lsoas_length]
        distances.append(just_clinic_to_lsoa_distances)
    else:
        print("Request failed")
        print(response.reason)
"""

# ---------------------------------------

#test print to check api call returns 491 rows in df for Derbyshire (as there are 491 LSOAs in Derbyshire)
matrix_to_df = pd.DataFrame(distances)

# ---------------------------------------

"""
Rename the column headers to match the *actual* clinic names in the original DataFrame
"""
matrix_to_df_copy = matrix_to_df.copy()

for num in matrix_to_df_copy.columns: #get a list of the column names in the order they were in the original clinics df
    temp_name = list(df_clinics_selected_LA_only.clinic_name)[num]
    matrix_to_df_copy.rename(columns={num: temp_name}, inplace = True)

#test print
#print(matrix_to_df_copy)

# ---------------------------------------

"""
Get all LSOA reference IDs into a new list
"""

lsoas_id_strings = []
for lsoa in lsoas:
    lsoas_id_strings.append(lsoa.lsoa_id[0])
#lsoas_id_strings

# ---------------------------------------
"""
create temporary dictionary to convert integer to unique lsoa ID - need to think about 
how to use the chunk size in the for loop for the API calls (if required at all?)
"""

chunk_size = 20
keys = list(range(len(lsoas)))
values = lsoas_id_strings
dict_convert_indices = {k:v for (k,v) in zip(keys, values)}

# ---------------------------------------
"""
Use this dictionary to rename the index for each LSOA from unique integer to the 
actual LSOA ID for that LSOA
"""
matrix_to_df_copy.rename(index=dict_convert_indices, inplace=True)
#matrix_to_df_copy #test print call, delete in final

# ---------------------------------------
"""
Finished distance matrix created and saved
"""
matrix_to_df_copy = matrix_to_df_copy.reindex(sorted(matrix_to_df_copy.columns), axis=1)
matrix_to_df_copy.sort_index(ascending=True, inplace=True)
matrix_to_df_copy.index.rename('lsoa_residence', inplace=True) 
#matrix_to_df_copy

#save as assets the distance matrix df that has now been produced:
file_prefix = number_saved_file(df_counter)
temp_filename = f"df{file_prefix}_DistanceMatrix.csv"
matrix_to_df_copy.to_csv(f'Assets_produced_by_code/04_Carbon_emissions_assets/{temp_filename}', index=True) #index was false, by making True this includes lsoa_residence in column A
df_counter+=1
print("Distance Matrix File saved.")

# ---------------------------------------
"""
Create like for like structured patient matrix

DEV REQUIRED: Need to check in pre-processing file where the file called
crosstab_total_activity_all_attend_reasons_lsoa_clinic_location_manual_adjusted.csv is created
as we need one of these for each condition in list of conditions. 

"""
# ---------------------------------------

patient_matrix = pd.read_csv("processed_data/2_Carbon_Emissions/crosstab_total_activity_all_attend_reasons_lsoa_clinic_location.csv")
#lsoa_residence = lsoa_residence = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 4).lower()
#patient_matrix.rename(columns={lsoa_residence : 'lsoa_residence'}, inplace=True)
lsoa_residence = 'lsoa_residence'

#the patient matrix has been produced to include ALL clinics present. Where a provider runs a service
#that spans >1 Local Authority area, this means there will be clinics present that are not relevant 
#for the LA of interest entered in the user_params file. 
#To control for this, this next block of code (for-if-if loop) adjusts the patient matrix df
#to only include the columns/clinic names that match those associated with the la of interest
#This also removes the 'All' total column which was originally was present to the far right of 
#the patient matrix file that was read-in above (assumption is there is no clinic called 'All'..)
list_patient_matrix_cols = patient_matrix.columns
for col in list_patient_matrix_cols:
    if col != 'lsoa_residence':
        if col not in list_of_clinics:
            patient_matrix.drop(col, axis=1, inplace=True)

#remove the 'All' total row from patient matrix
patient_matrix = patient_matrix[patient_matrix['lsoa_residence'] != 'All']

#The patient matrix currently has an incremental integer as the index
#We don't want this. The next code mades the lsoa_residence column the index for the df
#and then sorts the resulting df based on the lsoa_residence index.
patient_matrix.set_index('lsoa_residence', drop=True, inplace=True)
patient_matrix.reindex(sorted(patient_matrix.columns), axis=1)
patient_matrix.sort_index(ascending=True, inplace=True)
#patient_matrix

# ---------------------------------------
"""
DECISION NEEDED: 
Code to produce patient matrix inc all LSOAs in LA of choice is below. This is fine when calculating 
baseline emissions, but probably need to ALSO put this into a for loop for all condition(s) 
in scope and save complete patient matrixes for each condition in order to calculate potential
change to emissions based on potential new clinic location(s). 
To do this, need to check in HEA or pre-processing file for where the files called 
“tempdf_unique_patients_Both” are produced. Need to produce lsoa to cross tabs for each 
condition (con, GU, Both etc.) and save these, to then read these in within the loop, repurposing
the code on line 543 to 546 to do so. 
"""
# ---------------------------------------

df1_all_lsoas_in_la_of_choice = pd.DataFrame(index=matrix_to_df_copy.index) #has all lsoas present
df2_those_lsoas_from_we_have_seen_patients = pd.DataFrame(index=patient_matrix.index) #cannot be assumed to have all lsoas present
df1_all_lsoas_in_la_of_choice.reset_index(inplace=True)
df2_those_lsoas_from_we_have_seen_patients.reset_index(inplace=True)

#combine the df of lsoas in patient matrix (<= all lsoa in LA) to the df of ALL lsoas in the area
#this creates a new df with a column called _merge which shows whether the row was only in the 
#left df (ALL lsoas present) indicated by left_only in the _merge column, and meaning the LSOA
#WAS NOT in the patient matrix (So we had no one from that LSOA that was seen by the service), OR
#it will show that the LSOA was in BOTH dfs, meaning the service had at least 1 person from the LSOA
#seen by the service, indicated by 'both' string in the _merge column.
df_all = df1_all_lsoas_in_la_of_choice.merge(df2_those_lsoas_from_we_have_seen_patients.drop_duplicates(), on=['lsoa_residence'], how='left', indicator=True)

#create a mask and subset df_all, to create a new subset df called subset
#this subset df contains ONLY the rows from the df_all dataframe corresponding to LSOAs
#with a value of 'left_only' in the '_merge' column (i.e. all LSOAs from which the service
# has not seen anyone)
subset_lsoas_from_which_we_have_not_seen_patients = df_all[df_all['_merge'] == 'left_only']
#subset

#drop the _merge column
subset_new = subset_lsoas_from_which_we_have_not_seen_patients.drop('_merge',axis=1)
#subset

#reset index for patient matrix file
df_patient_matrix_reset_index = patient_matrix.reset_index()

#create a concatenated df consisting of the count of patients we have seen from each lsoa going to each clinic, and a 0 count for each lsoa from which we have NOT seen patients at any of the clinics
df_full_patient_matrix = pd.concat([df_patient_matrix_reset_index, subset_new])
df_full_patient_matrix = df_full_patient_matrix.fillna(0)
#df_full_patient_matrix

#If the data included patients with LSOAs from an LA area OTHER THAN the one we are running this analysis for, remove those LSOA rows.
df_full_patient_matrix_id_target_lsoas = df_full_patient_matrix.merge(df1_all_lsoas_in_la_of_choice.drop_duplicates(), on=['lsoa_residence'], how='left', indicator=True)
df_full_patient_matrix_id_target_lsoas = df_full_patient_matrix_id_target_lsoas[df_full_patient_matrix_id_target_lsoas['_merge'] == 'both']
df_full_patient_matrix_id_target_lsoas.drop('_merge', axis=1, inplace=True)
df_full_patient_matrix_id_target_lsoas.reset_index(inplace = True)
df_full_patient_matrix_id_target_lsoas.set_index('lsoa_residence', drop=True, inplace=True)
df_full_patient_matrix_id_target_lsoas.sort_index(ascending=True, inplace=True)
df_full_patient_matrix_id_target_lsoas.drop('index', axis=1, inplace=True)
df_full_patient_matrix_id_target_lsoas

#save as assets the patient matrix df that has now been produced:
file_prefix = number_saved_file(df_counter)
temp_filename = f"df{file_prefix}_PatientMatrix.csv"
df_full_patient_matrix_id_target_lsoas.to_csv(f'Assets_produced_by_code/04_Carbon_emissions_assets/{temp_filename}', index=True) #index was false, by making True this includes lsoa_residence in column A
df_counter+=1
print("Patient Matrix File saved.")

# ----------------------------------------------------------
#End of file original code for distance and patient matrixes
# ----------------------------------------------------------

patient_matrix = pd.read_csv(r"Assets_produced_by_code//04_Carbon_emissions_assets//df002_PatientMatrix.csv")
#patient_matrix = df_full_patient_matrix_id_target_lsoas
distance_matrix =  pd.read_csv(r"Assets_produced_by_code//04_Carbon_emissions_assets//df001_DistanceMatrix.csv").fillna(0)
#distance_matrix =  matrix_to_df_copy

#16/6/23 - code to ensure patient matrix definitely matches structure of distance matrix. 
#it is possible, once processed, the processed data for this section doesnt include all clinics 
#in which case the patient matrix will have fewer columns than (and not match) the distance matrix
#this will cause an error. The below code updates the patient matrix with any missing clinics and assigns zero values
patient_matrix_columns = patient_matrix.columns
distance_matrix_columns = distance_matrix.columns

missing_columns = set(distance_matrix_columns) - set(patient_matrix_columns)

for col in missing_columns:
    patient_matrix[col] = 0

patient_matrix = patient_matrix.reindex(columns=distance_matrix_columns)

#get lsoas from each matrix into respective lists
patient_lsoa = patient_matrix['lsoa_residence'].tolist()
distance_lsoa = distance_matrix['lsoa_residence'].tolist() #was all_lsoa

#NEW approach - creating a new dict for each distance category. This populates a nested dict with travel mode as outer key, and distance as inner key.
df_proportion_travelling_each_mode_varying_distance = pd.read_excel(filename, 'emissions_values_new', skiprows=2, usecols=[0, 2,3,4,5,6])
df_proportion_travelling_each_mode_varying_distance.set_index('means_of_travel', inplace=True)
list_col_headers = list(df_proportion_travelling_each_mode_varying_distance.columns)
dict_travel_means_by_distance_travelled = {}
for col in list_col_headers:
    dict_travel_means_by_distance_travelled[col] = df_proportion_travelling_each_mode_varying_distance.loc[:, list_col_headers[0]].to_dict()

#Carbon dioxide emissions in kg per mile per travel modality, taken from:
#https://www.gov.uk/government/publications/transport-energy-and-environment-statistics-notes-and-definitions/journey-emissions-comparisons-methodology-and-guidance

df_travel_means_kg_emissions_per_mile = pd.read_excel(filename, 'emissions_values_new', skiprows=2, usecols=[0,7])
df_travel_means_kg_emissions_per_mile.set_index('means_of_travel', inplace=True)
dict_kg_co2e_per_mile_per_travel_means = df_travel_means_kg_emissions_per_mile.to_dict()['mean_co2e_kg_per_mile_per_travel_means']
#dict_kg_co2e_per_mile_per_travel_means

#set up lists to populate with the distances travelled
list_dist_less_than_1_mile = []
list_dist_1_to_2_miles = []
list_dist_2_to_5_miles = []
list_dist_5_to_10_miles = []
list_dist_more_than_10_miles = []

#set up lists to populate with the patients that travelled the distances in each category
list_patients_less_than_1_mile = []
list_patients_1_to_2_miles = []
list_patients_2_to_5_miles = []
list_patients_5_to_10_miles = []
list_patients_more_than_10_miles = []

#populate lists with tuples consisting of the index and column label for all cells 
#currently hardcoded to distance <1 miles in the nested for loop below
list_less_than_1_mile_coords = []
#currently hardcoded to distance 1-2 miles in the nested for loop below
list_1_to_2_miles_coords = []
#currently hardcoded to distance 2-5 miles in the nested for loop below
list_2_to_5_miles_coords = []
#currently hardcoded to distance 5-10 miles in the nested for loop below
list_5_to_10_miles_coords = []
#currently hardcoded to distance >=10 miles in the nested for loop below
list_more_than_10_miles_coords = []

#set index of distance matrix to be lsoa
distance_matrix_reset_index = distance_matrix.set_index('lsoa_residence')

#set the index of patient matrix to be lsoa
patient_matrix_reset_index = patient_matrix.set_index('lsoa_residence')

for col in distance_matrix_reset_index:
    for i, row_value in distance_matrix_reset_index[col].iteritems():
        #df[col][i] = row_value * df['views'][i]
        dist_matrix_cell_value = distance_matrix_reset_index.loc[i,col]

        if dist_matrix_cell_value < 1:
            list_dist_less_than_1_mile.append(dist_matrix_cell_value)
            list_less_than_1_mile_coords.append((i, col))
            patient_num  = patient_matrix_reset_index.loc[i,col] 
            list_patients_less_than_1_mile.append(patient_num)

        elif dist_matrix_cell_value <2:
            list_dist_1_to_2_miles.append(dist_matrix_cell_value)
            list_1_to_2_miles_coords.append((i, col))
            patient_num  = patient_matrix_reset_index.loc[i,col] 
            list_patients_1_to_2_miles.append(patient_num)

        elif dist_matrix_cell_value <5:
            list_dist_2_to_5_miles.append(dist_matrix_cell_value)
            list_2_to_5_miles_coords.append((i, col))
            patient_num  = patient_matrix_reset_index.loc[i,col] 
            list_patients_2_to_5_miles.append(patient_num)

        elif dist_matrix_cell_value <=10:
            list_dist_5_to_10_miles.append(dist_matrix_cell_value)
            list_5_to_10_miles_coords.append((i, col))
            patient_num  = patient_matrix_reset_index.loc[i,col] 
            list_patients_5_to_10_miles.append(patient_num)

        else:
            list_dist_more_than_10_miles.append(dist_matrix_cell_value)
            list_more_than_10_miles_coords.append((i, col))
            patient_num  = patient_matrix_reset_index.loc[i,col] 
            list_patients_more_than_10_miles.append(patient_num)

#convert the patient lists to numpy arrays
list_patients_less_than_1_mile = np.array(list_patients_less_than_1_mile)
list_patients_1_to_2_miles = np.array(list_patients_1_to_2_miles)
list_patients_2_to_5_miles = np.array(list_patients_2_to_5_miles)
list_patients_5_to_10_miles = np.array(list_patients_5_to_10_miles)
list_patients_more_than_10_miles = np.array(list_patients_more_than_10_miles)

#convert the lists to arrays to allow multiplication
list_dist_less_than_1_mile = np.array(list_dist_less_than_1_mile)
list_dist_1_to_2_miles = np.array(list_dist_1_to_2_miles)
list_dist_2_to_5_miles = np.array(list_dist_2_to_5_miles)
list_dist_5_to_10_miles = np.array(list_dist_5_to_10_miles)
list_dist_more_than_10_miles = np.array(list_dist_more_than_10_miles)

#derive the estimated CO2e for each group. first create and populate a dictionary with the summed miles travelled by patients in each distance category derived above
dict_patients_per_distance_category = {}
dict_patients_per_distance_category['less than 1 mile'] = (list_patients_less_than_1_mile * list_dist_less_than_1_mile).sum()
dict_patients_per_distance_category['1-2 miles'] = (list_patients_1_to_2_miles * list_dist_1_to_2_miles).sum()
dict_patients_per_distance_category['2-5 miles'] = (list_patients_2_to_5_miles * list_dist_2_to_5_miles).sum()
dict_patients_per_distance_category['5-10 miles'] = (list_patients_5_to_10_miles * list_dist_5_to_10_miles).sum()
dict_patients_per_distance_category['> 10 miles'] = (list_patients_more_than_10_miles * list_dist_more_than_10_miles).sum()

#set up the lists containing the values we will loop over shortly to derive the CO2e estimate
list_distance_categories = list(dict_travel_means_by_distance_travelled.keys())
list_travel_means = list(dict_kg_co2e_per_mile_per_travel_means.keys())

total_emissions = 0
for dist_category in list_distance_categories:
    for travel_means in list_travel_means:
        co2e_for_travel_means_and_distance_combination = (dict_patients_per_distance_category[dist_category] * dict_travel_means_by_distance_travelled[dist_category][travel_means]) * dict_kg_co2e_per_mile_per_travel_means[travel_means]
        total_emissions += co2e_for_travel_means_and_distance_combination


if total_emissions > 1000:
    baseline_emissions_result_statement = f"The current service design has a total CO2e emissions of {round(total_emissions / 1000, 2)} tonnes."
else:
    baseline_emissions_result_statement = f"The current service design has a total CO2e emissions of {round(total_emissions, 2)} Kg."

print(baseline_emissions_result_statement)

file_prefix = number_saved_file(df_counter)
temp_filename = "TotalEmissionsBaseline.txt"
filepath = f'Assets_produced_by_code/04_Carbon_emissions_assets/{temp_filename}'

#open text file
text_file = open(filepath, "w")
 
#write string to file
text_file.write(baseline_emissions_result_statement)
 
#close file
text_file.close()
print("Total emissions txt file saved.")
