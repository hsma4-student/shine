#A file to create the maps of IMD by LSOA, and for each condition in scope, a map of modelled unmet need by lsoa.
# -----------------------------------------------------
#Import libraries
# -----------------------------------------------------

import pandas as pd
import geopandas
import contextily as cx #for base maps
import matplotlib.pyplot as plt
import os
from bng_latlon import WGS84toOSGB36
from mpl_toolkits.axes_grid1 import make_axes_locatable

#import other .py files for use in this file
#import HEA_unmet_needv03 as hea

# -----------------------------------------------------------------
# <<< Setting up folder structures, if not already in existence >>>
# -----------------------------------------------------------------

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

# -----------------------------------------------------
# <<< set up parameters >>>
# -----------------------------------------------------

#identify the current directory from which the code is running / in which the file is located
current_directory = os.getcwd()

#read-in la of interest from user params file
filename = f"/raw_data/user_and_data_parameters/user_and_data_params.xlsx"
filepath = current_directory+filename

la_of_interest = pd.read_excel(filepath, 'HEA_parameters', index_col=None, usecols = "C", header = 3, nrows=0)
la_of_interest = list(la_of_interest)[0]
la_of_interest = str(la_of_interest)

map_counter = 1


# -----------------------------------------------------
# <<< Define functions >>>
# -----------------------------------------------------

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


"""
file_prefix = number_saved_file(map_counter)
    temp_filename = f"chart{file_prefix}_{string_for_file_name}-{field_mapped}.png"
    file_path = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}'
"""

#----------------------------------------

def make_lsoa_map(
    chosen_la, 
    merge_LSOA_boundaries, 
    df_with_data_to_merge, 
    list_of_fields_to_merge, 
    lsoa11cd_fieldname,
    field_to_map,
    color_map,
    title,
    category_bool,
    legend_bool, 
    map_counter, 
    filepath):
    
    """"
    Function to create a shaded map for the LA of interest. 
    Allows shading to be defined when called.
    """

    """
    #sallys code to add in below 
    geometry = geopandas.GeoDataFrame(clinics_data, geometry=geopandas.points_from_xy(clinics_data.long, clinics_data.lat))
    fig,ax = plt.subplots(figsize = (20,50))
    merged_Boundaries.plot(ax=ax, column='pt_count', categorical = True, legend = True)
    geometry.plot(ax=ax, markersize = 70, color = 'red', marker = '*')
    plt.axis('off')
    plt.show()
    """

    mask = merge_LSOA_boundaries["UTLA21NM"] == chosen_la
    masked_lsoa = merge_LSOA_boundaries[mask]
    full_merge = masked_lsoa.merge(df_with_data_to_merge[list_of_fields_to_merge], left_on='lsoa21cd', right_on=lsoa11cd_fieldname)
    plot_map = full_merge.plot(figsize=(15, 15), column=field_to_map, categorical = category_bool, legend = legend_bool, cmap=color_map)
    plot_map.set(title=title)
    plot_map.set_axis_off()

    #plt.savefig(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}')
    plt.savefig(filepath, bbox_inches='tight')
    map_counter+=1
    print("File saved.")

    return(plot_map, map_counter)



# -----------------------------------------------------

def create_condition_to_param_dict(df, desired_col_list, value_column_string, number_of_conditions, cast_as_type):
    
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

# -----------------------------------------------------
#new plot map function to include clinic points and labels
def make_lsoa_map_with_clinic_labels(
    chosen_la, 
    merge_LSOA_boundaries, 
    df_with_data_to_merge, 
    list_of_fields_to_merge, 
    lsoa11cd_fieldname,
    field_to_map,
    color_map,
    title,
    category_bool,
    legend_bool,
    map_counter,
    filepath,
    df_geocoded_clinic_locations,
    label_colour):
    
    """"
    Function to create a shaded map for the LA of interest. 
    Allows shading to be defined when called.
    """

    mask = merge_LSOA_boundaries["UTLA21NM"] == chosen_la
    masked_lsoa = merge_LSOA_boundaries[mask]
    full_merge = masked_lsoa.merge(df_with_data_to_merge[list_of_fields_to_merge], left_on='lsoa21cd', right_on=lsoa11cd_fieldname)
    
    list_tuples = []
    for num in range(len(df_geocoded_clinic_locations)):
        temp_tuple = (df_geocoded_clinic_locations.iloc[num, 0], df_geocoded_clinic_locations.iloc[num,1], df_geocoded_clinic_locations.iloc[num,2])
        list_tuples.append(temp_tuple)

    #loop through the lat long for each clinic in the df and convert to easting northing
    list_tupe_easting_northing = []
    for pair in list_tuples:
        list_tupe_easting_northing.append(WGS84toOSGB36(pair[1], pair[2]))

    #loop through each lat long and convert to easting northing (excludes clinic name)
    list_clinic_name_easting_northing = [] #test_list
    for num in range(len(df_geocoded_clinic_locations)):
        temp_list = []
        temp_list.append(list_tuples[num][0])
        temp_list.append(list_tupe_easting_northing[num][0])
        temp_list.append(list_tupe_easting_northing[num][1])
        list_clinic_name_easting_northing.append(temp_list)

    #test_list
    geo_df_clinic_name_easting_northing = geopandas.GeoDataFrame(list_clinic_name_easting_northing, columns = ['ClinicName', 'Easting', 'Northing'])
    geo_df_clinic_name_easting_northing['EastingNorthing'] = list_tupe_easting_northing
    
    geometry_clinic_points = geopandas.points_from_xy(geo_df_clinic_name_easting_northing.Easting, geo_df_clinic_name_easting_northing.Northing)
    geodf_geometry_clinic_points = geopandas.GeoDataFrame(geometry_clinic_points, columns=["clinic_easting_northing"])
    
    #commence plot
    fig, ax = plt.subplots (figsize = (15,15))
    #divider = make_axes_locatable(ax) #new line
    #cax = divider.append_axes("bottom", size="5%", pad=0.1) #new line
    geodf_geometry_clinic_points['clinic_easting_northing'].plot(ax = ax, alpha = 0.7, color = label_colour, zorder = 2, markersize=100, marker="*")
    #mapplot = full_merge['geometry'].plot(ax = ax, zorder=1)
    mapplot = full_merge.plot(ax = ax, figsize=(15, 15), zorder=1, column=field_to_map, categorical=category_bool, cmap=color_map, legend = legend_bool, alpha=0.5, edgecolor='black') #add cax=cax though requires work
    
    mapplot.set_axis_off()
    #mapplot.set(title = title)
    plt.title(label = title, fontsize=16)
    for idx, row in geo_df_clinic_name_easting_northing.iterrows():
        mapplot.annotate(text=row['ClinicName'], xy=row['EastingNorthing'], horizontalalignment='left', weight = 'normal', color='blue', font='Arial', fontsize='large')
    
    cx.add_basemap(mapplot, crs=full_merge.crs.to_string())

    #plt.savefig(f'Assets_produced_by_code/02_HEA_assets/{temp_filename}')
    plt.savefig(filepath, bbox_inches='tight')
    map_counter+=1
    print("File saved.")
    return(mapplot, map_counter)

# -----------------------------------------------------
# -----------------------------------------------------
# <<< Code starts! >>>
# -----------------------------------------------------
# -----------------------------------------------------

#Read in data required
cwd = os.getcwd()
file_path = f"{cwd}/Assets_produced_by_code/04_Carbon_emissions_assets/OutputFromAPI.csv"
df_geocoded_clinic_locations = pd.read_csv(file_path)
df_geocoded_clinic_locations


boundaries_LSOA = geopandas.read_file(r"raw_data\open_geography_boundary_files\LSOA_shapefiles\Lower_Layer_Super_Output_Areas_(December_2011)_Boundaries_Generalised_Clipped_(BGC)_EW_V3.shp")
LSOA_UTLA = pd.read_csv(r"raw_data\open_geography_portal_lookups\Lower_Layer_Super_Output_Area_(2011)_to_Upper_Tier_Local_Authorities_(2021)_Lookup_in_England_and_Wales_.csv")
boundaries_LSOA.rename(columns={"LSOA11CD":"lsoa21cd"}, inplace=True) # this line can be removed once ONS release 2021 boundaries and these 2021 boundaries are read-in instead

#filter / subset data sets to the la of interest
merge_LSOA_boundaries = boundaries_LSOA.merge(LSOA_UTLA[['UTLA21NM','LSOA11CD']],left_on='lsoa21cd', right_on='LSOA11CD')
mask = merge_LSOA_boundaries["UTLA21NM"] == la_of_interest
masked_lsoa = merge_LSOA_boundaries[mask] 

# -----------------------------------------------------

#read in data on IMD decile value for each LSOA in the country
## 2019 IMD decile file csv
LSOA_ALL_ID2019 = pd.read_csv("https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/833982/File_7_-_All_IoD2019_Scores__Ranks__Deciles_and_Population_Denominators.csv")
#LSOA_ALL_ID2019 = pd.read_csv(current_directory + "\\raw_data\ministry_of_housing_communities_local_gov\Indices_of_Multiple_Deprivation_(IMD)_2019.csv")

##changing column names because it is easier to use later 
# ## remember to update the lad**cd column to whichever year matches the full list of 
# LAD/OA/MSOA/LSOA - this won't change on IMD until they do a new one

LSOA_ALL_ID2019.columns.values[0:34]= ["lsoa21cd", "LSOA_Name","lad22cd", "LA_Name", "IMD_Score", "IMD_Rank", "IMD_Decile", "Income_Score", "Income_Rank", "Income_Decile", "Emp_Score", "Emp_Rank", "Emp_Decile", "Edu_Score", "Edu_Rank", "Edu_Decile", "Health_Score", "Health_Rank", "Health_Decile", "Crime_Score", "Crime_Rank", "Crime_Decile", "Barriers_Score", "Barriers_Rank", "Barriers_Decile", "Liv_Env_Score", "Liv_Env_Rank", "Liv_Env_Decile", "IDACI_Score", "IDACI_Rank", "IDACI_Decile", "IDAOPI_Score", "IDAOPI_Rank", "IDAOPI_Decile"]

#subset just the first 7 columns from the huge IMD df 
IMD_score_subset = LSOA_ALL_ID2019.iloc[:,:34]

#get a list of all field names in the IMD_score_subet df
list_fields = list(IMD_score_subset.columns)

#get a subset of these field names in a new list called "items we want" where they are Decile values
items_we_want = [item for item in list_fields if item.split("_")[-1] == "Decile"]

#merge the IMD decile info with the shapefile df for derbyshire lsoa
final_items_list = ['lsoa21cd']

for item in items_we_want:
    final_items_list.append(item)

df_full_merge = masked_lsoa.merge(IMD_score_subset[final_items_list])

# -----------------------------------------------------------
# <<< Call map function to create a local IMD map by LSOA >>>
# -----------------------------------------------------------

file_prefix = number_saved_file(map_counter)
string_for_file_name = '_Lsoa'
field_mapped = 'IMDDecile'

temp_filename = f"map{file_prefix}{string_for_file_name}{field_mapped}.png"
filepath = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}'

#original function call - working
'''
test_var, map_counter = make_lsoa_map(
    la_of_interest,
    merge_LSOA_boundaries,
    df_full_merge,
    ['lsoa21cd','IMD_Decile'],
    'lsoa21cd',
    'IMD_Decile',
    "RdYlGn",
    'IMD Deciles for LSOAs',
    True,
    True,
    map_counter,
    filepath
    )
'''
#test function call to add clinic names
test_var, map_counter = make_lsoa_map_with_clinic_labels(
    la_of_interest,
    merge_LSOA_boundaries,
    df_full_merge,
    ['lsoa21cd','IMD_Decile'],
    'lsoa21cd',
    'IMD_Decile',
    "RdYlGn",
    'IMD Deciles for LSOAs',
    True,
    True,
    map_counter,
    filepath,
    df_geocoded_clinic_locations,
    'pink'
    )


# -----------------------------------------------------------------------
# <<< Set up a for loop to map unmet need for each condition in scope >>>
# -----------------------------------------------------------------------
#New code to read in list of conditions from the parameters file:
#Read-in user parameters
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

#df_condition_params_original['prevalence_multiplier'] = df_condition_params_original[temp_numerator_col_name] / df_condition_params_original[temp_denominator_col_name]
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
    temp_proportion_seen_by_this_service: "proportion_seen_by_service"
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

#new code 17/9 - start
#create dictionary of condition name to min age
dict_min_age_for_each_condition = create_condition_to_param_dict(df_condition_params_original, [0, 4], min_age_col_name, number_of_conditions_modelling_for, str)

#create dictionary of condition name to max age
dict_max_age_for_each_condition = create_condition_to_param_dict(df_condition_params_original, [0, 5], max_age_col_name, number_of_conditions_modelling_for, str)

#create dictionary of condition to prevalence rate type
dict_condition_to_prevalence_rate = create_condition_to_param_dict(df_condition_params_original, [0, 1], prev_type_col_name, number_of_conditions_modelling_for, str)
dict_prev_ref_for_each_condition = {}

for condition in list_of_conditions:
    ref_num = int(dict_condition_to_prevalence_rate[condition].split(":")[0])
    dict_prev_ref_for_each_condition[condition] = ref_num

#new code 17/9 - end


# -----------------------------------------------------------------------
#create a variable containing the target directory to save map assets into
hea_assets_folder = current_directory+"\Assets_produced_by_code\\02_HEA_assets"

#create a dict which will be populated with conditions (keys) and data 
# files for unmet need (values)
dict_condition_df_unmet_need = {}

#list_of_conditions = ['GUM', 'Contraception']

for condition in list_of_conditions:
#for condition in hea.list_of_conditions:
    for file in os.listdir(hea_assets_folder):
        if file.endswith(".csv"):
            # Prints only text file present in My Folder
            df_count = file.split("_")[0]
            remove_file_extension = file.split(".")[0]
            initial_split = remove_file_extension.split('_')
            final_split = initial_split[1].split("-")
            if final_split[-1] == condition and final_split[0] == "EstimatedMetAndUnmetNeed":
                print("This is a file we want!")
                dict_condition_df_unmet_need[condition] = hea_assets_folder+"\\"+file
        #print(final_split)

# ----------------------------------------------------------------------------------------
# <<< Call map function within a for loop to create maps of all conditions' unmet need >>>
# ----------------------------------------------------------------------------------------

"""
Now we have the files for each condition in scope, we run a for loop, to 
call the map function, for each conditions' unmet need file, as relevant
"""

for condition in list_of_conditions:
#for condition in hea.list_of_conditions:
    df_condition_unmet_need_data = pd.read_csv(dict_condition_df_unmet_need[condition])
    df_condition_unmet_need_data.columns.values[3] = "unmet_need"

    file_prefix = number_saved_file(map_counter)
    string_for_file_name = "_LSOA"
    field_mapped = f'UnmetNeed-{condition}'

    temp_filename = f"map{file_prefix}{string_for_file_name}{field_mapped}.png"
    filepath = f'Assets_produced_by_code/02_HEA_assets/{temp_filename}'

    #new code 16/9
    #1 = Age standardised rate
    #2 = Crude rate
    #3 = Census or no rate
    if dict_prev_ref_for_each_condition[condition] == 1:
        title = f'Modelled unmet need using age standardised rates for {condition}'
    elif dict_prev_ref_for_each_condition[condition] == 2:
        title = f'Modelled unmet need using a crude prevalence rate for {condition}'
    else:
        title = f'Estimated pop. aged {dict_min_age_for_each_condition[condition]} to {dict_max_age_for_each_condition[condition]} not seen for {condition}'
    #new code ends
    '''
    test_var, map_counter = make_lsoa_map(
        la_of_interest,
        merge_LSOA_boundaries,
        df_condition_unmet_need_data,
        ['LSOA11CD','unmet_need'],
        'LSOA11CD',
        'unmet_need',
        "Reds",
        title, #f'Modelled Unmet Need for {condition}'
        False,
        True,
        map_counter,
        filepath
        )
    '''
    
    test_var, map_counter = make_lsoa_map_with_clinic_labels(
        la_of_interest,
        merge_LSOA_boundaries,
        df_condition_unmet_need_data,
        ['LSOA11CD','unmet_need'],
        'LSOA11CD',
        'unmet_need',
        "Reds",
        title, #f'Modelled Unmet Need for {condition}'
        False,
        True,
        map_counter,
        filepath,
        df_geocoded_clinic_locations,
        'blue'
        )

# -----------------------------------------------------------
# <<< End of file >>>
# -----------------------------------------------------------

