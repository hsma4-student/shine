'''Purpose of file
To produce profile of features more commonly associated with patients 
who DNA Using Supervised Machine Learning to do so

Method/overview of what this file does:
- ingest / load pre-processed data
- Split data into features (X) and label (y)
- Split data into training and test sets (we will test on data that has 
not been used to fit the model)
- Standardise data
- Fit a logistic regression model (from sklearn learn)
- Predict survival of the test set
- Define a function to calculate a range of accuracy measure (and return 
as a dictionary)
- Report multiple accuracy scores for model

... repeat above for total of 9 variants of Logistic Regression model.

Measures of accuracy
The common measure of accuracy is "proportion of cases where the 
classification / prediction was correct. DNAs are likely routinely a 
relatively rare event and the data likely to be imbalanced (i.e. it is 
not the case that 50% attend 50% DNA). If 1 person in 20 DNA's, the model 
could have a 95% accuracy, by predicting no one ever DNA's and only 
being wrong 1 in 20 times (i.e. missing every single DNA it exists 
to predict). For this reason, alternative, more sophisticated  accuracy 
measure(s) are needed:
    
    sensitivity = proportion of positive cases (DNA) correctly classified
    specificity = proportion of negative (didnt DNA) correctly classified
    
Machine learning thresholds can be adjusted to change the balance between 
these sensitivity and specificity.

In addition to sensitivity and specificity, common measures used in ML are:
    
    precision
    recall
    f1 (combination of precision and recall)

Full list of measures that returned from accuracy functon are listed below.
NOTE: These are relevant for binomial classification problems (attend/DNA).
Where there are >two possible classes, a confusion matrix is commonly used 
(see https://pythonhealthcare.org/2018/04/21/77-machine-learning-visualising-accuracy-and-error-in-a-classification-model-with-a-confusion-matrix/)

01) observed positive rate: proportion of observed cases that are +ve
02) predicted positive rate: proportion of predicted cases that are +ve
03) observed negative rate: proportion of observed cases that are -ve
04) predicted negative rate: proportion of predicted cases that are -ve  
05) accuracy: proportion of predicted results that are correct    
06) precision: proportion of predicted +ve that are correct
07) recall: proportion of true +ve correctly identified
08) f1: harmonic mean of precision and recall
09) sensitivity: Same as recall
10) specificity: Proportion of true -ve identified:        
11) positive likelihood: increased probability of true +ve if test +ve
12) negative likelihood: reduced probability of true +ve if test -ve
13) false positive rate: proportion of false +ves in true -ve patients
14) false negative rate: proportion of false -ves in true +ve patients
15) true positive rate: Same as recall
16) true negative rate
17) positive predictive value: chance of true +ve if test +ve
18) negative predictive value: chance of true -ve if test -ve
'''


# --------------------------------------------------
# <<<   import libraries   >>>
# --------------------------------------------------

#import standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import machine learning modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics #test new line
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score #for info: https://builtin.com/data-science/roc-curves-auc
from sklearn.calibration import calibration_curve
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures
from imblearn import under_sampling
from imblearn import over_sampling
from imblearn.over_sampling import SMOTENC # Use SMOTE for continuous data

#import neural net modules
#from keras.models import Sequential
#from keras.layers import Dense

import os

#import libraries for optimisation
import optuna

from tableone import TableOne, load_dataset

# --------------------------------------------------
# <<<   set up folders if they dont exist   >>>
# --------------------------------------------------

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

# --------------------------------------------------
# <<<   setting paramers   >>>
# --------------------------------------------------

#Parameters
test_fraction = 0.25
rand_state = 42 #global integer for use in all random_state keyword arguments
number_of_splits = 5 #(was 10) - used for k-fold cross validation stage
number_of_bins = 5 #used for learning plot in log-reg function 

#polynomial expansion parameters
max_features = 5 #originally this was 20, reduce to 5 due to laptop limitations

#top X number of features associated with DNA's
#top_x = 5 replaced with user_params file input below

#identify the value for top X number of features associated with DNA's from the user parameters file
filename = "raw_data/user_and_data_parameters/user_and_data_params.xlsx"
top_x = pd.read_excel(filename, 'DNA_ML_parameters', index_col=None, usecols = "C", header = 1, nrows=0)
top_x = list(top_x)[0]


#Finely-tuned SMOTENC parameters 
min_number_attendances = 150 #Used when varying balance of attend:DNA

#Neural Network Parameters - not yet used as NN not inc in the model yet.
num_epochs = 50
batch_size = 32
num_dims = 5

#set sample size for sample size vs accuracy chart. The dummy data has this set to 10. For large live data sets, it is recommended this is increased to 250
#Note, this would require you to have a data set with at least 250 data points present.
#sample_size_increment = 250 #original hard coded approach, replaced by user_parameter below
sample_size_increment = pd.read_excel(filename, 'DNA_ML_parameters', index_col=None, usecols = "C", header = 2, nrows=0)
sample_size_increment = list(sample_size_increment)[0]

"""
#Code to create a new sub-directory within the current working directory, to store model outputs
directory_name = "Assets"
try:
    #Create new directory
    os.mkdir(directory_name)
    print(f"Directory titled '{directory_name}' successfully created.")
except FileExistsError:
    #print if directory already exists
    print(f"A directory called '{directory_name}' already exists.")
    """


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

# --------------------------------------------------
# <<<   Define functions   >>>
# --------------------------------------------------

def standardise_data(X_train, X_test):
    """
    Standardise data
    We want all of out features to be on roughly the same scale, generally 
    leads to a better model, and allows us to more easily compare the 
    importance of different features.
    One simple method is to scale all features 0-1 (by subtracting the minimum 
    value for each value, and dividing by the new remaining maximum value).
    But a more common method used in many machine learning methods is 
    standardisation, where we use the mean and standard deviation of the training 
    set of data to normalise the data. We subtract the mean of the test set values, 
    and divide by the standard deviation of the training data. Note that the mean 
    and standard deviation of the training data are used to standardise the test 
    set data as well.
    Here we will use sklearn's StandardScaler method. This method also copes with 
    problems we might otherwise have (such as if one feature has zero standard 
    deviation in the training set).
    """ 
    # Initialise a new scaling object for normalising input data
    sc = StandardScaler() 
    # Set up the scaler just on the training set
    sc.fit(X_train)
    # Apply the scaler to the training and test sets
    train_std=sc.transform(X_train)
    test_std=sc.transform(X_test)    
    return train_std, test_std

# --------------------------------------------------

def calculate_accuracy(observed, predicted):
    
    # Converts list to NumPy arrays
    if type(observed) == list:
        observed = np.array(observed)
    if type(predicted) == list:
        predicted = np.array(predicted)
    
    # Calculate accuracy scores
    observed_positives = observed == 1
    observed_negatives = observed == 0
    predicted_positives = predicted == 1
    predicted_negatives = predicted == 0
    
    true_positives = (predicted_positives == 1) & (observed_positives == 1)
    
    false_positives = (predicted_positives == 1) & (observed_positives == 0)
    
    true_negatives = (predicted_negatives == 1) & (observed_negatives == 1)
    
    false_negatives = (predicted_negatives == 1) & (observed_negatives == 0)
    
    accuracy = np.mean(predicted == observed)
    
    precision = (np.sum(true_positives) /
                 (np.sum(true_positives) + np.sum(false_positives)))
        
    recall = np.sum(true_positives) / np.sum(observed_positives)
    
    sensitivity = recall
    
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    specificity = np.sum(true_negatives) / np.sum(observed_negatives)
    
    positive_likelihood = sensitivity / (1 - specificity)
    
    negative_likelihood = (1 - sensitivity) / specificity
    
    false_positive_rate = 1 - specificity
    
    false_negative_rate = 1 - sensitivity
    
    true_positive_rate = sensitivity
    
    true_negative_rate = specificity
    
    positive_predictive_value = (np.sum(true_positives) / 
                                 np.sum(observed_positives))
    
    negative_predictive_value = (np.sum(true_negatives) / 
                                  np.sum(observed_negatives))
    
    # Create dictionary for results, and add results
    results = dict()
    
    results['observed_positive_rate'] = np.mean(observed_positives)
    results['observed_negative_rate'] = np.mean(observed_negatives)
    results['predicted_positive_rate'] = np.mean(predicted_positives)
    results['predicted_negative_rate'] = np.mean(predicted_negatives)
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['sensitivity'] = sensitivity
    results['specificity'] = specificity
    results['positive_likelihood'] = positive_likelihood
    results['negative_likelihood'] = negative_likelihood
    results['false_positive_rate'] = false_positive_rate
    results['false_negative_rate'] = false_negative_rate
    results['true_positive_rate'] = true_positive_rate
    results['true_negative_rate'] = true_negative_rate
    results['positive_predictive_value'] = positive_predictive_value
    results['negative_predictive_value'] = negative_predictive_value
    
    return results

# --------------------------------------------------
'''
def get_values_for_learning_curve_log_reg(
        test_fraction, 
        max_train_size,
        X_np,
        y_np,
        rand_state):
    
    """
    Function to derive the accuracy for incremental test sizes from 10, to
    the maximum training size (whole data set) in increments of 10.
    """
    
    #Loop through increasing training set sizes
    # Set up list to collect results
    results_training_size = []
    results_accuracy = []
    
    for train_size in range(250, max_train_size, 500): #previously max_training_size instead of max_train_size but not an input param, so changed to match? previously interval of 10, now 250 (was 500, 1000)
        replicate_accuracy = []
        for replicate in range(10):
            # Split data into training and test
            #added Shuffle = True 
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size = test_fraction, shuffle=True, random_state = rand_state)

            # Reduce training set size (use np random choice for random index values)
            selection_index = np.random.choice(
                max_train_size, train_size, replace=False) #previously max_training_size instead of max_train_size but not an input param, so changed to match?
            X_train = X_train[selection_index]
            y_train = y_train[selection_index]

            # Standardise
            X_train_std, X_test_std = standardise_data(X_train, X_test)

            # Fit model
            model = LogisticRegression(solver='lbfgs')
            #print('X_train_std:')
            #print(X_train_std)
            #print('')
            #print('y_train')
            #print(y_train)
            model.fit(X_train_std,y_train)

            # Predict test set
            y_pred_test = model.predict(X_test_std)

            # Get accuracy and record results
            accuracy = np.mean(y_pred_test == y_test)
            replicate_accuracy.append(accuracy)
        
        results_accuracy.append(np.mean(replicate_accuracy))
        results_training_size.append(train_size)
        
    return results_accuracy, results_training_size
'''

def get_values_for_learning_curve_log_reg(
        test_fraction, 
        max_train_size,
        X_np,
        y_np,
        rand_state,
        sample_size_increment):
    
    """
    Function to derive the accuracy for incremental test sizes from 10, to
    the maximum training size (whole data set) in increments of 10.
    """
    
    #Loop through increasing training set sizes
    # Set up list to collect results
    results_training_size = []
    results_accuracy = []
    
    for train_size in range(sample_size_increment, max_train_size, sample_size_increment): #previously max_training_size instead of max_train_size but not an input param, so changed to match? previously interval of 10, now 250 (was 500, 1000)
        replicate_accuracy = []
        for replicate in range(10):
            # Split data into training and test
            #added Shuffle = True 
            X_train, X_test, y_train, y_test = train_test_split(
                X_np, y_np, test_size = test_fraction, shuffle=True, random_state = rand_state)

            # Reduce training set size (use np random choice for random index values)
            selection_index = np.random.choice(
                max_train_size, train_size, replace=False) #previously max_training_size instead of max_train_size but not an input param, so changed to match?
            X_train = X_train[selection_index]
            y_train = y_train[selection_index]

            # Standardise
            X_train_std, X_test_std = standardise_data(X_train, X_test)

            # Fit model
            model = LogisticRegression(solver='lbfgs')
            #print('X_train_std:')
            #print(X_train_std)
            #print('')
            #print('y_train')
            #print(y_train)
            model.fit(X_train_std,y_train)

            # Predict test set
            y_pred_test = model.predict(X_test_std)

            # Get accuracy and record results
            accuracy = np.mean(y_pred_test == y_test)
            replicate_accuracy.append(accuracy)
        
        results_accuracy.append(np.mean(replicate_accuracy))
        results_training_size.append(train_size)
        
    return results_accuracy, results_training_size

# --------------------------------------------------

#required for training size chart to run. require min data set size of 250
def round_nearest_250(n):
    number_of_250s = n // 250
    nearest_250 = number_of_250s * 250
    return nearest_250

# --------------------------------------------------

def plot_learning_curve(results_training_size, results_accuracy):
    """
    Function to plot the learning curve based on outputs from the 
    get_values_for_learning_curve_log_reg function
    """
    #Plot the learning curve, inc. moving average (mean of 5 points). 
    #Moving averages can help show trends when data is noisy.
    #Calculate moving average (of last 5 points) with np.convolve
    moving_average = np.convolve(results_accuracy, np.ones((5,))/5, mode='valid')
    x_moving_average = results_training_size[2:-2] # Include offset to centre mean

    plt.scatter(results_training_size, results_accuracy, 
             label='Accuracy')

    plt.plot(x_moving_average, moving_average,
            label='Moving average',
            color='orange',
            linewidth=3)

    plt.xlabel('Training set size')
    plt.ylabel('Test set accuracy')
    plt.legend()
    plt.title("Scatter plot of Training set size vs. Test set accuracy.")
    plt.grid(True)
    plt.figtext(0.5, 0.0001, "Interpretation: look for diminishing return on accuracy for increased sample size. Adjust sample size if limited computational power.", ha="center", va="baseline", fontsize=6)
    plt.show()
    return (moving_average, x_moving_average)

# --------------------------------------------------

#Round input int to nearest 10, default to 30 if int <30 
#check lowest value in moving_average (from plot_learning_curve function) is 30
#check the max value in moving_average (from plot_learning curve function is sample size - 20)
#when running real data
def round_nearest_10(x, base=10):
    if x <30:
        return 30
    else:
        return (base * round(x/base)) - 20

# --------------------------------------------------
''' NEW ALTERNATIVE PASTED BELOW THIS ONE
def plot_learning_curve_with_sample_size_new(results_training_size, results_accuracy, sample_size):
    """
    Function to plot the learning curve based on outputs from the 
    get_values_for_learning_curve_log_reg function
    """
    #Plot the learning curve, inc. moving average (mean of 5 points). 
    #Moving averages can help show trends when data is noisy.
    #Calculate moving average (of last 5 points) with np.convolve
    moving_average = np.convolve(results_accuracy, np.ones((5,))/5, mode='valid')
    x_moving_average = results_training_size[2:-2] # Include offset to centre mean
    
    fig, ax = plt.subplots() #new line

    #fig = plt.figure(figsize=(6,5)) #defines the dimensions of the figure (horizontal, vertical)

    ax.scatter(results_training_size, results_accuracy, 
             label='Accuracy')

    ax.plot(x_moving_average, moving_average,
            label='Moving average',
            color='orange',
            linewidth=3)
    
    index = x_moving_average.index(round_nearest_10(sample_size)-10)
    sample_size_x_mov_avg = x_moving_average[index]
    sample_size_mov_avg = moving_average[index]
    
    ax.scatter(sample_size_x_mov_avg,sample_size_mov_avg, label="selected sample size", color="red", marker="x", s=200, linewidths=3)

    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test set accuracy')
    ax.set_title("Scatter plot of Training set size vs. Test set accuracy.")
    ax.legend()
    ax.grid(True)
    ax.tick_params(labelsize = 14)
    plt.figtext(0.5, 0.0001, "Interpretation: look for diminishing return on accuracy for increased sample size. Adjust sample size if limited computational power.", ha="center", va="baseline", fontsize=6)
    
    # Set background color of Figure
    fig.patch.set_facecolor('white')
    # Set transparency of figure
    fig.patch.set_alpha(1)
    
    plt.savefig(f"{dna_assets_path}/chart001_FigSelectedSampleSizeAccuracy.png", bbox_inches='tight')
    
    plt.show()
    fig.tight_layout()
    
    return(plt)
'''
'''
def plot_learning_curve_with_sample_size_new(results_training_size, results_accuracy, sample_size):
    """
    Function to plot the learning curve based on outputs from the 
    get_values_for_learning_curve_log_reg function
    """
    #Plot the learning curve, inc. moving average (mean of 5 points). 
    #Moving averages can help show trends when data is noisy.
    #Calculate moving average (of last 5 points) with np.convolve
    moving_average = np.convolve(results_accuracy, np.ones((5,))/5, mode='valid')
    x_moving_average = results_training_size[2:-2] # Include offset to centre mean
    
    fig, ax = plt.subplots() #new line

    #fig = plt.figure(figsize=(6,5)) #defines the dimensions of the figure (horizontal, vertical)

    ax.scatter(results_training_size, results_accuracy, 
             label='Accuracy')

    ax.plot(x_moving_average, moving_average,
            label='Moving average',
            color='orange',
            linewidth=3)
    
    #index = x_moving_average.index(round_nearest_10(sample_size)-500)
    #index = x_moving_average.index(results_training_size)
    index = [results_training_size.index(item) for item in x_moving_average]
    #sample_size_x_mov_avg = x_moving_average[index]
    sample_size_x_mov_avg = [x_moving_average[num] for num in index[:-3]]
    #sample_size_x_mov_avg = x_moving_average[index]
    sample_size_mov_avg = [moving_average[num] for num in index[:-3]]
    
    ax.scatter(sample_size_x_mov_avg,sample_size_mov_avg, label="selected sample size", color="red", marker="x", s=200, linewidths=3)

    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test set accuracy')
    ax.set_title("Scatter plot of Training set size vs. Test set accuracy.")
    ax.legend()
    ax.grid(True)
    ax.tick_params(labelsize = 14)
    plt.figtext(0.5, 0.0001, "Interpretation: look for diminishing return on accuracy for increased sample size. Adjust sample size if limited computational power.", ha="center", va="baseline", fontsize=6)
    
    # Set background color of Figure
    fig.patch.set_facecolor('white')
    # Set transparency of figure
    fig.patch.set_alpha(1)
    
    plt.savefig(f"{dna_assets_path}/chart001_FigSelectedSampleSizeAccuracy.png", bbox_inches='tight')
    
    plt.show()
    fig.tight_layout()
    
    return(plt)
'''

#revised function - needs replacing in above function section
def plot_learning_curve_with_sample_size_new(results_training_size, results_accuracy, sample_size):
    """
    Function to plot the learning curve based on outputs from the 
    get_values_for_learning_curve_log_reg function
    """
    #Plot the learning curve, inc. moving average (mean of 5 points). 
    #Moving averages can help show trends when data is noisy.
    #Calculate moving average (of last 5 points) with np.convolve
    moving_average = np.convolve(results_accuracy, np.ones((5,))/5, mode='valid')
    x_moving_average = results_training_size[2:-2] # Include offset to centre mean
    
    fig, ax = plt.subplots() #new line

    #fig = plt.figure(figsize=(6,5)) #defines the dimensions of the figure (horizontal, vertical)

    ax.scatter(results_training_size, results_accuracy, 
             label='Accuracy')

    ax.plot(x_moving_average, moving_average,
            label='Moving average',
            color='orange',
            linewidth=3)
    
    #round down the given sample size to the nearest 250
    rounded_sample_size = round_nearest_250(sample_size)
    if rounded_sample_size not in range(250, max_training_size, 250):
        result = range(250, max_training_size, 500)[-1]
    else:
        result = rounded_sample_size
    
    #locate accuracy for the rounded sample size to visually display on chart as a X
    accuracy_index_chosen_sample_size = results_training_size.index(result)
    result_chosen_sample_size = results_accuracy[accuracy_index_chosen_sample_size]

    ax.scatter(result,result_chosen_sample_size, label="selected sample size", color="red", marker="x", s=200, linewidths=3)

    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test set accuracy')
    ax.set_title("Scatter plot of Training set size vs. Test set accuracy.")
    ax.legend()
    ax.grid(True)
    ax.tick_params(labelsize = 14)
    plt.figtext(0.5, 0.0001, "Interpretation: look for diminishing return on accuracy for increased sample size. Adjust sample size if limited computational power.", ha="center", va="baseline", fontsize=6)
    
    # Set background color of Figure
    fig.patch.set_facecolor('white')
    # Set transparency of figure
    fig.patch.set_alpha(1)
    
    plt.savefig(f"{dna_assets_path}/chart001_FigSelectedSampleSizeAccuracy.png", bbox_inches='tight')
    
    plt.show()
    fig.tight_layout()
    
    return(plt)

# --------------------------------------------------

def plot_learning_curve_with_sample_size(results_training_size, results_accuracy, sample_size):
    """
    Function to plot the learning curve based on outputs from the 
    get_values_for_learning_curve_log_reg function
    """
    #Plot the learning curve, inc. moving average (mean of 5 points). 
    #Moving averages can help show trends when data is noisy.
    #Calculate moving average (of last 5 points) with np.convolve
    moving_average = np.convolve(results_accuracy, np.ones((5,))/5, mode='valid')
    x_moving_average = results_training_size[2:-2] # Include offset to centre mean
    
    fig, ax = plt.subplots() #new line

    fig = plt.figure(figsize=(6,5)) #defines the dimensions of the figure (horizontal, vertical)

    plt.scatter(results_training_size, results_accuracy, 
             label='Accuracy')

    plt.plot(x_moving_average, moving_average,
            label='Moving average',
            color='orange',
            linewidth=3)
    
    index = x_moving_average.index(round_nearest_10(sample_size))
    sample_size_x_mov_avg = x_moving_average[index]
    sample_size_mov_avg = moving_average[index]
    
    plt.scatter(sample_size_x_mov_avg,sample_size_mov_avg, label="selected sample size", color="red", marker="x", s=200, linewidths=3)

    plt.xlabel('Training set size')
    plt.ylabel('Test set accuracy')
    plt.title("Scatter plot of Training set size vs. Test set accuracy.")
    plt.legend()
    plt.grid(True)
    plt.tick_params(labelsize = 14)
    plt.figtext(0.5, 0.0001, "Interpretation: look for diminishing return on accuracy for increased sample size. Adjust sample size if limited computational power.", ha="center", va="baseline", fontsize=6)
    plt.show()
    fig.tight_layout()
    
    return(fig)

# --------------------------------------------------

def add_performance_metrics_to_summary(summary_df, dict_metrics_summary, dict_metrics, model_variant_name):
    
    """
    Function to add the identified model performance metrics of precision, 
    recall, f1 and specificity, to the provided summary_df. 
    Use this function after each model, and subsequent methods to boost predictive
    power, is run, to add the performance metrics of each model variant to the 
    growing summary df. 
    """
    
    list_of_metrics_measures = []
    list_of_metrics_labels = []
    for num in range(len(dict_metrics_summary)):
        list_of_metrics_measures.append(dict_metrics[dict_metrics_summary[num]])
        list_of_metrics_labels.append(dict_metrics_summary[num])
    data_tuples = list(zip(list_of_metrics_labels,list_of_metrics_measures))
    temp_df = pd.DataFrame(data_tuples, columns=['performance_metrics',model_variant_name])
    temp_df.set_index('performance_metrics', inplace=True)
    new_df = pd.merge(summary_df, temp_df, how="left", left_index=True, right_index=True)
    return new_df

# --------------------------------------------------

#test cell to check incorporating coeff lists to average
def run_log_reg_within_k_fold_and_output_accuracy_scores(
        num_splits,
        random_state,
        X,
        X_np,
        y_np,
        num_bins,
        model_name,
        ):
    
    """
    New function to use, replacing previous function of same name. 
    This revision has less repeated lines of code and also outputs both 
    test set and training set mean performance measures as 2 separate dictionaries
    from which the summary of performance will later be created for this baseline 
    log reg
    """
    
    """
    The following code:
    - sets up lists to hold the results of each k-fold split
    - Sets up the splits using sklearn's StratifiedKFold method
    - Trains a logistic regression model, and tests its fit, for each k-fold split
    - Add each k-fold training / test accuracy to the lists
    - generates the observed and predicted probabilities for each split - later used for ROC curve
    """
    
    # Set up lists to hold results for each k-fold run
    training_acc_results = []
    test_acc_results = []

    #extended accuracy lists
    list_all_performance_metrics = [
        "observed_positive_rate",
        "observed_negative_rate",
        "predicted_positive_rate", 
        "predicted_negative_rate", 
        "accuracy",
        "precision",
        "recall",
        "f1",
        "sensitivity",
        "specificity",
        "positive_likelihood",
        "negative_likelihood",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive_rate",
        "true_negative_rate",
        "positive_predictive_value",
        "negative_predictive_value"
        ]
    
    #create a nested dictionary containing all performance metrics for both train and test
    #as keys e.g. ["train"]["f1"]. All values are place holders at this point (empty lists)
    list_outer_dict_levels = ["train", "test"]    
    dict_train_test_metrics = {level: {metric:list() for metric in list_all_performance_metrics} for level in list_outer_dict_levels}

    # Set up splits
    number_of_splits = num_splits #was 10
    
    #skf = StratifiedKFold(n_splits = number_of_splits) #original logistic regression stratified k-fold split code
    skf = StratifiedKFold(n_splits = number_of_splits, shuffle=True, random_state = random_state) #from Random Forest example, should ensure repeatable splits
    skf.get_n_splits(X_np, y_np) 

    # Set up results lists (to get results from each run)
    #These will be used to create the reliability plot
    results_model_probability = []
    results_fraction_positive = []

    #Define number of bins - used for reliability plot
    number_of_bins = num_bins

    # Set up lists for observed and predicted - used for the ROC Curve
    observed = []
    predicted_proba = []
    predicted = [] #CHECK why is this list present, looks empty?

    #set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
    coeff_list_of_lists = []
    
    #set up list to store the auc for each split
    replicate_auc = []
    
    counter = 0
    # Loop through the k-fold splits
    for train_index, test_index in skf.split(X_np, y_np):
        counter +=1
        
        # Get X and Y train/test
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        
        # Standardise X data
        X_train_std, X_test_std = standardise_data(X_train, X_test)
        
        # Set up and fit model
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train_std,y_train)
        
        #get coefficients for this fold
        """
        Weights with higher negative numbers mean that that feature correlates with 
        reduced chance of DNA (outcome variable, y). 
        Weights with higher positive numbers mean that that feature correlates with 
        increased chance of DNA (outcome variable, y). 
        Those weights with values closer to zero (either positive or negative) have 
        less influence in the model.
        We access model weights via the model coef_ attribute. 
        A model may predict >1 outcome label, in which case we have weights for each label. 
        This model only predicts a single label (DNA or not), so the weights are found 
        in the first element ([0]) of the coef_ attribute.
        """
        co_eff = model.coef_[0]
        coeff_list_of_lists.append(co_eff)
      
        # Predict training and test set labels, using standardised data
        y_pred_train = model.predict(X_train_std)
        y_pred_test = model.predict(X_test_std)
        
        # Get test set proabilities - reliability curve 
        y_calibrate_probabilities = model.predict_proba(X_test_std)[:,1]
        
        #Get ROC AUC for this split, append to list of AUC's
        auc = roc_auc_score(y_test, y_calibrate_probabilities)
        replicate_auc.append(auc)
        
        # Get calibration curve (use quantile to make sure all bins exist) - reliability curve
        fraction_pos, model_prob = calibration_curve(
            y_test, y_calibrate_probabilities, 
            n_bins=number_of_bins,
            strategy='quantile') 
        
        # record run results - reliability curve
        results_model_probability.append(model_prob)
        results_fraction_positive.append(fraction_pos)
        
        #loop to update nested dict
        dict_of_splits = {"train": [y_train, y_pred_train], "test": [y_test, y_pred_test]}
        temp_dict = {}
        for level in list_outer_dict_levels:
            temp_dict[level] = calculate_accuracy(dict_of_splits[level][0], dict_of_splits[level][1])
        
        #append performance metric to the dictionary of performance values for 
        #each split (train / test), taken from the temp_dict created by the 
        #calculate_accuracy function
        for level in list_outer_dict_levels:
            for metric in list_all_performance_metrics:
                dict_train_test_metrics[level][metric].append(temp_dict[level][metric])
        
        # Calculate accuracy of training and test sets
        accuracy_train = np.mean(y_pred_train == y_train)
        accuracy_test = np.mean(y_pred_test == y_test)
        
        # Add accuracy to lists
        training_acc_results.append(accuracy_train)
        test_acc_results.append(accuracy_test)
        
        #ROC code:
        # Get predicted probabilities
        y_probs = model.predict_proba(X_test_std)[:,1] #check that 1 relates to attended status
        y_class = model.predict(X_test_std) #predict the actual class label
        observed.append(y_test)
        predicted_proba.append(y_probs)
        
        # Print accuracy
        accuracy = np.mean(y_class == y_test)
        print (f'Run {counter}, accuracy: {accuracy:0.3f}')
    
    # Transfer results to dataframe
    results_auc = pd.DataFrame(columns=[model_name]) #previously had "auc_" prefix in f-string NB was columns=['model_name])
    mean_auc_across_all_splits = np.mean(replicate_auc)
    results_auc.loc[0] = mean_auc_across_all_splits
    
    #Calculate the mean of the various accuracy scores created in previous step:
    dict_mean_train_test_metrics = {level: {f"mean_{metric}":list() for metric in list_all_performance_metrics} for level in list_outer_dict_levels}
    
    for level in list_outer_dict_levels:
        for metric in list_all_performance_metrics:
            dict_mean_train_test_metrics[level][f"mean_{metric}"] = np.mean(dict_train_test_metrics[level][metric])
    
    dict_mean_test_performance = dict_mean_train_test_metrics["test"]
    dict_mean_train_performance = dict_mean_train_test_metrics["train"]

    #average the coefficient values produced for each fold, to get to a single array of average coefficients. 
    #these are in the same order as the columns in X
    average_coefficients = np.mean(coeff_list_of_lists, 0)
    
    co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(average_coefficients, X)

    return(dict_mean_test_performance, 
           dict_mean_train_performance,
           observed, 
           predicted_proba, 
           results_model_probability, 
           results_fraction_positive,
           skf,
           results_auc,
           co_eff_df, 
           co_eff_df_reduced_chance_dna, 
           co_eff_df_higher_chance_dna
           )

# --------------------------------------------------

#function to produce the ROC Area Under the Curve (AUC)
def get_roc(num_splits,
            observed,
            predicted_proba,
            results_model_probability,
            results_fraction_positive
            ):
    
    #Scikit-Learn’s ROC method will automatically test the rate of true postive 
    #rate (tpr) and false positive rate (fpr) at different thresholds of 
    #classification. It will return tpr, fpr for each threshold tested. We also 
    #use Scikit-Learn’s method for caluclating the area under the curve.
    
    """
    #Reciever Operator Characteristic (ROC) Curve
    The ROC curve allows us to better understand the trade-off between 
    sensitivity (the ability to detect positives of a certain class) and 
    specificity (the ability to detect negatives of a certain class). 
    The area under the ROC curve is also often used to compare different 
    models: a higher Area Under Curve (AUC) is frequently the sign of a 
    better model.

    ROC curve is created by plotting the true positive rate (TPR) against 
    the false positive rate (FPR) at various threshold settings. 
    The true-positive rate is also known as sensitivity or recall. 
    The false-positive rate can be calculated as (1 − specificity).
    """
    
    # Set up lists for results
    k_fold_fpr = [] # false positive rate
    k_fold_tpr = [] # true positive rate
    k_fold_thresholds = [] # threshold applied
    k_fold_auc = [] # area under curve

    # Loop through k fold predictions and get ROC results 
    for i in range(number_of_splits):
        # Get fpr, tpr and thresholds foir each k-fold from scikit-learn's ROC method
        fpr, tpr, thresholds = roc_curve(observed[i], predicted_proba[i])
        # Use scikit-learn's method for calulcating auc
        roc_auc = auc(fpr, tpr)
        # Store results
        k_fold_fpr.append(fpr)
        k_fold_tpr.append(tpr)
        k_fold_thresholds.append(thresholds)
        k_fold_auc.append(roc_auc)
        # Print auc result
        print (f'Run {i} AUC {roc_auc:0.4f}')

    # Show mean area under curve  
    mean_auc = np.mean(k_fold_auc)
    sd_auc = np.std(k_fold_auc)
    print (f'\nMean AUC: {mean_auc:0.4f}')
    print (f'SD AUC: {sd_auc:0.4f}')

    #Plot ROCs
    fig = plt.figure(figsize=(6,6))

    ax1 = fig.add_subplot()
    for i in range(5):
        ax1.plot(k_fold_fpr[i], k_fold_tpr[i], color='orange')
    ax1.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operator Characteristic Curve')
    text = f'Mean AUC: {mean_auc:.3f}'
    ax1.text(0.64,0.07, text, 
             bbox=dict(facecolor='white', edgecolor='black'))
    plt.grid(True)
    plt.show()


    #Reliability curve 
    # Convert results to DataFrame to enable plotting reliability curve
    results_model_probability = pd.DataFrame(results_model_probability)
    results_fraction_positive = pd.DataFrame(results_fraction_positive)

    # Add individual k-fold runs
    for run in range(number_of_splits):
        plt.plot(results_model_probability.loc[run],
                 results_fraction_positive.loc[run],
                 linestyle='--',
                 linewidth=0.75,
                 color='0.5')
        
    # Add mean
    plt.plot(results_model_probability.mean(axis=0),
             results_fraction_positive.mean(axis=0),
             linestyle='-',
             linewidth=2,
             color='darkorange',
             label='mean')

    # Add diagonal
    plt.plot([0,1],[0,1],
             linestyle='--',
             label='theoretical')

    plt.xlabel('Model probability')
    plt.ylabel('Fraction positive')

    plt.title('Reliability plot')

    plt.grid()
    plt.legend()
    plt.show()

# --------------------------------------------------

#Feature selection
def univariate_statistical_selection(
        data,
        skf
        ):
    
    """
    Feature Selection: Univariate Statistical Selection
    Why: 
        Because if you are limited by computational power you can use this method
    to identify the features which, in isolation, appear to be correlated with the 
    target variable. 

    Pros:
        + it is simple
        + it is fast
            
    Cons:
        - it may miss features which have little effect alone, but which are 
        infliential when combined
        - it may include features which are highly correlated which could be 
        reduced to choosing just one of the highly correlated features 

    """
    #Re-establish X and y variables as DataFrames and not NumPy arrays
    # X = all 'data' except the 'encoded_attended' column
    X = data.drop('encoded_attend_status_with_missing_Attended',axis=1) 
    # y = 'encoded_attended' column from 'data'
    y = data['encoded_attend_status_with_missing_Attended'] 

    #Calculate correlation coefficients
    features = list(X)
    correlation = []
    significance = []
    for feature in features:
        correl = pearsonr(X[feature].values, y.values)
        correlation.append(correl[0])
        significance.append(correl[1])
    df = pd.DataFrame()
    df['feature'] = features
    df['correlation'] = correlation
    df['abs_correlation'] = np.abs(correlation)
    df['significance'] = significance
    df['significant'] = df['significance'] < 0.05 # Label those P<0.01
    df.sort_values(by='abs_correlation', ascending=False, inplace=True)

    ordered_features = list(df['feature'])

    #Testing our selected features
    #what is happening in this section?:
    #    incrementally adding features to list of features to use (chosen in order 
    #    of their correlation coefficients) and see the effect on model accuracy 
    #    and ROC AUC as measured by k-fold stratification

    # Create list to store accuracies
    accuracy_by_feature_number = []
    roc_auc_by_feature_number = []

    # Loop through feature list
    number_of_features = len(ordered_features)
    for i in range(number_of_features):
        # print ("{0} features of {1}".format(i, number_of_features))
        features_to_use = ordered_features[0:i+1]
        X_selected = X[features_to_use]
        
        # Convert to NumPy (needed for k-fold method)
        # Convert DataFrames to NumPy arrays
        X_np = X_selected.values
        y_np = y.values
        
        # Set up lists to hold results for each k-fold run
        test_acc_results = []
        test_auc_results = []

        # Set up splits
        #number_of_splits = 10
        #skf = StratifiedKFold(n_splits = number_of_splits)
        #skf.get_n_splits(X_np, y)

        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            # Get X and Y train/test
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y[train_index], y[test_index]  

            # Get X and Y train/test
            X_train_std, X_test_std = standardise_data(X_train, X_test)

            # Set up and fit model
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train_std,y_train)

            # Predict test set labels
            y_pred_test = model.predict(X_test_std)
            
            # Calculate accuracy of test sets
            accuracy_test = np.mean(y_pred_test == y_test)
            test_acc_results.append(accuracy_test)
            
            # Get ROC AUC
            probabilities = model.predict_proba(X_test_std)
            probabilities = probabilities[:, 1] # Probability of 'attended' class
            fpr, tpr, thresholds = roc_curve(y_test, probabilities)
            roc_auc = auc(fpr, tpr)
            test_auc_results.append(roc_auc)      
            
        # Add mean accuracy and AUC to record of accuracy by feature number
        accuracy_by_feature_number.append(np.mean(test_acc_results))
        roc_auc_by_feature_number.append(np.mean(test_auc_results)) 
        
    #Accuracy and ROC by cumulative number of features
    chart_x = list(range(1, number_of_features + 1))
    
    fig = plt.figure(figsize=(6,5)) #defines the dimensions of the figure (horizontal, vertical)
    
    plt.plot(chart_x, accuracy_by_feature_number,
            label = 'Accuracy')

    plt.plot(chart_x, roc_auc_by_feature_number,
            label = 'ROC AUC')

    plt.xlabel('Number of features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title("Chart to show the relationship between the number \nof features used in the model and the accuracy of \npredicting a patient who Does Not Attend their appointment")

    plt.tick_params(labelsize = 14)

    # Add a footnote below and to the right side of the chart
    #Reminder: ROC AUC is a measure of the balance between true positive & false 
    #positives as the threshold to classify a case as a positive is changed.
    plt.annotate("Interpretation: look for accuracy and ROC AUC peak by number of features. \nLimit to this number of features if limited by computational power.",
                xy = (-0.01, -0.25),
                xycoords='axes fraction',
                ha='left',
                va="center",
                fontsize=8)

    fig.tight_layout()

    plt.show()
    return(fig)

# --------------------------------------------------

#Feature expansion
def add_polynomial_features_original(
        X,
        X_np,
        y_np,
        num_splits
        ):
    
    """
    Feature Expansion
    Why: 
        to increase accuracy of the prediction that the model makes
    how:
        Models such as logistic regression do not incorporate complex interactions 
        between features. If 2 features produce more than an additive effect, this 
        will not be fitted in logistic regression. In order to allow for feature 
        interaction we need to add terms that create new features by producing the 
        product of each product pair.
        When we use polynomial expansion of features, we create new features that 
        are the product of two features. 
        For example if we had 3 features, A, B and C, a full polynomial expansion 
        would produce the following extra features:
        A.A, A.B, A.C
        B.A, B.B, B.C
        C.A, C.B, C.C
        
        This is reduced in 2 ways:
            1) duplicate terms are removed (A.B and B.A are the same, so we only need A.B)
            2) The interaction_only argument is used to remove powers of single features (e.g. A.A)

    Pros:
        increases accuracy
        
    Cons:
        model may start to over-fit to the training data
        
    Dealing with over-fitting
    If over-fitting identified as an issue following use of polynomial feature expansion
    it can be handled in one (or both) of two ways:

    1) Increase the regularisation strength in the model (reduce the value of C in 
    the logistic regression model)
                                                          
    2) Use feature selection to pick only the most important features (which now 
    may include polynomial features)
    """
    
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    #check shape of data sets (first value is number of samples, second value is number of features)
    print ('Shape of X:', X_np.shape)
    print ('Shape of X_poly:', X_poly.shape)

    #TRAINING AND TESTING NORMAL AND POLYNOMIAL MODELS
    reg_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    #Set up lists to hold results
    training_acc_results = []
    test_acc_results = []
    training_acc_results_poly = []
    test_acc_results_poly = []

    #set up lists for normal (non-polynomial) model
    #training
    training_accuracy_results = []
    training_precision_results = []
    training_recall_results = []
    training_f1_results = []
    training_specificity_results = []
    training_predicted_positive_rate_results = []
    training_observed_positive_rate_results = []
    training_roc_auc_results = []
    
    #test
    test_accuracy_results = []
    test_precision_results = []
    test_recall_results = []
    test_f1_results = []
    test_specificity_results = []
    test_predicted_positive_rate_results = []
    test_observed_positive_rate_results = []
    test_roc_auc_results = []

    #set up lists for polynomial model
    #training
    training_accuracy_results_poly = []
    training_precision_results_poly = []
    training_recall_results_poly = []
    training_f1_results_poly = []
    training_specificity_results_poly = []
    training_predicted_positive_rate_results_poly = []
    training_observed_positive_rate_results_poly = []
    training_roc_auc_results_poly = []

    #test
    test_accuracy_results_poly = []
    test_precision_results_poly = []
    test_recall_results_poly = []
    test_f1_results_poly = []
    test_specificity_results_poly = []
    test_predicted_positive_rate_results_poly = []
    test_observed_positive_rate_results_poly = []
    test_roc_auc_results_poly = []

    #Set up splits
    skf = StratifiedKFold(n_splits = num_splits)
    skf.get_n_splits(X_np, y_np)
    skf.get_n_splits(X_poly, y_np)

    #Set up model type

    for reg in reg_values:
        # Show progress
        print(reg, end=' ')
        
        # Set up lists for results for each of k splits
        training_k_results = []
        test_k_results = []
        training_k_results_poly = []
        test_k_results_poly = []

        #lists for each performance metric to capture - need to add similar lists below for poly versions, and outside loop to capture mean of the splits.
        #train - normal
        training_k_accuracy_results = []
        training_k_precision_results = []
        training_k_recall_results = []
        training_k_f1_results = []
        training_k_specificity_results = []
        training_k_predicted_positive_rate_results = []
        training_k_observed_positive_rate_results = []
        training_k_roc_auc_results = []
        
        #test - normal
        test_k_accuracy_results = []
        test_k_precision_results = []
        test_k_recall_results = []
        test_k_f1_results = []
        test_k_specificity_results = []
        test_k_predicted_positive_rate_results = []
        test_k_observed_positive_rate_results = []
        test_k_roc_auc_results = []
        
        #lists to capture poly metrics
        #training - poly
        training_k_accuracy_results_poly = []
        training_k_precision_results_poly = []
        training_k_recall_results_poly = []
        training_k_f1_results_poly = []
        training_k_specificity_results_poly = []
        training_k_predicted_positive_rate_results_poly = []
        training_k_observed_positive_rate_results_poly = []
        training_k_roc_auc_results_poly = []
        
        #test - poly
        test_k_accuracy_results_poly = []
        test_k_precision_results_poly = []
        test_k_recall_results_poly = []
        test_k_f1_results_poly = []
        test_k_specificity_results_poly = []
        test_k_predicted_positive_rate_results_poly = []
        test_k_observed_positive_rate_results_poly = []
        test_k_roc_auc_results_poly = []
        
        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            
            # Normal (non-polynomial model)
            
            # Get X and Y train/test
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]
            # Standardise X data
            X_train_std, X_test_std = standardise_data(X_train, X_test)
            # Fit model with regularisation (C)
            model = LogisticRegression(C=reg, solver='lbfgs', max_iter=1000)
            model.fit(X_train_std,y_train)
            # Predict training and test set labels
            y_pred_train = model.predict(X_train_std)
            y_pred_test = model.predict(X_test_std)
            # Calculate accuracy of training and test sets
            accuracy_train = np.mean(y_pred_train == y_train)
            accuracy_test = np.mean(y_pred_test == y_test)


            #enter code to calculate the performance metrics and add to list for this kfold
            #training performance - normal (non-poly)
            train_results = calculate_accuracy(y_train, y_pred_train)
            test_results = calculate_accuracy(y_test, y_pred_test)
            
            training_k_accuracy_results.append(sklearn.metrics.accuracy_score(y_train, y_pred_train))
            training_k_precision_results.append(sklearn.metrics.precision_score(y_train, y_pred_train))
            training_k_recall_results.append(sklearn.metrics.recall_score(y_train, y_pred_train))
            training_k_f1_results.append(sklearn.metrics.f1_score(y_train, y_pred_train))
            training_k_specificity_results.append(train_results["specificity"])
            training_k_predicted_positive_rate_results.append(train_results["predicted_positive_rate"])
            training_k_observed_positive_rate_results.append(train_results["observed_positive_rate"])
            training_k_roc_auc_results.append(roc_auc_score(y_train, y_pred_train))
            
            #test performance - non-poly
            test_k_accuracy_results.append(sklearn.metrics.accuracy_score(y_test, y_pred_test))
            test_k_precision_results.append(sklearn.metrics.precision_score(y_test, y_pred_test))
            test_k_recall_results.append(sklearn.metrics.recall_score(y_test, y_pred_test))
            test_k_f1_results.append(sklearn.metrics.f1_score(y_test, y_pred_test))
            test_k_specificity_results.append(test_results["specificity"])
            test_k_predicted_positive_rate_results.append(test_results["predicted_positive_rate"])
            test_k_observed_positive_rate_results.append(test_results["observed_positive_rate"])
            test_k_roc_auc_results.append(roc_auc_score(y_test, y_pred_test))

            # Record accuracy for each k-fold split
            training_k_results.append(accuracy_train)
            test_k_results.append(accuracy_test)
            
            # Polynomial model (same as above except use X with polynomial features)
            
            # Get X and Y train/test
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]
            # Standardise X data
            X_train_std, X_test_std = standardise_data(X_train, X_test)
            # Fit model with regularisation (C)
            model = LogisticRegression(C=reg, solver='lbfgs', max_iter=1000)
            model.fit(X_train_std,y_train)
            # Predict training and test set labels
            y_pred_train = model.predict(X_train_std)
            y_pred_test = model.predict(X_test_std)
            # Calculate accuracy of training and test sets
            accuracy_train = np.mean(y_pred_train == y_train)
            accuracy_test = np.mean(y_pred_test == y_test)

            #enter code to calculate the performance metrics and add to list for this kfold for poly expanded
            train_results_poly = calculate_accuracy(y_train, y_pred_train)
            test_results_poly = calculate_accuracy(y_test, y_pred_test)
            
            #training performance - poly
            training_k_accuracy_results_poly.append(sklearn.metrics.accuracy_score(y_train, y_pred_train))
            training_k_precision_results_poly.append(sklearn.metrics.precision_score(y_train, y_pred_train))
            training_k_recall_results_poly.append(sklearn.metrics.recall_score(y_train, y_pred_train))
            training_k_f1_results_poly.append(sklearn.metrics.f1_score(y_train, y_pred_train))
            training_k_specificity_results_poly.append(train_results_poly["specificity"])
            training_k_predicted_positive_rate_results_poly.append(train_results_poly["predicted_positive_rate"])
            training_k_observed_positive_rate_results_poly.append(train_results_poly["observed_positive_rate"])
            training_k_roc_auc_results_poly.append(roc_auc_score(y_train, y_pred_train))
            
            #test performance - poly
            test_k_accuracy_results_poly.append(sklearn.metrics.accuracy_score(y_test, y_pred_test))
            test_k_precision_results_poly.append(sklearn.metrics.precision_score(y_test, y_pred_test))
            test_k_recall_results_poly.append(sklearn.metrics.recall_score(y_test, y_pred_test))
            test_k_f1_results_poly.append(sklearn.metrics.f1_score(y_test, y_pred_test))
            test_k_specificity_results_poly.append(test_results_poly["specificity"])
            test_k_predicted_positive_rate_results_poly.append(test_results_poly["predicted_positive_rate"])
            test_k_observed_positive_rate_results_poly.append(test_results_poly["observed_positive_rate"])
            test_k_roc_auc_results_poly.append(roc_auc_score(y_test, y_pred_test))

            # Record accuracy for each k-fold split
            training_k_results_poly.append(accuracy_train)
            test_k_results_poly.append(accuracy_test)
            
        # Record average accuracy for each k-fold split for normal (non-polynomial) model
        #training average accuracy - normal non-poly
        training_accuracy_results.append(np.mean(training_k_accuracy_results))
        training_precision_results.append(np.mean(training_k_precision_results))
        training_recall_results.append(np.mean(training_k_recall_results))
        training_f1_results.append(np.mean(training_k_f1_results))
        training_specificity_results.append(np.mean(training_k_specificity_results))
        training_predicted_positive_rate_results.append(np.mean(training_k_predicted_positive_rate_results))
        training_observed_positive_rate_results.append(np.mean(training_k_observed_positive_rate_results))
        training_roc_auc_results.append(np.mean(training_k_roc_auc_results))
        
        #test average accuracy - normal non-poly
        test_accuracy_results.append(np.mean(test_k_accuracy_results))
        test_precision_results.append(np.mean(test_k_precision_results))
        test_recall_results.append(np.mean(test_k_recall_results))
        test_f1_results.append(np.mean(test_k_f1_results))
        test_specificity_results.append(np.mean(test_k_specificity_results))
        test_predicted_positive_rate_results.append(np.mean(test_k_predicted_positive_rate_results))
        test_observed_positive_rate_results.append(np.mean(test_k_observed_positive_rate_results))
        test_roc_auc_results.append(np.mean(test_k_roc_auc_results))

        # Record average accuracy for each k-fold split for polynomial model
        #training average accuracy - poly
        training_accuracy_results_poly.append(np.mean(training_k_accuracy_results_poly))
        training_precision_results_poly.append(np.mean(training_k_precision_results_poly))
        training_recall_results_poly.append(np.mean(training_k_recall_results_poly))
        training_f1_results_poly.append(np.mean(training_k_f1_results_poly))
        training_specificity_results_poly.append(np.mean(training_k_specificity_results_poly))
        training_predicted_positive_rate_results_poly.append(np.mean(training_k_predicted_positive_rate_results_poly))
        training_observed_positive_rate_results_poly.append(np.mean(training_k_observed_positive_rate_results_poly))
        training_roc_auc_results_poly.append(np.mean(training_k_roc_auc_results_poly))
        
        #test average accuracy - poly
        test_accuracy_results_poly.append(np.mean(test_k_accuracy_results_poly))
        test_precision_results_poly.append(np.mean(test_k_precision_results_poly))
        test_recall_results_poly.append(np.mean(test_k_recall_results_poly))
        test_f1_results_poly.append(np.mean(test_k_f1_results_poly))
        test_specificity_results_poly.append(np.mean(test_k_specificity_results_poly))
        test_predicted_positive_rate_results_poly.append(np.mean(test_k_predicted_positive_rate_results_poly))
        test_observed_positive_rate_results_poly.append(np.mean(test_k_observed_positive_rate_results_poly))
        test_roc_auc_results_poly.append(np.mean(test_k_roc_auc_results_poly))

        #original code
        training_acc_results.append(np.mean(training_k_results))
        test_acc_results.append(np.mean(test_k_results))
        training_acc_results_poly.append(np.mean(training_k_results_poly))
        test_acc_results_poly.append(np.mean(test_k_results_poly))

    #NEXT STEPS:
    #add in code from the threshold or undersampling function, to create df of the various regularisation values and performance metrics for each
    #and another df for auc - DONE
    #return this, then outside the functoin, call the function to identify the max value and store in the growing dfs
    #also, look at the coeff code, to see if can identify a way to produce the coeffs from this code and store for later use
    #later we can identify the optimal model, and return the coeffs for that model? to make this work all models need this revision making - test on this function

    # Transfer normal (non-poly) results to dataframe
    df_test_results = pd.DataFrame(reg_values, columns=['reg_values'])
    df_test_results['accuracy'] = test_accuracy_results
    df_test_results['precision'] = test_precision_results
    df_test_results['recall'] = test_recall_results
    df_test_results['f1'] = test_f1_results
    df_test_results['specificity'] = test_specificity_results
    df_test_results['predicted_positive_rate'] = test_predicted_positive_rate_results
    df_test_results['observed_positive_rate'] = test_observed_positive_rate_results
    df_test_results['AUC'] = test_roc_auc_results #new

    # Transfer poly results to dataframe
    df_poly_test_results = pd.DataFrame(reg_values, columns=['reg_values'])
    df_poly_test_results['accuracy'] = test_accuracy_results_poly
    df_poly_test_results['precision'] = test_precision_results_poly
    df_poly_test_results['recall'] = test_recall_results_poly
    df_poly_test_results['f1'] = test_f1_results_poly
    df_poly_test_results['specificity'] = test_specificity_results_poly
    df_poly_test_results['predicted_positive_rate'] = test_predicted_positive_rate_results_poly
    df_poly_test_results['observed_positive_rate'] = test_observed_positive_rate_results_poly
    df_poly_test_results['AUC'] = test_roc_auc_results_poly #new

    #original code
    best_test_non_poly = np.max(test_acc_results)
    best_test_poly = np.max(test_acc_results_poly)
    best_test_auc_non_poly = np.max(test_roc_auc_results)
    best_test_auc_poly = np.max(test_roc_auc_results_poly)
    print ("\nBest accuracy for non-poly and poly were:")
    print(f"Non-poly:{best_test_non_poly:.3f}")
    print(f"Poly: {best_test_poly:.3f}")
    print ("\nBest AUC for non-poly and poly were:")
    print(f"Non-poly:{best_test_auc_non_poly:.3f}")
    print(f"Poly: {best_test_auc_poly:.3f}")

    return(
        reg_values,
        training_acc_results,
        test_acc_results,
        training_acc_results_poly,
        test_acc_results_poly,
        poly,
        X_poly,
        df_test_results,
        df_poly_test_results,
        best_test_auc_non_poly,
        best_test_auc_poly
        )


# --------------------------------------------------

#Function to plot impact of polynomial expansion on accuracy
def plot_polynomial_expansion_impact_on_accuracy(
        reg_values,
        training_acc_results,
        test_acc_results,
        training_acc_results_poly,
        test_acc_results_poly
        ):
    
    #NOTE: This function needs to be called AFTER the add_polynomial_features function
    #as add_polynomial_features returns the variables required to create the plot
    
    # Define data for chart
    x = reg_values
    y1 = training_acc_results
    y2 = test_acc_results
    y3 = training_acc_results_poly
    y4 = test_acc_results_poly

    # Set up figure
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111)

    # Plot training set accuracy
    ax1.plot(x, y1,
            color = 'k',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='k',
            markeredgecolor='k',
            label  = 'Training set accuracy')

    # Plot test set accuracy
    ax1.plot(x, y2,
            color = 'r',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='r',
            markeredgecolor='r',
            label  = 'Test set accuracy')

    # Plot training set accuracy (poly model)
    ax1.plot(x, y3,
            color = 'g',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='g',
            markeredgecolor='g',
            label  = 'Training set accuracy (poly)')

    # Plot test set accuracy (poly model)
    ax1.plot(x, y4,
            color = 'b',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='b',
            markeredgecolor='b',
            label  = 'Test set accuracy (poly)')

    # Customise axes
    ax1.grid(True, which='both')
    ax1.set_xlabel('Regularisation\n(lower value = greater regularisation)')
    ax1.set_ylabel('Accuracy')
    ax1.set_xscale('log')

    # Add legend
    ax1.legend()

    # Show plot
    plt.show()
    
    return(fig)

# --------------------------------------------------

#Function to undertake feature reduction after expansion
def feature_reduction_after_poly_expansion(
        X_poly,
        poly,
        max_features,
        X,
        num_splits
        ):
    
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names())

    # Create list to store accuracies and chosen features
    accuracy_by_feature_number = []
    chosen_features = []

    # Initialise chosen features list and run tracker
    available_features = list(poly.get_feature_names())
    run = 0
    number_of_features = len(list(X))

    # Loop through feature list to select next feature
    maximum_features_to_choose = max_features #This was 20, changed to 5 due to laptop limitations

    for i in range(maximum_features_to_choose):

        # Track and print progress
        run += 1
        print (f"Feature run {run} of {maximum_features_to_choose}")
        
        # Reset best feature and accuracy
        best_result = 0
        best_feature = ''

        # Loop through available features
        for feature in available_features:

            # Create copy of already chosen features to avoid original being changed
            features_to_use = chosen_features.copy()
            # Create a list of features from features already chosen + 1 new feature
            features_to_use.append(feature)
            # Get data for features, and convert to NumPy array
            X_np = X_poly_df[features_to_use].values
            
            # Set up lists to hold results for each selected features
            test_accuracy_results = []
        
            # Set up k-fold training/test splits
            number_of_splits = num_splits
            skf = StratifiedKFold(n_splits = number_of_splits)
            skf.get_n_splits(X_np, y)
        
            # Loop through the k-fold splits
            for train_index, test_index in skf.split(X_np, y):
                
                # Get X and Y train/test
                X_train, X_test = X_np[train_index], X_np[test_index]
                y_train, y_test = y[train_index], y[test_index]
        
                # Get X and Y train/test
                X_train_std, X_test_std = standardise_data(X_train, X_test)
        
                # Set up and fit model
                model = LogisticRegression(solver='lbfgs')
                model.fit(X_train_std,y_train)
        
                # Predict test set labels
                y_pred_test = model.predict(X_test_std)
                            
                # Calculate accuracy of test sets
                accuracy_test = np.mean(y_pred_test == y_test)
                test_accuracy_results.append(accuracy_test)
              
            # Get average result from all k-fold splits
            feature_accuracy = np.mean(test_accuracy_results)
        
            # Update chosen feature and result if this feature is a new best
            if feature_accuracy > best_result:
                best_result = feature_accuracy
                best_feature = feature
        
        # k-fold splits are complete    
        # Add mean accuracy and AUC to record of accuracy by feature number
        accuracy_by_feature_number.append(best_result)
        chosen_features.append(best_feature)
        available_features.remove(best_feature)

    # Put results in DataFrame
    results = pd.DataFrame()
    results['feature to add'] = chosen_features
    results['accuracy'] = accuracy_by_feature_number

    #Chart of accuracy by cumulative number of features
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot()
    
    chart_x = list(range(1, maximum_features_to_choose+1))
    ax1.plot(chart_x, accuracy_by_feature_number,
            label = 'Accuracy')
    ax1.set_xlabel('Number of features')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_title("Chart to show how incrementally increasing features in the model \nimpacts on it's accuracy when predicting if a given patient \nDNA's their appointment")
    ax1.grid(True)
    plt.show()

    #Note that sklearn's polynomial method outputs features names in relation to 
    #the original X index.  
    #This code converts the "best" feature (or features if a polynomial .product of 2 features)
    #back to the original feature names in X:

    X_index_names = list(X)

    list_of_features = results['feature to add'][0].split(" ")

    for feature in list_of_features:
        print(f"\n{feature}: {X_index_names[int(feature[1:])]}")

    if len(list_of_features) >1:
        print("\n...So these features are the single biggest predictors of DNA")
    else:
        print("\n...So this feature is the single biggest predictor of DNA")
        
    return(
        fig,
        results,
        list_of_features
        )

# --------------------------------------------------


# this version is the one that has been changed to set the random state so 
# output is reproducible each time the cell wiht this functon in is run. 

#Undersampling majority class
def reduce_majority_class(X, y, rand_state, maj_sample_size=100):

    """
    #from Mike Allen's titantic ML tutorial workbooks.
    Purpose is to progressively reduce the size of the majority class examples 
    ('died') in the training set, and look at the effect on various measures of 
    accuracy.
    
    What this function is doing-
    Identify majority class (for binary classification of y) and reduce 
    size of majority class to `n` (defaults to 100). Uses bootstrap sampling,
    so will not error if requested sample size is greater than available 
    samples.
    
    Input: X and y NumPy arrays
    Output X and y NumPy arrays
    """
    
    np.random.seed(rand_state)

    # Get majority class
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    index_max = counts_elements.argmax()
    majority_class = unique_elements[index_max]

    # Separate out classes by index
    majority_index = np.where(y == majority_class)[0]
    minority_index = np.where(y != majority_class)[0]
    
    # Sample from majority class
    restricted_sample_index = np.random.choice(
            majority_index, maj_sample_size, replace=True)
    
    # Get restricted X and y for restricted data
    X_majority_restricted = X[restricted_sample_index, :]
    y_majority_restricted = y[restricted_sample_index]
    
    # Get all of minority class
    X_minority = X[minority_index, :]
    y_minority = y[minority_index]
    
    # Combine X and y 
    X_new = np.concatenate([X_majority_restricted, X_minority])
    y_new = np.concatenate([y_majority_restricted, y_minority])
    
    # Shuffle (use random index to shuffle X and y in same order)
    count_rows = X_new.shape[0]
    random_index = np.random.permutation(count_rows)
    X_new = X_new[random_index, :]
    y_new = y_new[random_index]
    
    return X_new, y_new


# --------------------------------------------------

def id_highest_auc_return_df(
        df, 
        auc_column, 
        column_of_interest, 
        summary_df_col_label,
        roundingnumber = 1
        ):
    #identify the max value of the auc_column in the given df
    max_auc_oversample_min_class = df[auc_column].max()
    
    #identify the index of that max value
    index_max_auc = df[auc_column].idxmax()
    
    #in the given column of interest, identify the value associated with the 
    #max auc value identified above e.g. if the column of interest is sample
    #size, this will return the sample size which yielded the highest auc
    column_of_interest_max_auc = df[column_of_interest].iloc[index_max_auc].round(roundingnumber)
    
    #convert the column of interest value from above into string type for
    #subsequent use in the column header in the returned df
    str_column_of_interest_max_auc = str(column_of_interest_max_auc)
    
    data_for_df = {f"{summary_df_col_label}{str_column_of_interest_max_auc}":
                   [max_auc_oversample_min_class]}
    df_temp_max_auc = pd.DataFrame(data_for_df)
    return df_temp_max_auc

# --------------------------------------------------

#original function code - delete in final. This has been replaced with the function "increase_minority"
#The difference is that in the replacement function, the random seed is set to ensure results are replicable

#Oversampling minority class
def increase_minority_original(X, y, maj_sample_size=100):
    """
    Purpose - progressively increase the size of the minority class examples 
    in the training set, and look at the effect on various measures of accuracy. 
    Over-sampling is achieved by repeatedly re-sampling from the original data.
    
    What this function is doing-
    Identify majority class (for binary classification of y) and reduce 
    size of majority class to `n` (defaults to 100). Uses bootstrap sampling,
    so will not error if requested sample size is greater than available 
    samples.
    
    Input: X and y NumPy arrays
    Output X and y NumPy arrays
    """
    
    # Get minority class
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    index_max = counts_elements.argmin()
    minority_class = unique_elements[index_max]

    # Separate out classes by index
    minority_index = np.where(y == minority_class)[0]
    majority_index = np.where(y != minority_class)[0]
    
    # Sample from majority class
    enhanced_sample_index = np.random.choice(
            minority_index, maj_sample_size, replace=True)
    
    # Get restricted X and y for restricted data
    X_minority_enhanced = X[enhanced_sample_index, :]
    y_minority_enhanced = y[enhanced_sample_index]
    
    # Get all of majority class
    X_majority = X[majority_index, :]
    y_majority = y[majority_index]
    
    # Combine X and y 
    X_new = np.concatenate([X_minority_enhanced, X_majority])
    y_new = np.concatenate([y_minority_enhanced, y_majority])
    
    # Shuffle (use random index to shuffle X and y in same order)
    count_rows = X_new.shape[0]
    random_index = np.random.permutation(count_rows)
    X_new = X_new[random_index, :]
    y_new = y_new[random_index]
    
    return X_new, y_new

# --------------------------------------------------

#update function code - keep in final. This is the replacement for the function "increase_minority_original"
#The difference is that in the replacement function, the random seed is set to ensure results are replicable

#Oversampling minority class
def increase_minority(X, y, rand_state, maj_sample_size=100):
    """
    Purpose - progressively increase the size of the minority class examples 
    in the training set, and look at the effect on various measures of accuracy. 
    Over-sampling is achieved by repeatedly re-sampling from the original data.
    
    What this function is doing-
    Identify majority class (for binary classification of y) and reduce 
    size of majority class to `n` (defaults to 100). Uses bootstrap sampling,
    so will not error if requested sample size is greater than available 
    samples.
    
    Input: X and y NumPy arrays
    Output X and y NumPy arrays
    """

    #set random seed to ensure replicable results
    np.random.seed(rand_state)

    # Get minority class
    unique_elements, counts_elements = np.unique(y, return_counts=True)
    index_max = counts_elements.argmin()
    minority_class = unique_elements[index_max]

    # Separate out classes by index
    minority_index = np.where(y == minority_class)[0]
    majority_index = np.where(y != minority_class)[0]
    
    # Sample from majority class
    enhanced_sample_index = np.random.choice(
            minority_index, maj_sample_size, replace=True)
    
    # Get restricted X and y for restricted data
    X_minority_enhanced = X[enhanced_sample_index, :]
    y_minority_enhanced = y[enhanced_sample_index]
    
    # Get all of majority class
    X_majority = X[majority_index, :]
    y_majority = y[majority_index]
    
    # Combine X and y 
    X_new = np.concatenate([X_minority_enhanced, X_majority])
    y_new = np.concatenate([y_minority_enhanced, y_majority])
    
    # Shuffle (use random index to shuffle X and y in same order)
    count_rows = X_new.shape[0]
    random_index = np.random.permutation(count_rows)
    X_new = X_new[random_index, :]
    y_new = y_new[random_index]
    
    return X_new, y_new

# --------------------------------------------------

def use_smote_to_balance_data(
        categorical,
        X_np,
        y_np,
        num_splits,
        rand_state
        ):
    
    results_names = [
        'accuracy',
        'precision', 
        'recall', 
        'f1', 
        'specificity', 
        'predicted_positive_rate', 
        'observed_positive_rate'
        ]
    results = pd.DataFrame(index = results_names)
    
    # Set up lists to hold results for each k-fold run
    replicate_accuracy = []
    replicate_precision = []
    replicate_recall = []
    replicate_f1 = []
    replicate_specificity = []
    replicate_predicted_positive_rate = [] #new
    replicate_observed_positive_rate = [] #new
    
    replicate_auc = [] #new
    
    # Set up splits
    number_of_splits = 10
    skf = StratifiedKFold(n_splits = number_of_splits, shuffle = True, random_state=rand_state)
    skf.get_n_splits(X, y)
    
    # Loop through the k-fold splits
    for train_index, test_index in skf.split(X_np, y_np):
        
        # Get X and Y train/test
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        
        # Create an enhanced data set with SMOTENC
        smote_nc = SMOTENC(categorical_features=categorical, random_state=rand_state)
        X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
        
        # Standardise X data
        X_train_std, X_test_std = standardise_data(X_resampled, X_test)
        
        # Set up and fit model
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train_std, y_resampled)
        
        # Predict training and test set labels
        y_pred_train = model.predict(X_train_std)
        y_pred_test = model.predict(X_test_std)
        
        # Get test set proabilities
        y_pred_proba = model.predict_proba(X_test_std)[:,1]
        
        #Get ROC AUC for this split, append to list of AUC's
        auc = roc_auc_score(y_test, y_pred_proba)
        replicate_auc.append(auc)
        
        # Predict test set labels and get accuracy scores
        y_pred_test = model.predict(X_test_std)
        accuracy_scores = calculate_accuracy(y_test, y_pred_test)
        replicate_accuracy.append(accuracy_scores['accuracy'])
        replicate_precision.append(accuracy_scores['precision'])
        replicate_recall.append(accuracy_scores['recall'])
        replicate_f1.append(accuracy_scores['f1'])
        replicate_specificity.append(accuracy_scores['specificity'])
        replicate_predicted_positive_rate.append(accuracy_scores['predicted_positive_rate'])
        replicate_observed_positive_rate.append(accuracy_scores['observed_positive_rate'])
        
    # Transfer results to list and add to data frame
    smote_results = [np.mean(replicate_accuracy),
                         np.mean(replicate_precision),
                         np.mean(replicate_recall),
                         np.mean(replicate_f1),
                         np.mean(replicate_specificity),
                         np.mean(replicate_predicted_positive_rate),
                         np.mean(replicate_observed_positive_rate)
                         ]
    
    mean_auc = np.mean(replicate_auc)
    
    col_name_smote = "auc_Smote"
    df_smote_kfold = pd.DataFrame(columns=[col_name_smote])
    df_smote_kfold.loc[0] = mean_auc

    results['smote'] = smote_results
    return results, df_smote_kfold

# --------------------------------------------------

def get_new_performance_summary_df(list_of_metrics_for_summary):
    #----------------------------------------------------
    #create DataFrame into which all metrics will later be added
    #----------------------------------------------------
    
    #create empty DataFrame
    df_summary_of_models = pd.DataFrame()

    #Iterate over list of metrics for summary, add to df above, and set these metric labels as the index
    dict_item_counter = 0
    dict_metrics_summary = {}
    for item in list_of_metrics_for_summary:
        dict_metrics_summary[dict_item_counter] = item
        dict_item_counter += 1

    df_summary_of_models['performance_metrics'] = list_of_metrics_for_summary
    df_summary_of_models.set_index('performance_metrics', inplace=True)
    return(df_summary_of_models)

# --------------------------------------------------

def id_coefficients(model):

    #Examining the model co-efficients
    """
    Weights with higher negative numbers mean that that feature correlates with 
    reduced chance of DNA (outcome variable, y). 
    Weights with higher positive numbers mean that that feature correlates with 
    increased chance of DNA (outcome variable, y). 
    Those weights with values closer to zero (either positive or negative) have 
    less influence in the model.
    We access model weights via the model coef_ attribute. 
    A model may predict >1 outcome label, in which case we have weights for each label. 
    This model only predicts a single label (DNA or not), so the weights are found 
    in the first element ([0]) of the coef_ attribute.
    """

    co_eff = model.coef_[0]

    #Transferring the weights array to a Pandas DataFrame to improve readability 
    """
    The array order is in the same order of the list of features of X, 
    so we will put that those into the DataFrame as well. 
    We will sort by influence in the model. Because both large negative and positive 
    values are more influential in the model we will take the absolute value of the 
    weight (that is remove any negative sign), and then sort by that absolute value. 
    That will give us a more readable table of most influential features in the model.
    """

    co_eff_df = pd.DataFrame() # create empty DataFrame
    co_eff_df['feature'] = list(X) # Get feature names from X
    co_eff_df['co_eff'] = co_eff
    co_eff_df['abs_co_eff'] = np.abs(co_eff)
    co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)

    #use np.sign to label any negative value as -1 and any positive value as +1
    co_eff_from_df = co_eff_df['co_eff'].values
    co_eff_df['sign_(pos=higher_dna_chance)'] = np.sign(co_eff_from_df)

    #creating 2 subset DataFrames from co_eff_df
    """
    one for factors associated with higher chance of dna (positive sign)
    one for factors associated with lower chance of dna (negative sign)
    """
    co_eff_df_reduced_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == -1]
    co_eff_df_higher_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == 1]

    #Print the df of ALL features ordered by most influential to least influential
    return(co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna)


# --------------------------------------------------

def id_top_x_coefficients(df_reduced_chance, df_increased_chance, top_x, model_name):
    #Print the subset df of top 5 features ordered by most influential to least influential, that have a reduced chance of dna
    #TO GO INTO OUTPUT REPORT
    #print(co_eff_df_reduced_chance_dna)
    features = list(df_reduced_chance['feature'][:top_x])

    #create a df of the top_x associated featutes with reduced chance of DNA
    top_x_features = list(df_reduced_chance['feature'][:top_x])
    top_x_feature_coeff_values = list(df_reduced_chance['co_eff'][:top_x])
    zipped = list(zip(top_x_features, top_x_feature_coeff_values))
    df_top_x_reduce_chance_dna = pd.DataFrame(zipped, columns=['feature', model_name])
    
    print(df_top_x_reduce_chance_dna)

    #code to print to screen the top_x - use to check df is identifical, then remove
    print(f"\nThe most influential {top_x} features reducing the chance of a DNA are:")
    for feature in features:
        if feature[:8] == "encoded_":
            print(feature[8:]) #this slice removes the prefix 'encoded_' where present
        else:
            print(feature)

    #Print the subset df of top 5 features ordered by most influential to least influential, that have an increased chance of dna
    #TO GO INTO OUTPUT REPORT
    #print(co_eff_df_higher_chance_dna)
    features = list(df_increased_chance['feature'][:top_x])

    #create a df of the top_x associated featutes with increased chance of DNA
    top_x_features = list(df_increased_chance['feature'][:top_x])
    top_x_feature_coeff_values = list(df_increased_chance['co_eff'][:top_x])
    zipped = list(zip(top_x_features, top_x_feature_coeff_values))
    df_top_x_increased_chance_dna = pd.DataFrame(zipped, columns=['feature', model_name])
    
    print(df_top_x_increased_chance_dna)

    print(f"\nThe most influential {top_x} features increasing the chance of a DNA are:")
    for feature in features:
        if feature[:8] == "encoded_":
            print(feature[8:]) #this slice removes the prefix 'encoded_' where present
        else:
            print(feature)

    return(df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna)


# --------------------------------------------------

def id_top_x_coefficients(df_reduced_chance, df_increased_chance, top_x, model_name):
    #Print the subset df of top 5 features ordered by most influential to least influential, that have a reduced chance of dna
    #TO GO INTO OUTPUT REPORT
    #print(co_eff_df_reduced_chance_dna)
    features = list(df_reduced_chance['feature'][:top_x])

    #create a df of the top_x associated featutes with reduced chance of DNA
    top_x_features = list(df_reduced_chance['feature'][:top_x])
    top_x_feature_coeff_values = list(df_reduced_chance['co_eff'][:top_x])
    zipped = list(zip(top_x_features, top_x_feature_coeff_values))
    df_top_x_reduce_chance_dna = pd.DataFrame(zipped, columns=['feature', model_name])
    
    print(df_top_x_reduce_chance_dna)

    #code to print to screen the top_x - use to check df is identifical, then remove
    print(f"\nThe most influential {top_x} features reducing the chance of a DNA are:")
    for feature in features:
        if feature[:8] == "encoded_":
            print(feature[8:]) #this slice removes the prefix 'encoded_' where present
        else:
            print(feature)

    #Print the subset df of top 5 features ordered by most influential to least influential, that have an increased chance of dna
    #TO GO INTO OUTPUT REPORT
    #print(co_eff_df_higher_chance_dna)
    features = list(df_increased_chance['feature'][:top_x])

    #create a df of the top_x associated featutes with increased chance of DNA
    top_x_features = list(df_increased_chance['feature'][:top_x])
    top_x_feature_coeff_values = list(df_increased_chance['co_eff'][:top_x])
    zipped = list(zip(top_x_features, top_x_feature_coeff_values))
    df_top_x_increased_chance_dna = pd.DataFrame(zipped, columns=['feature', model_name])
    
    print(df_top_x_increased_chance_dna)

    print(f"\nThe most influential {top_x} features increasing the chance of a DNA are:")
    for feature in features:
        if feature[:8] == "encoded_":
            print(feature[8:]) #this slice removes the prefix 'encoded_' where present
        else:
            print(feature)

    return(df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna)


# --------------------------------------------------

def add_coeff_to_dict(target_dict, source_df, model_name):
    target_dict[model_name] = source_df
    return(target_dict)

# --------------------------------------------------


def create_coefficients_df(average_coefficients, X):

    #Transferring the weights array to a Pandas DataFrame to improve readability 
    """
    The array order is in the same order of the list of features of X, 
    so we will put that those into the DataFrame as well. 
    We will sort by influence in the model. Because both large negative and positive 
    values are more influential in the model we will take the absolute value of the 
    weight (that is remove any negative sign), and then sort by that absolute value. 
    That will give us a more readable table of most influential features in the model.
    """
    co_eff_df = pd.DataFrame() # create empty DataFrame
    co_eff_df['feature'] = list(X) # Get feature names from X
    co_eff_df['co_eff'] = average_coefficients
    co_eff_df['abs_co_eff'] = np.abs(average_coefficients)
    co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)

    #use np.sign to label any negative value as -1 and any positive value as +1
    co_eff_from_df = co_eff_df['co_eff'].values
    co_eff_df['sign_(pos=higher_dna_chance)'] = np.sign(co_eff_from_df)

    #creating 2 subset DataFrames from co_eff_df
    """
    one for factors associated with higher chance of dna (positive sign)
    one for factors associated with lower chance of dna (negative sign)
    """
    co_eff_df_reduced_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == -1]
    co_eff_df_higher_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == 1]

    #Print the df of ALL features ordered by most influential to least influential
    return(co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna)


# --------------------------------------------------

def id_coefficients(model):

    #Examining the model co-efficients
    """
    Weights with higher negative numbers mean that that feature correlates with 
    reduced chance of DNA (outcome variable, y). 
    Weights with higher positive numbers mean that that feature correlates with 
    increased chance of DNA (outcome variable, y). 
    Those weights with values closer to zero (either positive or negative) have 
    less influence in the model.
    We access model weights via the model coef_ attribute. 
    A model may predict >1 outcome label, in which case we have weights for each label. 
    This model only predicts a single label (DNA or not), so the weights are found 
    in the first element ([0]) of the coef_ attribute.
    """

    co_eff = model.coef_[0]

    #Transferring the weights array to a Pandas DataFrame to improve readability 
    """
    The array order is in the same order of the list of features of X, 
    so we will put that those into the DataFrame as well. 
    We will sort by influence in the model. Because both large negative and positive 
    values are more influential in the model we will take the absolute value of the 
    weight (that is remove any negative sign), and then sort by that absolute value. 
    That will give us a more readable table of most influential features in the model.
    """

    co_eff_df = pd.DataFrame() # create empty DataFrame
    co_eff_df['feature'] = list(X) # Get feature names from X
    co_eff_df['co_eff'] = co_eff
    co_eff_df['abs_co_eff'] = np.abs(co_eff)
    co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)

    #use np.sign to label any negative value as -1 and any positive value as +1
    co_eff_from_df = co_eff_df['co_eff'].values
    co_eff_df['sign_(pos=higher_dna_chance)'] = np.sign(co_eff_from_df)

    #creating 2 subset DataFrames from co_eff_df
    """
    one for factors associated with higher chance of dna (positive sign)
    one for factors associated with lower chance of dna (negative sign)
    """
    co_eff_df_reduced_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == -1]
    co_eff_df_higher_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == 1]

    #Print the df of ALL features ordered by most influential to least influential
    return(co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna)


# --------------------------------------------------

#NEED TO INCORPORATE AN IF STATEMENT TO CONTROL FOR SCENARIO WHERE COEFFICIENTS ARE <5 IN TOTAL. 
def id_top_x_coefficients(df_reduced_chance, df_increased_chance, top_x, model_name):
    #Print the subset df of top 5 features ordered by most influential to least influential, that have a reduced chance of dna
    #TO GO INTO OUTPUT REPORT
    #print(co_eff_df_reduced_chance_dna)
    features = list(df_reduced_chance['feature'][:top_x])

    #create a df of the top_x associated featutes with reduced chance of DNA
    top_x_features = list(df_reduced_chance['feature'][:top_x])
    top_x_feature_coeff_values = list(df_reduced_chance['co_eff'][:top_x])
    zipped = list(zip(top_x_features, top_x_feature_coeff_values))
    df_top_x_reduce_chance_dna = pd.DataFrame(zipped, columns=['feature', model_name])
    
    print(df_top_x_reduce_chance_dna)

    #code to print to screen the top_x - use to check df is identifical, then remove
    print(f"\nThe most influential {top_x} features reducing the chance of a DNA are:")
    for feature in features:
        if feature[:8] == "encoded_":
            print(feature[8:]) #this slice removes the prefix 'encoded_' where present
        else:
            print(feature)

    #Print the subset df of top 5 features ordered by most influential to least influential, that have an increased chance of dna
    #TO GO INTO OUTPUT REPORT
    #print(co_eff_df_higher_chance_dna)
    features = list(df_increased_chance['feature'][:top_x])

    #create a df of the top_x associated featutes with increased chance of DNA
    top_x_features = list(df_increased_chance['feature'][:top_x])
    top_x_feature_coeff_values = list(df_increased_chance['co_eff'][:top_x])
    zipped = list(zip(top_x_features, top_x_feature_coeff_values))
    df_top_x_increased_chance_dna = pd.DataFrame(zipped, columns=['feature', model_name])
    
    print(df_top_x_increased_chance_dna)

    print(f"\nThe most influential {top_x} features increasing the chance of a DNA are:")
    for feature in features:
        if feature[:8] == "encoded_":
            print(feature[8:]) #this slice removes the prefix 'encoded_' where present
        else:
            print(feature)

    return(df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna)


# --------------------------------------------------

def add_coeff_to_dict(target_dict, source_df, model_name):
    target_dict[model_name] = source_df
    return(target_dict)

# --------------------------------------------------


#test function to include coeff - CODE BEING WORKED ON TO SAVE FIG APPROPRIATELY
def run_log_reg_undersample_maj_class_new(
        max_sample_size,
        X,
        X_np,
        y_np,
        num_splits,
        rand_state,
        class_descriptor,
        dna_assets_path
        ):
    
    """
    Must be called after the 'reduce_majority_class' function as 
    reduce_majority_class is called within this function
    
    THINGS TO CONFIRM IN THIS FUNCTION-
    is it ok to take mean of the ROC AUC scores of each split for changing 
    majority class size?
    how to use the returned df - locate the highest AUC and use that row 
    as the results / select that undersampled sample size as the post-hoc parameter?
    
    
    """
    
    #np.random.seed(0)

    # Create lists for overall results
    results_raw_majority_count = []
    results_accuracy = []
    results_precision = []
    results_recall = []
    results_f1 = []
    results_specificity = [] #new
    results_predicted_positive_rate = []
    results_observed_positive_rate = [] #new
    results_auc = [] #new addition
    results_coeff = [] #new addition to capture coeffs for each sample size
    results_coeff_dict = {} #new addition to capture coeffs for each sample size - as a dict
    
    samples_sizes = list(np.arange(50, max_sample_size, 50))
    for maj_sample_size in samples_sizes:
    
        # Set up loop for replicates for each training majority class size
        replicate_raw_majority_count = []
        replicate_accuracy = []
        replicate_precision = []
        replicate_recall = []
        replicate_f1 = []
        replicate_specificity = [] #new
        replicate_predicted_positive_rate = []
        replicate_observed_positive_rate = [] #new
        replicate_auc = [] #new addition
        
        #set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
        replicate_coeff_list_of_lists = []

        # Set up k-fold training/test splits
        number_of_splits = num_splits
        skf = StratifiedKFold(n_splits = number_of_splits, shuffle=True, random_state=rand_state)
        skf.get_n_splits(X_np, y_np)
         
        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            
            # Get X and Y train/test
            X_train_full, X_test = X_np[train_index], X_np[test_index]
            y_train_full, y_test = y_np[train_index], y_np[test_index]
            
            # Record original training majority class
            count_of_training_died = (y_train_full == 0).sum()
            replicate_raw_majority_count.append(count_of_training_died)
            
            # Get restricted training set
            X_train, y_train = reduce_majority_class(
                    X_train_full, y_train_full, rand_state, maj_sample_size)
            
            # Get X and Y train/test
            X_train_std, X_test_std = standardise_data(X_train, X_test)
            
            # Set up and fit model
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train_std,y_train)
            
            #get coefficients for this fold
            """
            Weights with higher negative numbers mean that that feature correlates with 
            reduced chance of DNA (outcome variable, y). 
            Weights with higher positive numbers mean that that feature correlates with 
            increased chance of DNA (outcome variable, y). 
            Those weights with values closer to zero (either positive or negative) have 
            less influence in the model.
            We access model weights via the model coef_ attribute. 
            A model may predict >1 outcome label, in which case we have weights for each label. 
            This model only predicts a single label (DNA or not), so the weights are found 
            in the first element ([0]) of the coef_ attribute.
            """
            co_eff = model.coef_[0]
            replicate_coeff_list_of_lists.append(co_eff)

            #Get ROC AUC for this split, append to list of AUC's
            y_pred_proba = model.predict_proba(X_test)[::,1]
            auc = roc_auc_score(y_test, y_pred_proba)
            replicate_auc.append(auc)
            
            # Predict test set labels and get accuracy scores
            y_pred_test = model.predict(X_test_std)
            accuracy_scores = calculate_accuracy(y_test, y_pred_test)
            replicate_accuracy.append(accuracy_scores['accuracy'])
            replicate_precision.append(accuracy_scores['precision'])
            replicate_recall.append(accuracy_scores['recall'])
            replicate_f1.append(accuracy_scores['f1'])
            replicate_specificity.append(accuracy_scores['specificity'])
            replicate_predicted_positive_rate.append(accuracy_scores['predicted_positive_rate'])
            replicate_observed_positive_rate.append(accuracy_scores['observed_positive_rate'])
                            
        # Add mean results to overall results
        results_raw_majority_count.append(np.mean(replicate_raw_majority_count))
        results_accuracy.append(np.mean(replicate_accuracy))
        results_precision.append(np.mean(replicate_precision))
        results_recall.append(np.mean(replicate_recall))
        results_f1.append(np.mean(replicate_f1))
        results_specificity.append(np.mean(replicate_specificity)) #new
        results_predicted_positive_rate.append(np.mean(replicate_predicted_positive_rate))
        results_observed_positive_rate.append(np.mean(replicate_observed_positive_rate)) #new
        results_auc.append(np.mean(replicate_auc)) #new - take mean of auc's (confirm permissable)

        #average the coefficient values produced for each fold, to get to a single array of average coefficients. 
        #these are in the same order as the columns in X
        results_coeff.append(np.mean(replicate_coeff_list_of_lists, 0))
        results_coeff_temp = np.mean(replicate_coeff_list_of_lists, 0) 
        results_coeff_dict[maj_sample_size] = results_coeff_temp #test
        #co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(average_coefficients, X)
    
    # Transfer results to dataframe
    results = pd.DataFrame(samples_sizes, columns=['sample_size'])
    results['accuracy'] = results_accuracy
    results['precision'] = results_precision
    results['recall'] = results_recall
    results['f1'] = results_f1
    results['specificity'] = results_specificity
    results['predicted_positive_rate'] = results_predicted_positive_rate
    results['observed_positive_rate'] = results_observed_positive_rate
    results['AUC'] = results_auc #new
    
    # Transfer coeff to DataFrame
    df_results_coeff = pd.DataFrame(samples_sizes, columns=['sample_size'])
    df_results_coeff['coeff'] = results_coeff

    #plot results
    chart_x = results['sample_size']
    
    fig, ax = plt.subplots()

    ax.plot(chart_x, results['accuracy'],
             linestyle = '-',
             label = 'Accuracy')
    
    ax.plot(chart_x, results['precision'],
             linestyle = '--',
             label = 'Precision')
    
    ax.plot(chart_x, results['recall'],
             linestyle = '-.',
             label = 'Recall')
    
    ax.plot(chart_x, results['f1'],
             linestyle = ':',
             label = 'F1')
    
    ax.plot(chart_x, results['predicted_positive_rate'],
             linestyle = '-',
             label = 'Predicted positive rate')
    
    actual_positive_rate = np.repeat(y.mean(), len(chart_x))
    
    ax.plot(chart_x, actual_positive_rate,
             linestyle = '--',
             color='k',
             label = 'Actual positive rate')
    
    # Add text showing original majority class training size
    original_majority_training_sample = int(np.mean(results_raw_majority_count))
    text = 'Original majority class training size = ' + str(
        original_majority_training_sample)
    
    
    ax.text(80, 0.95,text, bbox=dict(facecolor='white', alpha=1.0))
    
    ax.set_xlabel('Majority class sample size')
    ax.set_ylabel('Score')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.set_title(f"Plot to show the effect of undersampling the majority class ({class_descriptor}) on the prediction")

    # Set background color of Figure
    fig.patch.set_facecolor('white')
    # Set transparency of figure
    fig.patch.set_alpha(1)

    temp_filename = f"chart002_EffectOfUnderSamplingMajorityClass.png"
    file_path = f'{dna_assets_path}/{temp_filename}'
    plt.savefig(file_path, bbox_inches='tight')

    plt.show()
    return results, df_results_coeff, results_coeff_dict

# --------------------------------------------------

#REVISED CODE TO INCLUDE BACKGROUND COLOUR TO PLOT
#test cell for revised code
#revised function to incorporate coefficients - keep in final. currently working on this function.

def run_log_reg_oversample_min_class_new(
        max_sample_size,
        X,
        X_np,
        y_np,
        num_splits,
        rand_state,
        class_descriptor,
        dna_assets_path
        ):
    
    """
    Must be called after the 'increase_minority' function as 
    increase_minority is called within this function
    """

    # Create lists for overall results
    results_raw_minority_count = []
    results_accuracy = []
    results_precision = []
    results_recall = []
    results_f1 = []
    results_specificity = [] #new addition
    results_predicted_positive_rate = []
    results_observed_positive_rate = [] #new addition
    results_auc = [] #new addition
    results_coeff = [] #new addition to capture coeffs for each sample size
    results_coeff_dict = {} #new addition to capture coeffs for each sample size - as a dict
    
    samples_sizes = list(np.arange(50, max_sample_size, 50))
    for maj_sample_size in samples_sizes:
    
        # Set up loop for replicates for each training majority class size
        replicate_raw_minority_count = []
        replicate_accuracy = []
        replicate_precision = []
        replicate_recall = []
        replicate_f1 = []
        replicate_specificity = [] #new addition
        replicate_predicted_positive_rate = []
        replicate_observed_positive_rate = [] #new addition
        replicate_auc = [] #new addition

        #set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
        replicate_coeff_list_of_lists = []
        
        # Set up k-fold training/test splits
        #number_of_splits = num_splits
        skf = StratifiedKFold(n_splits = num_splits, shuffle = True, random_state=rand_state)
        skf.get_n_splits(X_np, y_np)
        
        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            
            # Get X and Y train/test
            X_train_full, X_test = X_np[train_index], X_np[test_index]
            y_train_full, y_test = y_np[train_index], y_np[test_index]
            
            # Record original training majority class
            count_of_training_survived = (y_train_full == 1).sum()
            replicate_raw_minority_count.append(count_of_training_survived)
            
            # Get enhances training set
            X_train, y_train = increase_minority(
                    X_train_full, y_train_full, rand_state, maj_sample_size)
            
            # Get X and Y train/test
            X_train_std, X_test_std = standardise_data(X_train, X_test)
            
            # Set up and fit model
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train_std,y_train)

            #get coefficients for this fold
            """
            Weights with higher negative numbers mean that that feature correlates with 
            reduced chance of DNA (outcome variable, y). 
            Weights with higher positive numbers mean that that feature correlates with 
            increased chance of DNA (outcome variable, y). 
            Those weights with values closer to zero (either positive or negative) have 
            less influence in the model.
            We access model weights via the model coef_ attribute. 
            A model may predict >1 outcome label, in which case we have weights for each label. 
            This model only predicts a single label (DNA or not), so the weights are found 
            in the first element ([0]) of the coef_ attribute.
            """
            co_eff = model.coef_[0]
            replicate_coeff_list_of_lists.append(co_eff)
            
            #Get ROC AUC for this split, append to list of AUC's
            y_pred_proba = model.predict_proba(X_test)[::,1]
            auc = roc_auc_score(y_test, y_pred_proba)
            replicate_auc.append(auc)
            
            # Predict test set labels and get accuracy scores
            y_pred_test = model.predict(X_test_std)
            accuracy_scores = calculate_accuracy(y_test, y_pred_test)
            replicate_accuracy.append(accuracy_scores['accuracy'])
            replicate_precision.append(accuracy_scores['precision'])
            replicate_recall.append(accuracy_scores['recall'])
            replicate_f1.append(accuracy_scores['f1'])
            replicate_specificity.append(accuracy_scores['specificity'])
            replicate_predicted_positive_rate.append(accuracy_scores['predicted_positive_rate'])
            replicate_observed_positive_rate.append(accuracy_scores['observed_positive_rate'])
                            
        # Add mean results to overall results
        results_raw_minority_count.append(np.mean(replicate_raw_minority_count))
        results_accuracy.append(np.mean(replicate_accuracy))
        results_precision.append(np.mean(replicate_precision))
        results_recall.append(np.mean(replicate_recall))
        results_f1.append(np.mean(replicate_f1))
        results_specificity.append(np.mean(replicate_specificity)) #new addition 
        results_predicted_positive_rate.append(np.mean(replicate_predicted_positive_rate))
        results_observed_positive_rate.append(np.mean(replicate_observed_positive_rate))
        results_auc.append(np.mean(replicate_auc)) #new - take mean of auc's (confirm permissable)

        #average the coefficient values produced for each fold, to get to a single array of average coefficients. 
        #these are in the same order as the columns in X
        results_coeff.append(np.mean(replicate_coeff_list_of_lists, 0))
        results_coeff_temp = np.mean(replicate_coeff_list_of_lists, 0) 
        results_coeff_dict[maj_sample_size] = results_coeff_temp #test
        #co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(average_coefficients, X)
    
    # Transfer results to dataframe
    results = pd.DataFrame(samples_sizes, columns=['sample_size'])
    results['accuracy'] = results_accuracy
    results['precision'] = results_precision
    results['recall'] = results_recall
    results['f1'] = results_f1
    results['specificity'] = results_specificity
    results['predicted_positive_rate'] = results_predicted_positive_rate
    results['observed_positive_rate'] = results_observed_positive_rate
    results['AUC'] = results_auc #new
    
    # Transfer coeff to DataFrame
    df_results_coeff = pd.DataFrame(samples_sizes, columns=['sample_size'])
    df_results_coeff['coeff'] = results_coeff
    
    #plot results
    chart_x = results['sample_size']
    
    fig, ax = plt.subplots()

    ax.plot(chart_x, results['accuracy'],
             linestyle = '-',
             label = 'Accuracy')
    
    ax.plot(chart_x, results['precision'],
             linestyle = '--',
             label = 'Precision')
    
    ax.plot(chart_x, results['recall'],
             linestyle = '-.',
             label = 'Recall')
    
    ax.plot(chart_x, results['f1'],
             linestyle = ':',
             label = 'F1')
    
    ax.plot(chart_x, results['specificity'],
             linestyle = ':',
             label = 'specificity')
    
    ax.plot(chart_x, results['predicted_positive_rate'],
             linestyle = '-',
             label = 'Predicted positive rate')
    
    actual_positive_rate = np.repeat(y.mean(), len(chart_x))
    
    ax.plot(chart_x, actual_positive_rate,
             linestyle = '--',
             color='k',
             label = 'Actual positive rate')
    
    # Add text showing original majority class training size
    original_minority_training_sample = int(np.mean(results_raw_minority_count))
    text = 'Original minority class training size = ' + str(
        original_minority_training_sample)
    
    
    ax.text(80, 0.95,text, bbox=dict(facecolor='white', alpha=1.0))
    
    ax.set_xlabel('Minority class sample size')
    ax.set_ylabel('Score')
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc='lower right')
    ax.grid(True)
    ax.set_title(f"Plot to show the effect of oversampling the minority class ({class_descriptor}) on the prediction")

    # Set background color of Figure
    fig.patch.set_facecolor('white')
    # Set transparency of figure
    fig.patch.set_alpha(1)

    temp_filename = f"chart003_LogRegOverSampleMinClass.png"
    file_path = f'{dna_assets_path}/{temp_filename}'
    plt.savefig(file_path, bbox_inches='tight')

    plt.show()
    return results, df_results_coeff, results_coeff_dict


# --------------------------------------------------

#revised version of this function to incorporate output coefficients 
#once this is confirmed as a working function, delete the _original version
#currently working on this function (!!)

def vary_classification_thresholds(
        X_np,
        y_np,
        number_of_splits,
        rand_state
        ):
    
    #Parameters X_np and y_np need to be numpy arrays of X and y (required for k-fold)
    
    # Set up thresholds
    thresholds = np.arange(0, 1.01, 0.2)

    # Create lists for overall results (changed from arrays, so analogous to other functions)
    results_accuracy = []
    results_precision = []
    results_recall = []
    results_f1 = []
    results_specificity = [] #new addition
    results_predicted_positive_rate = []
    results_observed_positive_rate = [] #new addition
    results_auc = [] #new addition
    results_coeff = [] #new addition to capture coeffs for each sample size
    results_coeff_dict = {} #new addition to capture coeffs for each sample size - as a dict    

    """
    # Create arrays for overall results (rows=threshold, columns=k fold replicate)
    results_accuracy = np.zeros((len(thresholds),number_of_splits))
    results_precision = np.zeros((len(thresholds),number_of_splits))
    results_recall = np.zeros((len(thresholds),number_of_splits))
    results_f1 = np.zeros((len(thresholds),number_of_splits))
    results_specificity = np.zeros((len(thresholds),number_of_splits)) #new addition
    results_predicted_positive_rate = np.zeros((len(thresholds),number_of_splits))
    results_observed_positive_rate = np.zeros((len(thresholds),number_of_splits)) #new addition
    results_auc = np.zeros((len(thresholds),number_of_splits)) #new for auc
    results_coeff = [] #new addition to capture coeffs for each sample size
    results_coeff_dict = {} #new addition to capture coeffs for each sample size - as a dict
    """

    # Loop through the k-fold splits
    loop_index = 0

    # Set up thresholds
    thresholds = np.arange(0, 1.01, 0.2)

    #loop through the threshold cut-offs
    for cutoff in thresholds: #  loop 0 --> 1 on steps of 0.2
        # Get whether patients attend using cutoff

        # Set up loop for replicates for each training majority class size
        threshold_accuracy = []
        threshold_precision = []
        threshold_recall = []
        threshold_f1 = []
        threshold_specificity = [] #new addition
        threshold_predicted_positive_rate = []
        threshold_observed_positive_rate = [] #new addition
        threshold_auc = [] #new for auc

        #set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
        replicate_coeff_list_of_lists = []

        # Set up k-fold training/test splits
        #number_of_splits = num_splits
        skf = StratifiedKFold(n_splits = number_of_splits, shuffle = True, random_state=rand_state)
        skf.get_n_splits(X_np, y_np)

        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
    
            # Get X and Y train/test
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]
        
            # Get X and Y train/test
            X_train_std, X_test_std = standardise_data(X_train, X_test)
        
            # Set up and fit model
            model = LogisticRegression(solver='lbfgs')
            model.fit(X_train_std,y_train)
            
            #get coefficients for this fold
            """
            Weights with higher negative numbers mean that that feature correlates with 
            reduced chance of DNA (outcome variable, y). 
            Weights with higher positive numbers mean that that feature correlates with 
            increased chance of DNA (outcome variable, y). 
            Those weights with values closer to zero (either positive or negative) have 
            less influence in the model.
            We access model weights via the model coef_ attribute. 
            A model may predict >1 outcome label, in which case we have weights for each label. 
            This model only predicts a single label (DNA or not), so the weights are found 
            in the first element ([0]) of the coef_ attribute.
            """
            co_eff = model.coef_[0]
            replicate_coeff_list_of_lists.append(co_eff)

            # Get probability of non-survive and survive
            probabilities = model.predict_proba(X_test_std)
            # Take just the survival probabilities (column 1)
            probability_survival = probabilities[:,1]
        
            predicted_survived = probability_survival >= cutoff
            # Call accuracy measures function
            accuracy = calculate_accuracy(y_test, predicted_survived)
            # Add accuracy scores to lists
            threshold_accuracy.append(accuracy['accuracy'])
            threshold_precision.append(accuracy['precision'])
            threshold_recall.append(accuracy['recall'])
            threshold_f1.append(accuracy['f1'])
            threshold_specificity.append(accuracy['specificity']) #new addition
            threshold_predicted_positive_rate.append(accuracy['predicted_positive_rate'])
            threshold_observed_positive_rate.append(accuracy['observed_positive_rate']) #new addition
            
            #Get ROC AUC for this split, append to list of AUC's
            y_pred_proba = model.predict_proba(X_test)[::,1]
            #auc = roc_auc_score(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, predicted_survived)
            threshold_auc.append(auc)
        
        """
        # Add results to results arrays
        results_accuracy[:,loop_index] = threshold_accuracy
        results_precision[:, loop_index] = threshold_precision
        results_recall[:, loop_index] = threshold_recall
        results_f1[:, loop_index] = threshold_f1
        results_specificity[:, loop_index] = threshold_specificity #new addition
        results_predicted_positive_rate[:, loop_index] = threshold_predicted_positive_rate
        results_observed_positive_rate[:, loop_index] = threshold_observed_positive_rate #new addition
        results_auc[:, loop_index] = threshold_auc
        """
        # Add mean results to overall results
        results_accuracy.append(np.mean(threshold_accuracy))
        results_precision.append(np.mean(threshold_precision))
        results_recall.append(np.mean(threshold_recall))
        results_f1.append(np.mean(threshold_f1))
        results_specificity.append(np.mean(threshold_specificity)) #new addition
        results_predicted_positive_rate.append(np.mean(threshold_predicted_positive_rate))
        results_observed_positive_rate.append(np.mean(threshold_observed_positive_rate)) #new addition
        results_auc.append(np.mean(threshold_auc))

        #average the coefficient values produced for each fold, to get to a single array of average coefficients. 
        #these are in the same order as the columns in X
        results_coeff.append(np.mean(replicate_coeff_list_of_lists, 0))
        results_coeff_temp = np.mean(replicate_coeff_list_of_lists, 0) 
        results_coeff_dict[cutoff] = results_coeff_temp #test

        # Increment loop index
        loop_index += 1
    
    # Transfer results to dataframe (take the mean of each predictive metric over all 10 splits, for each threshold respectively)
    """
    results = pd.DataFrame(thresholds, columns=['thresholds'])
    results['accuracy'] = results_accuracy.mean(axis=1)
    results['precision'] = results_precision.mean(axis=1)
    results['recall'] = results_recall.mean(axis=1)
    results['f1'] = results_f1.mean(axis=1)
    results['specificity'] = results_specificity.mean(axis=1) #new addition
    results['predicted_positive_rate'] = results_predicted_positive_rate.mean(axis=1)
    results['observed_positive_rate'] = results_observed_positive_rate.mean(axis=1) #new addition
    results['AUC'] = results_auc.mean(axis=1)
    """
    results = pd.DataFrame(thresholds, columns=['thresholds'])
    results['accuracy'] = results_accuracy
    results['precision'] = results_precision
    results['recall'] = results_recall
    results['f1'] = results_f1
    results['specificity'] = results_specificity
    results['predicted_positive_rate'] = results_predicted_positive_rate
    results['observed_positive_rate'] = results_observed_positive_rate
    results['AUC'] = results_auc #new

    # Transfer coeff to DataFrame
    df_results_coeff = pd.DataFrame(thresholds, columns=['thresholds'])
    df_results_coeff['coeff'] = results_coeff

    return results, df_results_coeff, results_coeff_dict


# --------------------------------------------------

def use_smote_to_balance_data_inc_coeffs(
        categorical,
        X_np,
        y_np,
        num_splits,
        rand_state
        ):
    
    results_names = [
        'accuracy',
        'precision', 
        'recall', 
        'f1', 
        'specificity', 
        'predicted_positive_rate', 
        'observed_positive_rate'
        ]
    results = pd.DataFrame(index = results_names)
    
    # Set up lists to hold results for each k-fold run
    replicate_accuracy = []
    replicate_precision = []
    replicate_recall = []
    replicate_f1 = []
    replicate_specificity = []
    replicate_predicted_positive_rate = [] #new
    replicate_observed_positive_rate = [] #new
    
    replicate_auc = [] #new
    
    #set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
    coeff_list_of_lists = []

    # Set up splits
    number_of_splits = num_splits
    skf = StratifiedKFold(n_splits = number_of_splits, shuffle = True, random_state=rand_state)
    skf.get_n_splits(X, y)
    
    # Loop through the k-fold splits
    for train_index, test_index in skf.split(X_np, y_np):
        
        # Get X and Y train/test
        X_train, X_test = X_np[train_index], X_np[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]
        
        # Create an enhanced data set with SMOTENC
        smote_nc = SMOTENC(categorical_features=categorical, random_state=rand_state)
        X_resampled, y_resampled = smote_nc.fit_resample(X_train, y_train)
        
        # Standardise X data
        X_train_std, X_test_std = standardise_data(X_resampled, X_test)
        
        # Set up and fit model
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train_std, y_resampled)
        
        #get coefficients for this fold
        """
        Weights with higher negative numbers mean that that feature correlates with 
        reduced chance of DNA (outcome variable, y). 
        Weights with higher positive numbers mean that that feature correlates with 
        increased chance of DNA (outcome variable, y). 
        Those weights with values closer to zero (either positive or negative) have 
        less influence in the model.
        We access model weights via the model coef_ attribute. 
        A model may predict >1 outcome label, in which case we have weights for each label. 
        This model only predicts a single label (DNA or not), so the weights are found 
        in the first element ([0]) of the coef_ attribute.
        """
        co_eff = model.coef_[0]
        coeff_list_of_lists.append(co_eff)

        # Predict training and test set labels
        y_pred_train = model.predict(X_train_std)
        y_pred_test = model.predict(X_test_std)
        
        # Get test set proabilities
        y_pred_proba = model.predict_proba(X_test_std)[:,1]
        
        #Get ROC AUC for this split, append to list of AUC's
        auc = roc_auc_score(y_test, y_pred_proba)
        replicate_auc.append(auc)
        
        # Predict test set labels and get accuracy scores
        y_pred_test = model.predict(X_test_std)
        accuracy_scores = calculate_accuracy(y_test, y_pred_test)
        replicate_accuracy.append(accuracy_scores['accuracy'])
        replicate_precision.append(accuracy_scores['precision'])
        replicate_recall.append(accuracy_scores['recall'])
        replicate_f1.append(accuracy_scores['f1'])
        replicate_specificity.append(accuracy_scores['specificity'])
        replicate_predicted_positive_rate.append(accuracy_scores['predicted_positive_rate'])
        replicate_observed_positive_rate.append(accuracy_scores['observed_positive_rate'])
        
    # Transfer results to list and add to data frame
    smote_results = [np.mean(replicate_accuracy),
                         np.mean(replicate_precision),
                         np.mean(replicate_recall),
                         np.mean(replicate_f1),
                         np.mean(replicate_specificity),
                         np.mean(replicate_predicted_positive_rate),
                         np.mean(replicate_observed_positive_rate)
                         ]
    
    mean_auc = np.mean(replicate_auc)
    
    col_name_smote = "auc_Smote"
    df_smote_kfold = pd.DataFrame(columns=[col_name_smote])
    df_smote_kfold.loc[0] = mean_auc

    results['auc_Smote'] = smote_results

    #average the coefficient values produced for each fold, to get to a single array of average coefficients. 
    #these are in the same order as the columns in X
    average_coefficients = np.mean(coeff_list_of_lists, 0)

    co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(average_coefficients, X)

    return results, df_smote_kfold, co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna

# --------------------------------------------------

#Feature expansion
def add_polynomial_features(
        X,
        X_np,
        y_np,
        num_splits
        ):
    
    """
    Feature Expansion
    Why: 
        to increase accuracy of the prediction that the model makes
    how:
        Models such as logistic regression do not incorporate complex interactions 
        between features. If 2 features produce more than an additive effect, this 
        will not be fitted in logistic regression. In order to allow for feature 
        interaction we need to add terms that create new features by producing the 
        product of each product pair.
        When we use polynomial expansion of features, we create new features that 
        are the product of two features. 
        For example if we had 3 features, A, B and C, a full polynomial expansion 
        would produce the following extra features:
        A.A, A.B, A.C
        B.A, B.B, B.C
        C.A, C.B, C.C
        
        This is reduced in 2 ways:
            1) duplicate terms are removed (A.B and B.A are the same, so we only need A.B)
            2) The interaction_only argument is used to remove powers of single features (e.g. A.A)

    Pros:
        increases accuracy
        
    Cons:
        model may start to over-fit to the training data
        
    Dealing with over-fitting
    If over-fitting identified as an issue following use of polynomial feature expansion
    it can be handled in one (or both) of two ways:

    1) Increase the regularisation strength in the model (reduce the value of C in 
    the logistic regression model)
                                                          
    2) Use feature selection to pick only the most important features (which now 
    may include polynomial features)
    """
    
    poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X)

    #check shape of data sets (first value is number of samples, second value is number of features)
    print ('Shape of X:', X_np.shape)
    print ('Shape of X_poly:', X_poly.shape)

    #TRAINING AND TESTING NORMAL AND POLYNOMIAL MODELS
    reg_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    #Set up lists to hold results
    training_acc_results = []
    test_acc_results = []
    training_acc_results_poly = []
    test_acc_results_poly = []

    #set up lists for normal (non-polynomial) model
    #training
    training_accuracy_results = []
    training_precision_results = []
    training_recall_results = []
    training_f1_results = []
    training_specificity_results = []
    training_predicted_positive_rate_results = []
    training_observed_positive_rate_results = []
    training_roc_auc_results = []
    
    #test
    test_accuracy_results = []
    test_precision_results = []
    test_recall_results = []
    test_f1_results = []
    test_specificity_results = []
    test_predicted_positive_rate_results = []
    test_observed_positive_rate_results = []
    test_roc_auc_results = []

    #normal non-polynomial coefficients
    results_coeff = [] #new addition to capture coeffs for each sample size
    results_coeff_dict = {} #new addition to capture coeffs for each sample size 

    #set up lists for polynomial model
    #training
    training_accuracy_results_poly = []
    training_precision_results_poly = []
    training_recall_results_poly = []
    training_f1_results_poly = []
    training_specificity_results_poly = []
    training_predicted_positive_rate_results_poly = []
    training_observed_positive_rate_results_poly = []
    training_roc_auc_results_poly = []

    #test
    test_accuracy_results_poly = []
    test_precision_results_poly = []
    test_recall_results_poly = []
    test_f1_results_poly = []
    test_specificity_results_poly = []
    test_predicted_positive_rate_results_poly = []
    test_observed_positive_rate_results_poly = []
    test_roc_auc_results_poly = []

    #polynomial coefficients
    results_coeff_poly = [] #new addition to capture coeffs for each sample size
    results_coeff_dict_poly = {} #new addition to capture coeffs for each sample size 

    #Set up splits
    skf = StratifiedKFold(n_splits = num_splits)
    skf.get_n_splits(X_np, y_np)
    skf.get_n_splits(X_poly, y_np)

    #Set up model type

    for reg in reg_values:
        # Show progress
        print(reg, end=' ')
        
        # Set up lists for results for each of k splits
        training_k_results = []
        test_k_results = []
        training_k_results_poly = []
        test_k_results_poly = []

        #lists for each performance metric to capture - need to add similar lists below for poly versions, and outside loop to capture mean of the splits.
        #train - normal
        training_k_accuracy_results = []
        training_k_precision_results = []
        training_k_recall_results = []
        training_k_f1_results = []
        training_k_specificity_results = []
        training_k_predicted_positive_rate_results = []
        training_k_observed_positive_rate_results = []
        training_k_roc_auc_results = []
        
        #test - normal
        test_k_accuracy_results = []
        test_k_precision_results = []
        test_k_recall_results = []
        test_k_f1_results = []
        test_k_specificity_results = []
        test_k_predicted_positive_rate_results = []
        test_k_observed_positive_rate_results = []
        test_k_roc_auc_results = []

        #non-poly set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
        replicate_coeff_list_of_lists = []
        
        #lists to capture poly metrics
        #training - poly
        training_k_accuracy_results_poly = []
        training_k_precision_results_poly = []
        training_k_recall_results_poly = []
        training_k_f1_results_poly = []
        training_k_specificity_results_poly = []
        training_k_predicted_positive_rate_results_poly = []
        training_k_observed_positive_rate_results_poly = []
        training_k_roc_auc_results_poly = []
        
        #test - poly
        test_k_accuracy_results_poly = []
        test_k_precision_results_poly = []
        test_k_recall_results_poly = []
        test_k_f1_results_poly = []
        test_k_specificity_results_poly = []
        test_k_predicted_positive_rate_results_poly = []
        test_k_observed_positive_rate_results_poly = []
        test_k_roc_auc_results_poly = []
        
        #poly set up list to capture the coefficient values for each fold in the kfold step - to later take average to give model coefficients
        replicate_coeff_list_of_lists_poly = []

        # Loop through the k-fold splits
        for train_index, test_index in skf.split(X_np, y_np):
            
            # Normal (non-polynomial model)
            
            # Get X and Y train/test
            X_train, X_test = X_np[train_index], X_np[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]
            
            # Standardise X data
            X_train_std, X_test_std = standardise_data(X_train, X_test)
            
            # Fit model with regularisation (C)
            model = LogisticRegression(C=reg, solver='lbfgs', max_iter=1000)
            model.fit(X_train_std,y_train)
            
            #get coefficients for this fold
            """
            Weights with higher negative numbers mean that that feature correlates with 
            reduced chance of DNA (outcome variable, y). 
            Weights with higher positive numbers mean that that feature correlates with 
            increased chance of DNA (outcome variable, y). 
            Those weights with values closer to zero (either positive or negative) have 
            less influence in the model.
            We access model weights via the model coef_ attribute. 
            A model may predict >1 outcome label, in which case we have weights for each label. 
            This model only predicts a single label (DNA or not), so the weights are found 
            in the first element ([0]) of the coef_ attribute.
            """
            co_eff = model.coef_[0]
            replicate_coeff_list_of_lists.append(co_eff)

            # Predict training and test set labels
            y_pred_train = model.predict(X_train_std)
            y_pred_test = model.predict(X_test_std)
            
            # Calculate accuracy of training and test sets
            accuracy_train = np.mean(y_pred_train == y_train)
            accuracy_test = np.mean(y_pred_test == y_test)


            #enter code to calculate the performance metrics and add to list for this kfold
            #training performance - normal (non-poly)
            train_results = calculate_accuracy(y_train, y_pred_train)
            test_results = calculate_accuracy(y_test, y_pred_test)
            
            training_k_accuracy_results.append(sklearn.metrics.accuracy_score(y_train, y_pred_train))
            training_k_precision_results.append(sklearn.metrics.precision_score(y_train, y_pred_train))
            training_k_recall_results.append(sklearn.metrics.recall_score(y_train, y_pred_train))
            training_k_f1_results.append(sklearn.metrics.f1_score(y_train, y_pred_train))
            training_k_specificity_results.append(train_results["specificity"])
            training_k_predicted_positive_rate_results.append(train_results["predicted_positive_rate"])
            training_k_observed_positive_rate_results.append(train_results["observed_positive_rate"])
            training_k_roc_auc_results.append(roc_auc_score(y_train, y_pred_train))
            
            #test performance - non-poly
            test_k_accuracy_results.append(sklearn.metrics.accuracy_score(y_test, y_pred_test))
            test_k_precision_results.append(sklearn.metrics.precision_score(y_test, y_pred_test))
            test_k_recall_results.append(sklearn.metrics.recall_score(y_test, y_pred_test))
            test_k_f1_results.append(sklearn.metrics.f1_score(y_test, y_pred_test))
            test_k_specificity_results.append(test_results["specificity"])
            test_k_predicted_positive_rate_results.append(test_results["predicted_positive_rate"])
            test_k_observed_positive_rate_results.append(test_results["observed_positive_rate"])
            test_k_roc_auc_results.append(roc_auc_score(y_test, y_pred_test))

            # Record accuracy for each k-fold split
            training_k_results.append(accuracy_train)
            test_k_results.append(accuracy_test)
            
            # Polynomial model (same as above except use X with polynomial features)
            
            # Get X and Y train/test
            X_train, X_test = X_poly[train_index], X_poly[test_index]
            y_train, y_test = y_np[train_index], y_np[test_index]

            # Standardise X data
            X_train_std, X_test_std = standardise_data(X_train, X_test)
            
            # Fit model with regularisation (C)
            model = LogisticRegression(C=reg, solver='lbfgs', max_iter=1000)
            model.fit(X_train_std,y_train)

            #get coefficients for this fold - poly
            """
            Weights with higher negative numbers mean that that feature correlates with 
            reduced chance of DNA (outcome variable, y). 
            Weights with higher positive numbers mean that that feature correlates with 
            increased chance of DNA (outcome variable, y). 
            Those weights with values closer to zero (either positive or negative) have 
            less influence in the model.
            We access model weights via the model coef_ attribute. 
            A model may predict >1 outcome label, in which case we have weights for each label. 
            This model only predicts a single label (DNA or not), so the weights are found 
            in the first element ([0]) of the coef_ attribute.
            """
            co_eff = model.coef_[0]
            replicate_coeff_list_of_lists_poly.append(co_eff)
            
            # Predict training and test set labels
            y_pred_train = model.predict(X_train_std)
            y_pred_test = model.predict(X_test_std)
            
            # Calculate accuracy of training and test sets
            accuracy_train = np.mean(y_pred_train == y_train)
            accuracy_test = np.mean(y_pred_test == y_test)

            #enter code to calculate the performance metrics and add to list for this kfold for poly expanded
            train_results_poly = calculate_accuracy(y_train, y_pred_train)
            test_results_poly = calculate_accuracy(y_test, y_pred_test)
            
            #training performance - poly
            training_k_accuracy_results_poly.append(sklearn.metrics.accuracy_score(y_train, y_pred_train))
            training_k_precision_results_poly.append(sklearn.metrics.precision_score(y_train, y_pred_train))
            training_k_recall_results_poly.append(sklearn.metrics.recall_score(y_train, y_pred_train))
            training_k_f1_results_poly.append(sklearn.metrics.f1_score(y_train, y_pred_train))
            training_k_specificity_results_poly.append(train_results_poly["specificity"])
            training_k_predicted_positive_rate_results_poly.append(train_results_poly["predicted_positive_rate"])
            training_k_observed_positive_rate_results_poly.append(train_results_poly["observed_positive_rate"])
            training_k_roc_auc_results_poly.append(roc_auc_score(y_train, y_pred_train))
            
            #test performance - poly
            test_k_accuracy_results_poly.append(sklearn.metrics.accuracy_score(y_test, y_pred_test))
            test_k_precision_results_poly.append(sklearn.metrics.precision_score(y_test, y_pred_test))
            test_k_recall_results_poly.append(sklearn.metrics.recall_score(y_test, y_pred_test))
            test_k_f1_results_poly.append(sklearn.metrics.f1_score(y_test, y_pred_test))
            test_k_specificity_results_poly.append(test_results_poly["specificity"])
            test_k_predicted_positive_rate_results_poly.append(test_results_poly["predicted_positive_rate"])
            test_k_observed_positive_rate_results_poly.append(test_results_poly["observed_positive_rate"])
            test_k_roc_auc_results_poly.append(roc_auc_score(y_test, y_pred_test))

            # Record accuracy for each k-fold split
            training_k_results_poly.append(accuracy_train)
            test_k_results_poly.append(accuracy_test)
            
        # Record average accuracy for each k-fold split for normal (non-polynomial) model
        #training average accuracy - normal non-poly
        training_accuracy_results.append(np.mean(training_k_accuracy_results))
        training_precision_results.append(np.mean(training_k_precision_results))
        training_recall_results.append(np.mean(training_k_recall_results))
        training_f1_results.append(np.mean(training_k_f1_results))
        training_specificity_results.append(np.mean(training_k_specificity_results))
        training_predicted_positive_rate_results.append(np.mean(training_k_predicted_positive_rate_results))
        training_observed_positive_rate_results.append(np.mean(training_k_observed_positive_rate_results))
        training_roc_auc_results.append(np.mean(training_k_roc_auc_results))
        
        #test average accuracy - normal non-poly
        test_accuracy_results.append(np.mean(test_k_accuracy_results))
        test_precision_results.append(np.mean(test_k_precision_results))
        test_recall_results.append(np.mean(test_k_recall_results))
        test_f1_results.append(np.mean(test_k_f1_results))
        test_specificity_results.append(np.mean(test_k_specificity_results))
        test_predicted_positive_rate_results.append(np.mean(test_k_predicted_positive_rate_results))
        test_observed_positive_rate_results.append(np.mean(test_k_observed_positive_rate_results))
        test_roc_auc_results.append(np.mean(test_k_roc_auc_results))

        #average the coefficient values produced for each fold, in non-poly model 
        # to get to a single array of average coefficients. 
        #these are in the same order as the columns in X
        results_coeff.append(np.mean(replicate_coeff_list_of_lists, 0))
        results_coeff_temp = np.mean(replicate_coeff_list_of_lists, 0) 
        results_coeff_dict[reg] = results_coeff_temp #test

        # Record average accuracy for each k-fold split for polynomial model
        #training average accuracy - poly
        training_accuracy_results_poly.append(np.mean(training_k_accuracy_results_poly))
        training_precision_results_poly.append(np.mean(training_k_precision_results_poly))
        training_recall_results_poly.append(np.mean(training_k_recall_results_poly))
        training_f1_results_poly.append(np.mean(training_k_f1_results_poly))
        training_specificity_results_poly.append(np.mean(training_k_specificity_results_poly))
        training_predicted_positive_rate_results_poly.append(np.mean(training_k_predicted_positive_rate_results_poly))
        training_observed_positive_rate_results_poly.append(np.mean(training_k_observed_positive_rate_results_poly))
        training_roc_auc_results_poly.append(np.mean(training_k_roc_auc_results_poly))
        
        #test average accuracy - poly
        test_accuracy_results_poly.append(np.mean(test_k_accuracy_results_poly))
        test_precision_results_poly.append(np.mean(test_k_precision_results_poly))
        test_recall_results_poly.append(np.mean(test_k_recall_results_poly))
        test_f1_results_poly.append(np.mean(test_k_f1_results_poly))
        test_specificity_results_poly.append(np.mean(test_k_specificity_results_poly))
        test_predicted_positive_rate_results_poly.append(np.mean(test_k_predicted_positive_rate_results_poly))
        test_observed_positive_rate_results_poly.append(np.mean(test_k_observed_positive_rate_results_poly))
        test_roc_auc_results_poly.append(np.mean(test_k_roc_auc_results_poly))

        #average the coefficient values produced for each fold, in poly model 
        # to get to a single array of average coefficients. 
        #these are in the same order as the columns in X
        results_coeff_poly.append(np.mean(replicate_coeff_list_of_lists_poly, 0))
        results_coeff_temp = np.mean(replicate_coeff_list_of_lists_poly, 0) 
        results_coeff_dict_poly[reg] = results_coeff_temp #test

        #original code
        training_acc_results.append(np.mean(training_k_results))
        test_acc_results.append(np.mean(test_k_results))
        training_acc_results_poly.append(np.mean(training_k_results_poly))
        test_acc_results_poly.append(np.mean(test_k_results_poly))

    #NEXT STEPS:
    #add in code from the threshold or undersampling function, to create df of the various regularisation values and performance metrics for each
    #and another df for auc - DONE
    #return this, then outside the functoin, call the function to identify the max value and store in the growing dfs
    #also, look at the coeff code, to see if can identify a way to produce the coeffs from this code and store for later use
    #later we can identify the optimal model, and return the coeffs for that model? to make this work all models need this revision making - test on this function

    # Transfer normal (non-poly) results to dataframe
    df_test_results = pd.DataFrame(reg_values, columns=['reg_values'])
    df_test_results['accuracy'] = test_accuracy_results
    df_test_results['precision'] = test_precision_results
    df_test_results['recall'] = test_recall_results
    df_test_results['f1'] = test_f1_results
    df_test_results['specificity'] = test_specificity_results
    df_test_results['predicted_positive_rate'] = test_predicted_positive_rate_results
    df_test_results['observed_positive_rate'] = test_observed_positive_rate_results
    df_test_results['AUC'] = test_roc_auc_results #new

    # Transfer poly results to dataframe
    df_poly_test_results = pd.DataFrame(reg_values, columns=['reg_values'])
    df_poly_test_results['accuracy'] = test_accuracy_results_poly
    df_poly_test_results['precision'] = test_precision_results_poly
    df_poly_test_results['recall'] = test_recall_results_poly
    df_poly_test_results['f1'] = test_f1_results_poly
    df_poly_test_results['specificity'] = test_specificity_results_poly
    df_poly_test_results['predicted_positive_rate'] = test_predicted_positive_rate_results_poly
    df_poly_test_results['observed_positive_rate'] = test_observed_positive_rate_results_poly
    df_poly_test_results['AUC'] = test_roc_auc_results_poly #new

    # Transfer non-poly coeff to DataFrame
    df_results_coeff = pd.DataFrame(reg_values, columns=['reg_values'])
    df_results_coeff['coeff'] = results_coeff

    # Transfer poly coeff to DataFrame
    df_results_coeff_poly = pd.DataFrame(reg_values, columns=['reg_values'])
    df_results_coeff_poly['coeff'] = results_coeff_poly

    #original code
    best_test_non_poly = np.max(test_acc_results)
    best_test_poly = np.max(test_acc_results_poly)
    best_test_auc_non_poly = np.max(test_roc_auc_results)
    best_test_auc_poly = np.max(test_roc_auc_results_poly)
    print ("\nBest accuracy for non-poly and poly were:")
    print(f"Non-poly:{best_test_non_poly:.3f}")
    print(f"Poly: {best_test_poly:.3f}")
    print ("\nBest AUC for non-poly and poly were:")
    print(f"Non-poly:{best_test_auc_non_poly:.3f}")
    print(f"Poly: {best_test_auc_poly:.3f}")

    return(
        reg_values,
        training_acc_results,
        test_acc_results,
        training_acc_results_poly,
        test_acc_results_poly,
        poly,
        X_poly,
        df_test_results,
        df_poly_test_results,
        best_test_auc_non_poly,
        best_test_auc_poly,
        df_results_coeff,
        results_coeff_dict,
        df_results_coeff_poly,
        results_coeff_dict_poly
        )

# --------------------------------------------------

#informed by https://youtu.be/j7WGVhF7g3M 
#optimizing regularisation strength (Logistic Regression)

def get_results(trial):
    C = trial.suggest_loguniform('C', 0.00001, 10000.0) 
    model = LogisticRegression(C=C)
    model.fit(X_train, y_train)
    # Get test set proabilities
    y_pred_proba = model.predict_proba(X_test_std)[:,1]
    #Get ROC AUC for this split, append to list of AUC's
    auc = roc_auc_score(y_test, y_pred_proba)
    return auc

'''
###### Currently not needed as have standardised columns in processing file ######
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
'''
# --------------------------------------------------
# --------------------------------------------------
#<<<     End of defining functions     >>>
# --------------------------------------------------


# --------------------------------------------------
#<<<     CODE BEGINS     >>>
# --------------------------------------------------

#----------------------------------------------------
#create DataFrame into which all metrics will later be added
#----------------------------------------------------
"""
Definitions of summary metrics:
    Precision
    Also known as the positive predictive value, this is the fraction of
    relevant instances among all the retrieved instances.
    Calculated as the number of true positives divided by the total 
    number of positive predictions i.e., the number of true positives plus 
    the number of false positives. 
    Precision can be seen as a measure of quality - higher precision means 
    an algorithm returns more relevant results than irrelevant ones.
    ref: https://en.wikipedia.org/wiki/Precision_and_recall
    
    Recall
    Also known as sensitivity, this is the fraction of relevant instances that
    were retrieved.  
    Recall can be seen as a measure of quantity. High recall means that an 
    algorithm returns most of the relevant results, whether or not irrelevant 
    ones are also returned.

    f1
    This is the harmonic mean of precision and recall. 
    Calculated as precision * recall / (precision + recall)
    The highest possible value of an F-score is 1.0, indicating perfect 
    precision and recall, and the lowest possible value is 0, if either the 
    precision or the recall is zero
    
    Specificity
    The proportion of true negatives that are correctly predicted by the model
    
"""

#create empty dictionaries to later add to with the coefficients from each model in the format model name (keys) df_of_coeffs (values)
dict_coeff_higher_chance_dna_each_model = {}
dict_coeff_lower_chance_dna_each_model = {}


#create empty DataFrame
df_summary_of_models = pd.DataFrame()

#list of metrics, with identical names to those in dict_mean_test_performance 
#and dict_mean_train_performance (added accuracy, predicted_positive_rate, observed_positive_rate)
list_of_metrics_for_summary = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'specificity',
    'predicted_positive_rate', 
    'observed_positive_rate'
    ]

#Iterate over list of metrics for summary, add to df above, and set these metric labels as the index
dict_item_counter = 0
dict_metrics_summary = {}
for item in list_of_metrics_for_summary:
    dict_metrics_summary[dict_item_counter] = item
    dict_item_counter += 1

df_summary_of_models['performance_metrics'] = list_of_metrics_for_summary
df_summary_of_models.set_index('performance_metrics', inplace=True)
"""
###### May be needed if it tries to load the whole darn file #####
#code to show all columns but only 20 rows in previewed dfs
pd.options.display.max_columns = None
pd.options.display.max_rows = 20
"""
#----------------------------------------------------
#Loading Data Section
#----------------------------------------------------

#Load data 
data = pd.read_csv('processed_data/1B_DNA_profile/processed_data_1B_DNA_Profile_new_test.csv')

# Make all data 'float' type
data = data.astype(float)

#data = data.drop('appt_date_time',1)
#data.describe()

"""
Looking at a summary of patients who attended or did not attend
Before running ML models, good to look at your data. 
Here we separate patients who attended from those who DNA'd, and we look at 
differences in features. We use a mask to select and filter patients.
"""

#create the mask to filter the data to those who DNAd their appointment
#attend_status = assign_field_name_to_variable(filename, "AssignFieldNamesToVars", 7).lower() #no longer required as mapped data field names in processing file
#attend_status = 'attend_status'

#first remove the attended column
if "encoded_attend_status_with_missing_Attended" in data.columns:
    data.drop("encoded_attend_status_with_missing_Attended", axis=1, inplace=True)

elif "encoded_attend_status_Attended" in data.columns:
    data.drop("encoded_attend_status_Attended", axis=1, inplace=True)

else:
    pass

#then create a mask for the dna column
if "encoded_attend_status_with_missing_DNA" in data.columns:
    mask_text = f"encoded_attend_status_with_missing_DNA"
    mask_dna = data[mask_text] == 1 # Mask for patients who did not attend
    mask_attend = data[mask_text] == 0 # Mask for patients who did not DNA

elif "encoded_attend_status_DNA" in data.columns:
    mask_text = f"encoded_attend_status_DNA"
    mask_dna = data[mask_text] == 1 # Mask for patients who did not attned
    mask_attend = data[mask_text] == 0 # Mask for patients who did not dna

else:
    pass

"""
Looking at a summary of patients who attended or did not attend
Before running ML models, good to look at your data. 
Here we separate patients who attended from those who DNA'd, and we look at 
differences in features. We use a mask to select and filter patients.
"""
#mask = data['encoded_attend_status_with_missing_Attended'] == 1 # Mask for patients who attended
attended = data[mask_attend] # filter using mask for thoes who did not attend

#mask = data['encoded_attend_status_with_missing_Attended'] == 0 # Mask for patients who DNA'd
dna = data[mask_dna] # filter using mask

#Now let's look at average (mean) values for attended and dna.
#We can make looking at them side by side more easy by putting these values in a new DataFrame
summary = pd.DataFrame() # New empty DataFrame
summary['attended'] = attended.mean()
summary['dna'] = dna.mean()
summary


# --------------------------------------------------

"""
Divide into X (features) and y (labels)
We will separate out our features (the data we use to make a prediction) 
from our label (what we are trying to predict). By convention our features are 
called X (upper case to denote multiple features), and the label (survived or 
not) y.
"""
# X = all 'data' except the 'encoded_attended' column, these are the features 
#we are using to predict the label
X = data.drop(mask_text,axis=1) 

# y = 'encoded_dna' column from 'data', this is the label we are trying 
#to predict
y = data[mask_text] #this is using DNA status as y variable

#convert the DataFrames to numpy arrays (NumPy arrays are simpler to refer to 
#by row/column index numbers, and sklearn’s k-fold method provides row indices 
#for each set)
X_np = X.values
y_np = y.values


#----------------------------------------------------
#Plotting the learning curve and sample size selection
#----------------------------------------------------

#Calculate the maximum training set size we can have

data_rows = X.shape[0]
max_training_size = int(data_rows * (1 - test_fraction))
print(f"Max training size: {max_training_size}")

#set the numpy random seed using global variable rand_state
np.random.seed(rand_state)

#call get_values_for_learning_curve_log_reg function
results_accuracy, results_training_size= get_values_for_learning_curve_log_reg(
        test_fraction, 
        max_training_size,
        X_np,
        y_np,
        rand_state,
        sample_size_increment)

#Plot the learning curve using the plot_learning_curve function
plot_learning_curve(results_training_size, results_accuracy)

#reset the numpy random seed using global variable rand_state
#This is reset so that if a sample is selected in the subsequent code this "choice"
#is identical to the one previously used to produce the learning curve 
#np.random.seed(rand_state)

print("\nPLEASE NOTE: If you are limited by computational power, you can use a sample of the full dataset. Be aware though, small datasets are susceptible to over-fitting.")
use_all_data = input("Press y to use all data, or n to enter a sample size to use: ").lower()
if use_all_data == "n":
    sample_size = int(input(f"\nPlease enter a sample size less than or equal to {max_training_size}. "))
    
    #use the user-provided sample size to run analysis on a subset of the data
    #no "else" as this is implicit - if no sample size given, all data used
    random_selection = np.random.choice(len(data), sample_size, replace=False)
    
    X = X.iloc[list(random_selection)]
    y = y.iloc[list(random_selection)]
    X_np = X.values
    y_np = y.values

else:
    sample_size = max_training_size

#call function to display another version of the learning curve
#this time, displaying the selected sample size on the curve with a bold red X
#then, save this chart for use in the summary report
fig_selected_sample_size_accuracy = plot_learning_curve_with_sample_size_new(results_training_size, results_accuracy, sample_size)
#fig_selected_sample_size_accuracy.savefig(f"{dna_assets_path}/chart001_fig_selected_sample_size_accuracy.png")

# --------------------------------------------------
#Model 1: Baseline 'out of the box' log regression
#without k-fold cross validation or other predictive 
#boosting methods applied
# --------------------------------------------------

model_name = 'auc_Baseline_LogReg'

#Divide into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = True, random_state = rand_state)

#Standardise data
"""
This is to get all features on roughly the same scale.
Benefit is this generally leads to a better model and makes it easier to 
compare the relative importance of different features
A common method is standardisation, where the mean and standard deviation of 
the training set is used to normalise the data. 
We subtract the mean of the test set values, and divide by the standard 
deviation of the training data. 
Note that the mean and standard deviation of the training data are used to 
standardise the test set data as well.
"""
X_train_std, X_test_std = standardise_data(X_train, X_test)

#Fit logistic regression model using y and standardised X
model = LogisticRegression()
model.fit(X_train_std, y_train)

#Predict training and test set labels
y_pred_train = model.predict(X_train_std)
y_pred_test = model.predict(X_test_std)

#Calculate accuracy
accuracy_dict_baseline_train = calculate_accuracy(y_train, y_pred_train)
accuracy_dict_baseline_test = calculate_accuracy(y_test, y_pred_test)

#Test print statements to compare the test and train performance on the "accuracy" metric
print(f"Training Accuracy: {accuracy_dict_baseline_train['accuracy']}")
print(f"Testing Accuracy: {accuracy_dict_baseline_test['accuracy']}")

#Call function to add the summary performance metrics of this model run to the summary
df_summary_of_models = add_performance_metrics_to_summary(
    df_summary_of_models,
    dict_metrics_summary,
    accuracy_dict_baseline_test,
    model_name
    )

#Get ROC AUC for this split, append to list of AUC's
y_pred_proba = model.predict_proba(X_test_std)[::,1]
auc = roc_auc_score(y_test, y_pred_proba)
col_name_base_log_reg = model_name #previously "auc_BaseLogReg"
df_baseline_logreg_auc_no_folds = pd.DataFrame(columns=[col_name_base_log_reg])
df_baseline_logreg_auc_no_folds.loc[0] = auc

# --------------------------------------------------

#code to add to all models after the model is fit with X and y
#to do this i need to add the following as inputs to each function for the model if not already present:

# dict_coeff_lower_chance_dna_each_model
# dict_coeff_higher_chance_dna_each_model
# top_x

#and add into the return statement the following, which should be the growing dictionaries with the coefficients for the given model, added
# to the prexisting key-value pairs of model-df_of_coeff_values:

# dict_coeff_higher_chance_dna_each_model 
# dict_coeff_lower_chance_dna_each_model

co_eff_df, df_reduced_chance, df_increased_chance = id_coefficients(model)
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(df_reduced_chance, df_increased_chance, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name)

# --------------------------------------------------

#DEV NOTE
#Should this section take place at the end once best performing model ID'd from 
#AUC df and then produced final model with optimised AUC, then run the coefficient code
#below?

#Examining the model co-efficients
"""
Weights with higher negative numbers mean that that feature correlates with 
reduced chance of DNA (outcome variable, y). 
Weights with higher positive numbers mean that that feature correlates with 
increased chance of DNA (outcome variable, y). 
Those weights with values closer to zero (either positive or negative) have 
less influence in the model.
We access model weights via the model coef_ attribute. 
A model may predict >1 outcome label, in which case we have weights for each label. 
This model only predicts a single label (DNA or not), so the weights are found 
in the first element ([0]) of the coef_ attribute.
"""

co_eff = model.coef_[0]

#Transferring the weights array to a Pandas DataFrame to improve readability 
"""
The array order is in the same order of the list of features of X, 
so we will put that those into the DataFrame as well. 
We will sort by influence in the model. Because both large negative and positive 
values are more influential in the model we will take the absolute value of the 
weight (that is remove any negative sign), and then sort by that absolute value. 
That will give us a more readable table of most influential features in the model.
"""

co_eff_df = pd.DataFrame() # create empty DataFrame
co_eff_df['feature'] = list(X) # Get feature names from X
co_eff_df['co_eff'] = co_eff
co_eff_df['abs_co_eff'] = np.abs(co_eff)
co_eff_df.sort_values(by='abs_co_eff', ascending=False, inplace=True)

#use np.sign to label any negative value as -1 and any positive value as +1
co_eff_from_df = co_eff_df['co_eff'].values
co_eff_df['sign_(pos=higher_dna_chance)'] = np.sign(co_eff_from_df)

#creating 2 subset DataFrames from co_eff_df
"""
one for factors associated with higher chance of dna (positive sign)
one for factors associated with lower chance of dna (negative sign)
"""
co_eff_df_reduced_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == -1]
co_eff_df_higher_chance_dna = co_eff_df.loc[co_eff_df['sign_(pos=higher_dna_chance)'] == 1]

#Print the df of ALL features ordered by most influential to least influential
print(co_eff_df)

# --------------------------------------------------

#Print the subset df of top 5 features ordered by most influential to least influential, that have a reduced chance of dna
#TO GO INTO OUTPUT REPORT
#print(co_eff_df_reduced_chance_dna)
features = list(co_eff_df_reduced_chance_dna['feature'][:top_x])
print(f"\nThe most influential {top_x} features reducing the chance of a DNA are:")
for feature in features:
    if feature[:8] == "encoded_":
        print(feature[8:]) #this slice removes the prefix 'encoded_' where present
    else:
        print(feature)

#Print the subset df of top 5 features ordered by most influential to least influential, that have an increased chance of dna
#TO GO INTO OUTPUT REPORT
#print(co_eff_df_higher_chance_dna)
features = list(co_eff_df_higher_chance_dna['feature'][:top_x])
print(f"\nThe most influential {top_x} features increasing the chance of a DNA are:")
for feature in features:
    if feature[:8] == "encoded_":
        print(feature[8:]) #this slice removes the prefix 'encoded_' where present
    else:
        print(feature)

# --------------------------------------------------

#Show predicted probabilities
"""
The predicted probabilities are for the two alternative classes 
0 (does not survive/does not dna) or 1 (survive/dna).
Ordinarily we do not see these probabilities - the predict method used above 
applies a cut-off of 0.5 to classify passengers/patient into survived/dna or not, 
but we can see the individual probabilities for each passenger/patient.
"""

# Show first ten predicted classes
classes = model.predict(X_test_std)
classes[0:10] #test print

# --------------------------------------------------

#Show first ten predicted probabilities 
"""
(note how the values relate to the classes predicted above)
Each patient has two values. 
These are the probability of NOT DNA'ing (first value) or DNA'ing (second value). 
As we only have two possible classes we need only look at one. 
Multiple values are important when there is >1 class being predicted.
"""

probabilities = model.predict_proba(X_test_std)
#probabilities[0:10] #test print
for prob in probabilities[0:10]:
    print(prob[1])

# --------------------------------------------------
#<<<     End of baseline log reg code     >>>
# --------------------------------------------------


# --------------------------------------------------
#Model variant 2: Log reg with kfold validation applied
# --------------------------------------------------

model_name = f"auc_BaseLogRegKFold_{number_of_splits}Folds"

mean_accuract_dict_test, mean_accuract_dict_train, observed, predicted_proba, results_model_probability,  results_fraction_positive, skf, df_logreg_kfold, co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = run_log_reg_within_k_fold_and_output_accuracy_scores(
        number_of_splits,
        rand_state,
        X,
        X_np,
        y_np,
        number_of_bins,
        model_name
        )

#append output auc for this model to the df of all models' auc
combined_auc_df = pd.concat([df_baseline_logreg_auc_no_folds, df_logreg_kfold], axis=1)


# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name)

# --------------------------------------------------

#amend the wording of the list of mean metrics to have prefix of "mean_" so as to align with the dict output above
list_of_metrics_for_summary_mean = [f"mean_{metric}" for metric in list_of_metrics_for_summary]
number_of_measures = int(len(list_of_metrics_for_summary_mean))

#taking just the metrics in the summary list from the dict, convert the key-value pairs for this subset of metrics to a DataFrame
cols = list(mean_accuract_dict_test.keys())
temp_dict = {}

counter=0
while counter < number_of_measures:
    value = mean_accuract_dict_test[list_of_metrics_for_summary_mean[counter]]
    temp_dict[list_of_metrics_for_summary[counter]] = value.astype(list)
    counter+=1

#create a list consisting of 1 value, the name of the model created previously in the code, for this specific model
#this is required as a parameter when creating the subset df
model_name_as_list = []
model_name_as_list.append(model_name)

#create subset df of performance measures for the log reg with kfold applied
df_logreg_kfold_metrics = pd.DataFrame.from_dict(temp_dict, orient='index', columns=model_name_as_list)

# --------------------------------------------------

combined_performance_metrics_df = pd.concat([df_summary_of_models, df_logreg_kfold_metrics], axis=1)

# --------------------------------------------------
#Model 2 ends
# --------------------------------------------------


# --------------------------------------------------
#Model variant 3: Log reg with methods to balance data applied
# --------------------------------------------------
#Method 3-01: Undersampling majority class
# --------------------------------------------------

#identify majority class and minority class
unique_elements, counts_elements = np.unique(y, return_counts=True)
index_max = counts_elements.argmax()
majority_class = unique_elements[index_max]
if majority_class == 1:
    majority_class_descriptor = "DNA"
    minority_class_descriptor = "attend"
else:
    majority_class_descriptor = "attend"
    minority_class_descriptor = "DNA"

print("\n")
print("\n<<< Identifying the majority and minority classes >>>")    
print(f"Majority class = patients who {majority_class_descriptor} appointments")
print(f"Minority class = patients who {minority_class_descriptor} appointments")

# --------------------------------------------------

results_undersample_maj_class, df_results_coeff, results_coeff_dict = run_log_reg_undersample_maj_class_new(
        sample_size,
        X,
        X_np,
        y_np,
        number_of_splits,
        rand_state,
        majority_class_descriptor,
        dna_assets_path
        )

# --------------------------------------------------

#identify the optimum sample size returning the highest AUC when undersampling the majority class
df_temp_undersample_auc = id_highest_auc_return_df(
        results_undersample_maj_class, 
        "AUC", 
        "sample_size", 
        "auc_UndersampleMaj_n="
        )

# --------------------------------------------------

#using the highest returned AUC, locate the sample size for this AUC, and associated metrics
#produce df of these
model_name = list(df_temp_undersample_auc.columns)[0]
optimum_auc_undersample_size = int(list(df_temp_undersample_auc.columns)[0].split('=')[1])
df_optimum_auc_undersample = results_undersample_maj_class.loc[results_undersample_maj_class['sample_size'] == optimum_auc_undersample_size]
df_optimum_auc_undersample_subset = df_optimum_auc_undersample.drop(columns=["sample_size", "AUC"])
df_optimum_auc_undersample_transpose = df_optimum_auc_undersample_subset.transpose()
df_optimum_auc_undersample_transpose.rename(columns={df_optimum_auc_undersample_transpose.columns[0]: model_name},inplace=True)

#add model name (key) and parameter (value) to dictionary
dict_of_optimal_auc_params = {}
dict_of_optimal_auc_params[model_name] = optimum_auc_undersample_size

# --------------------------------------------------

#create x3 DataFrames for 1) all coefficients in descending order 2) as before but just those which are associated with reduced DNA likelihood and 3) just those associated
#with higher DNA likelihood.

co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(results_coeff_dict[optimum_auc_undersample_size], X)
#co_eff_df

# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_auc_df = pd.concat([combined_auc_df, df_temp_undersample_auc], axis=1)

# --------------------------------------------------

combined_performance_metrics_df = pd.concat([combined_performance_metrics_df.copy(), df_optimum_auc_undersample_transpose], axis=1)

# --------------------------------------------------
#Model variant 4: Log reg with methods to balance 
# data applied
# --------------------------------------------------
#Method 02: Oversampling minority class
# --------------------------------------------------

#Call function to run the logistic regression model with the oversampled
#minority class. This returns the results in a DataFrame.

#DEV NOTE TO DO: need to combine these results into the master results summary df which
#might require the function to be adjusted??
#also need to consider how to incorporate ROC AUC into the function as a performance measure

results_oversample_min_class, df_results_coeff, results_coeff_dict = run_log_reg_oversample_min_class_new(
        sample_size,
        X,
        X_np,
        y_np,
        number_of_splits,
        rand_state,
        minority_class_descriptor,
        dna_assets_path)

"""
#original function call - delete once above revised call to new function is confirmed as working
results_oversample_min_class = run_log_reg_oversample_min_class(
        sample_size,
        X_np,
        y_np,
        number_of_splits,
        rand_state,
        minority_class_descriptor)
"""

# --------------------------------------------------

#identify the optimum sample size returning the highest AUC when oversampling the minority class
df_temp_oversample_auc = id_highest_auc_return_df(
        results_oversample_min_class, 
        "AUC", 
        "sample_size", 
        "auc_OversampleMin_n="
        )

# --------------------------------------------------

#using the highest returned AUC, locate the sample size for this AUC, and associated metrics
#produce df of these
model_name_oversample = list(df_temp_oversample_auc.columns)[0]
optimum_auc_oversample_size = int(list(df_temp_oversample_auc.columns)[0].split('=')[1])
df_optimum_auc_oversample = results_oversample_min_class.loc[results_oversample_min_class['sample_size'] == optimum_auc_oversample_size]
df_optimum_auc_oversample_subset = df_optimum_auc_oversample.drop(columns=["sample_size", "AUC"])
df_optimum_auc_oversample_transpose = df_optimum_auc_oversample_subset.transpose()
df_optimum_auc_oversample_transpose.rename(columns={df_optimum_auc_oversample_transpose.columns[0]: model_name_oversample},inplace=True)

#add model name (key) and parameter (value) to dictionary
dict_of_optimal_auc_params[model_name_oversample] = optimum_auc_oversample_size

# --------------------------------------------------

#create x3 DataFrames for 1) all coefficients in descending order 2) as before but just those which are associated with reduced DNA likelihood and 3) just those associated
#with higher DNA likelihood.
co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(results_coeff_dict[optimum_auc_oversample_size], X)
#co_eff_df

# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna, top_x, model_name_oversample)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name_oversample)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name_oversample)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_auc_df = pd.concat([combined_auc_df, df_temp_oversample_auc], axis=1)

# --------------------------------------------------

combined_performance_metrics_df = pd.concat([combined_performance_metrics_df, df_optimum_auc_oversample_transpose], axis=1)

# --------------------------------------------------
#Model Variant 5: Log reg with methods to balance data
#set applied
# --------------------------------------------------
#Method 03: varying the imbalanced data threshold
# --------------------------------------------------

"""
Confirmed by HSMA mentor: changing the classification threshold should not 
change the coefficients for the model. Here, the revised function 
"vary_classification_thresholds" doesn't change coefficients (correct, 
confirmed by mentor), so while the varying thresholds will affect AUC and 
other metrics, the coefficients are currently identical to the kfold 
model.
"""

# --------------------------------------------------


#replacement cell, calling revised function to include output coefficients
# note: the coefficients are the same regardless of the change made to the classification threshold
# Need to confirm if this is correct 
df_adjust_classification_threshold_auc_test, df_results_coeff_test, results_coeff_dict_test = vary_classification_thresholds(
    X_np,
    y_np,
    number_of_splits,
    rand_state
    )

# --------------------------------------------------

#identify the optimum threshold returning the highest AUC when varying the classification threshold
df_temp_max_auc_varying_threshold = id_highest_auc_return_df(
        df_adjust_classification_threshold_auc_test, 
        "AUC", 
        "thresholds", 
        "auc_Threshold_n="
        )


# --------------------------------------------------

#using the highest returned AUC, locate the sample size for this AUC, and associated metrics
#produce df of these
model_name_vary_threshold = list(df_temp_max_auc_varying_threshold.columns)[0]
optimum_auc_threshold_value = float(list(df_temp_max_auc_varying_threshold.columns)[0].split('=')[1])
#df_optimum_auc_threshold = df_adjust_classification_threshold_auc_test.loc[df_adjust_classification_threshold_auc_test['thresholds'] == optimum_auc_threshold_value] #original code, errors ad hoc
df_optimum_auc_threshold = df_adjust_classification_threshold_auc_test.loc[lambda df : abs(df['thresholds'] - optimum_auc_threshold_value) < 0.001] #revised logic from Drew to control for how floats are stored
df_optimum_auc_threshold_subset = df_optimum_auc_threshold.drop(columns=["thresholds", "AUC"])
df_optimum_auc_threshold_transpose = df_optimum_auc_threshold_subset.transpose()
df_optimum_auc_threshold_transpose.rename(columns={df_optimum_auc_threshold_transpose.columns[0]: model_name_vary_threshold},inplace=True)

#add model name (key) and parameter (value) to dictionary
dict_of_optimal_auc_params[model_name_vary_threshold] = optimum_auc_threshold_value

# --------------------------------------------------
optimum_auc_threshold_value_actual_from_df = df_optimum_auc_threshold['thresholds'].values[0] #get the actual value from the cell for the optimum threshold, which due to how floats are stored may be 0.60000001 and not 0.6 for e.g.

co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = create_coefficients_df(results_coeff_dict_test[optimum_auc_threshold_value_actual_from_df], X)

# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna, top_x, model_name_vary_threshold)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name_vary_threshold)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name_vary_threshold)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_auc_df = pd.concat([combined_auc_df, df_temp_max_auc_varying_threshold], axis=1)

# --------------------------------------------------

combined_performance_metrics_df = pd.concat([combined_performance_metrics_df, df_optimum_auc_threshold_transpose], axis=1)

# --------------------------------------------------
#Model variant 6: Log reg with methods to balance
#the data set applied
# --------------------------------------------------
#Method 04: Applying Synthetic Minority Over-sampling Technique (SMOTE)
# --------------------------------------------------

"""
From Mike Allen's titanic workbook:
Synthetic Minority Over-sampling Technique (SMOTE) is used to enhance the 
minority class with synthetic data. Essentially, SMOTE creates 
new cases by interpolating between two existing near-neighbour cases. SMOTE 
rebalances the data set, synthetically enhancing the minority class so that 
the number of minority examples are increased to match the number of majority 
samples.
"""

# Create list of non-categorigcal features - for use in SMOTENC to balance dataset
#While the pre-processing will ensure the structure of the df is consistent
#The variable names will change relative to the col names in the raw data set
#However, when all categorical were converted to 0s and 1s using one hot encoding 
#the prefix "encoded_" was added to the column/feature name. 
#The below code locates all cols/features that do not start with "encoded_" 
#and adds them to the non_cat list
non_cat = []
for feature in list(X):
    if feature[:8] != "encoded_":
        non_cat.append(feature)

# Create an array of ones for all features
number_of_features = X.shape[1]
categorical_array = np.ones(number_of_features)

# Assign non_categorical features in our 'categorical' array
features = list(X)
for index, feature in enumerate(features):
    if feature in non_cat:
        print ('Set {:} to non-categorical'.format(feature))
        categorical_array[index] = 0
 
# Get catagorical indices
categorical = np.where(categorical_array == 1)[0]

# Print our categorical array
print ('Categorical features')
print (categorical)

# --------------------------------------------------

#new cell calling revised smote function to incorporate coefficients output

#call smote function to test working
smote_results, df_smote_kfold, co_eff_df, co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna = use_smote_to_balance_data_inc_coeffs(
    categorical,
    X_np,
    y_np,
    number_of_splits,
    rand_state
    )

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_auc_df = pd.concat([combined_auc_df, df_smote_kfold], axis=1)

# --------------------------------------------------

combined_performance_metrics_df = pd.concat([combined_performance_metrics_df, smote_results], axis=1)

# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
model_name = "auc_Smote"
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(co_eff_df_reduced_chance_dna, co_eff_df_higher_chance_dna, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name)

# --------------------------------------------------

"""
Notes from Mike Allen's tutorial-
By default, SMOTE rebalances the data set, synthetically enhancing the minority 
class so that the number of minority examples are increased to match the number 
of majority samples. 
A risk of SMOTE is that it may over-compensate and lead to over-estimation of 
the occurrence of the minority class. 
To try to combat this risk of over-estimating the minority class, we 
fine-tune SMOTE by passing a dictionary of values for both the majority class 
(attended), and the minority class (DNA'd'). 
We will fix SMOTE to return 500 patients who attended, and vary the number of 
patients who DNA'd
"""

#get the size of each class of y (0 = attended, 1 = DNA)
size_y_equals_0 = (y==0).sum() #attended appointments
size_y_equals_1 = (y==1).sum() #did not attend appointment. y==1.sum() ensures int type

#round y==0 to nearest 50
size_y_equals_0_rounddown = (size_y_equals_0 - (size_y_equals_0%50))

#Create a list of dictionaries, with each dictionary a balance of attend:DNA
#The default parameter set at the start of this file is 150. 
#This assumes the data used in the model contains at least 150 attendances.
smote_alterantive_samples = []

#create a list of sample sizes, from the min_number_attendances parameter,
#to the size of the DNA class, in increments of 50. 
#This assumes the class of y==1 is always the minority class (likely true for 
#DNA use case, but would need validating if the code was to be used in other 
#binary use cases)
DNA_sample_sizes = list(range(min_number_attendances, size_y_equals_0_rounddown+50, 50))

for sample_size in DNA_sample_sizes:
    smote_input = dict()
    smote_input[0] = size_y_equals_0_rounddown # always have the actual number of attending patients in returned sample rounded down to nearest 50
    smote_input[1] = sample_size
    smote_alterantive_samples.append(smote_input)


# --------------------------------------------------
#Model variant 7: Log reg with feature expansion applied
# --------------------------------------------------

"""
to do: check whether coefficients added in to this function as outputs 
are working as expected.
Work through feature reduction tutorial and consider whether required to
add into this code.
"""

# --------------------------------------------------

poly = PolynomialFeatures(2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X)

# --------------------------------------------------

#print("Shape of X_np:", X_np.shape) #test print
#print("Shape of X_poly:", X_poly.shape) #test print
print(f"The original number of features was: {X_np.shape[1]}")
print(f"The polynomially expanded number of features is: {X_poly.shape[1]}")
print(f"The number of features has increased by {X_poly.shape[1] - X_np.shape[1]} features!")

# --------------------------------------------------

#Feature expansion - code cell im working on to add coefficients as output from the function

#reset the numpy random seed using global variable rand_state
#This is reset so that when k-fold undertaken within the polynomial expansion function
#the folds are identical to those previously used earlier in the code
np.random.seed(rand_state)

#call the add_polynomial features to apply polynomial expansion 
#return values assigned to variables for subsequent use by next function to plot
#normal and poly accuracy on training and test sets

reg_values, training_acc_results, test_acc_results, training_acc_results_poly, test_acc_results_poly, poly, X_poly, df_test_results, df_poly_test_results, best_test_auc_non_poly, best_test_auc_poly, df_results_coeff, results_coeff_dict, df_results_coeff_poly, results_coeff_dict_poly = add_polynomial_features(
    X,
    X_np,
    y_np,
    number_of_splits
    )

np.random.seed(rand_state)
    

# --------------------------------------------------

df_temp_regvalue_auc = id_highest_auc_return_df(
        df_test_results, 
        "AUC", 
        "reg_values", 
        "auc_RegValuesNonPoly_n=",
        3
        )
df_temp_regvalue_auc

# --------------------------------------------------

#using the highest returned AUC, locate the regularisation value for this AUC, and associated metrics
#produce df of these
model_name = list(df_temp_regvalue_auc.columns)[0]
optimum_auc_reg_value_param = float(list(df_temp_regvalue_auc.columns)[0].split('=')[1])

df_optimum_auc_reg_value = df_test_results.loc[df_test_results['reg_values'] == optimum_auc_reg_value_param]
df_optimum_auc_reg_value_subset = df_optimum_auc_reg_value.drop(columns=["reg_values", "AUC"])
df_optimum_auc_reg_value_transpose = df_optimum_auc_reg_value_subset.transpose()
df_optimum_auc_reg_value_transpose.rename(columns={df_optimum_auc_reg_value_transpose.columns[0]: model_name},inplace=True)

#add model name (key) and parameter (value) to dictionary
dict_of_optimal_auc_params = {}
dict_of_optimal_auc_params[model_name] = optimum_auc_reg_value_param
dict_of_optimal_auc_params

# --------------------------------------------------

co_eff_df_nonpoly, co_eff_df_reduced_chance_dna_nonpoly, co_eff_df_higher_chance_dna_nonpoly = create_coefficients_df(results_coeff_dict[optimum_auc_reg_value_param], X)

# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(co_eff_df_reduced_chance_dna_nonpoly, co_eff_df_higher_chance_dna_nonpoly, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_auc_df = pd.concat([combined_auc_df, df_temp_regvalue_auc], axis=1)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_performance_metrics_df = pd.concat([combined_performance_metrics_df, df_optimum_auc_reg_value_transpose], axis=1)

# --------------------------------------------------

#extract the row with the regularisation value that yielded the highest auc along with all performance metrics
#series_test_results_max_auc_poly = df_poly_test_results.iloc[df_poly_test_results['AUC'].idxmax()] #commented out to test if works with this line removed 

#convert the extracted series back to a dataframe
#df_test_results_max_auc_poly = pd.DataFrame(series_test_results_max_auc_poly) #commented out to test if works with this line removed 

#transport this df back to original orientation
#df_test_results_max_auc_poly_transposed = df_test_results_max_auc_poly.transpose() #commented out to test if works with this line removed 


# --------------------------------------------------

df_temp_regvalue_auc_poly = id_highest_auc_return_df(
        df_poly_test_results, 
        "AUC", 
        "reg_values", 
        "auc_RegValuesPoly_n=",
        3
        )
df_temp_regvalue_auc_poly

# --------------------------------------------------

#using the highest returned AUC, locate the regularisation value for this AUC, and associated metrics
#produce df of these
model_name = list(df_temp_regvalue_auc_poly.columns)[0]

optimum_auc_reg_value_param_poly = float(list(df_temp_regvalue_auc_poly.columns)[0].split('=')[1])
df_optimum_auc_reg_value_poly = df_poly_test_results.loc[df_poly_test_results['reg_values'] == optimum_auc_reg_value_param_poly]
df_optimum_auc_reg_value_subset_poly = df_optimum_auc_reg_value_poly.drop(columns=["reg_values", "AUC"])
df_optimum_auc_reg_value_transpose_poly = df_optimum_auc_reg_value_subset_poly.transpose()
df_optimum_auc_reg_value_transpose_poly.rename(columns={df_optimum_auc_reg_value_transpose_poly.columns[0]: model_name},inplace=True)

#add model name (key) and parameter (value) to dictionary
dict_of_optimal_auc_params[model_name] = optimum_auc_reg_value_param_poly


# --------------------------------------------------

#due to the poly expansion, create a df of X_poly

poly_temp = PolynomialFeatures(2, interaction_only=True, include_bias=False)
X_poly_test = poly_temp.fit_transform(X)
X_poly_df_temp = pd.DataFrame(X_poly, columns=poly_temp.get_feature_names_out())

#available_features_test = list(poly_temp.get_feature_names_out())

#print(len(available_features_test))

#print(X_poly_test_df.columns)

#features_test = pd.DataFrame(columns=X_poly_test.get_feature_names(X.columns))
#print(features_test)



# --------------------------------------------------

co_eff_df_poly, co_eff_df_reduced_chance_dna_poly, co_eff_df_higher_chance_dna_poly = create_coefficients_df(results_coeff_dict_poly[optimum_auc_reg_value_param_poly], X_poly_df_temp)

# --------------------------------------------------

#identify the top-x features associated with higher and lower chance of dna, respectively
#add these as a df to each respective dictionary as model_name (key) to df (value) pairs
#later, once optimum model ID'd, use model_name to identify the relevant coefficients from these growing dictionaries
df_top_x_reduce_chance_dna_poly, df_top_x_increased_chance_dna_poly = id_top_x_coefficients(co_eff_df_reduced_chance_dna_poly, co_eff_df_higher_chance_dna_poly, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna_poly, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna_poly, model_name)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_auc_df = pd.concat([combined_auc_df, df_temp_regvalue_auc_poly], axis=1)

# --------------------------------------------------

#Update combined dataframes for auc and summary of performance metrics, respectively
combined_performance_metrics_df = pd.concat([combined_performance_metrics_df, df_optimum_auc_reg_value_transpose_poly], axis=1)

# --------------------------------------------------
#Model variant 8: Optuna optimised model - Logistic Regression
# --------------------------------------------------
"""
Notes from Mike Allen's optuna notebook:
A model hyperparamter is a setting in the model that is set prior to 
learning from a data set. It may affect how well the model learns from data. 
Examples include the level of regularisation in a logistic regression model, 
or the number of trees in a random forest model

Many machine learning models have many hyperparameters that may be changed, 
and each may have many different possible levels to choose from. How do you 
find the best combination (though most machine learning packages come with 
sensible defaults which may already give optimal or close to optimal 
results). 

Optuna (optuna.org) is a machine learning model optimizer that may be 
used on any machine learning model type. Optuna uses heuristic (searching) 
algorithms to find the best combination of model hyperparameters. 
It also allows more traditional alternatives to heuristic algorithms, 
such as grid search and random search.

When optimising model hyperparameters, we use only the training set of 
data, and test independently with a test set not used to optimise those 
hyperparameters. This avoids reporting a spuriously high accuracy level 
due to over-fitting of model hyperparameters to the training set.

During optimisation we create new k-fold train/test splits from the 
training set. The model hyperparameters are then trained on the average 
accuracy of these new k-fold test splits. 

"""

# --------------------------------------------------

#This code imports warnings, and *many* print lines for status update, consolidating this into a simple progress bar
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --------------------------------------------------

study = optuna.create_study(direction = 'maximize')
study.optimize(get_results, n_trials=100, show_progress_bar=True)

# --------------------------------------------------

best_params = study.best_params

# --------------------------------------------------

#fit model with the identified optimum parameter
model = LogisticRegression(C=best_params['C'])

#best params has many decimal places. round to 3 for us in the df col name for this model
best_params_rounded = round(best_params['C'],3)
model_name = f"auc_Optuna_C={best_params_rounded}"

#model = LogisticRegression(C=1.0) #test line using default C param value 1.0 to compare to optimised model above.
model.fit(X_train, y_train)

#Predict training and test set labels, using standardised data
y_pred_train = model.predict(X_train_std)
y_pred_test = model.predict(X_test_std)

#loop to create and update nested dict - this dictionary consists of the performance metrics for the test and training sets respectively for this model
list_outer_dict_levels = ["train", "test"]
dict_of_splits = {"train": [y_train, y_pred_train], "test": [y_test, y_pred_test]}
temp_dict = {}
for level in list_outer_dict_levels:
    temp_dict[level] = calculate_accuracy(dict_of_splits[level][0], dict_of_splits[level][1])

# Get test set proabilities
y_pred_proba = model.predict_proba(X_test_std)[:,1]

#Get ROC AUC for this split, append to list of AUC's
optuna_log_reg_auc = roc_auc_score(y_test, y_pred_proba)

dict_mean_test_performance = temp_dict["test"] #this is used 
dict_mean_train_performance = temp_dict["train"]

print(optuna_log_reg_auc)


# --------------------------------------------------

#add coefficients from this model to the x2 dicts of coefficients that predict higher or lower (respectively) chance of DNA
co_eff_df, df_reduced_chance, df_increased_chance = id_coefficients(model)
df_top_x_reduce_chance_dna, df_top_x_increased_chance_dna = id_top_x_coefficients(df_reduced_chance, df_increased_chance, top_x, model_name)
dict_coeff_higher_chance_dna_each_model = add_coeff_to_dict(dict_coeff_higher_chance_dna_each_model, df_top_x_increased_chance_dna, model_name)
dict_coeff_lower_chance_dna_each_model = add_coeff_to_dict(dict_coeff_lower_chance_dna_each_model, df_top_x_reduce_chance_dna, model_name)

# --------------------------------------------------

#create a df consisting of the roc auc for the otimal model which optuna identified. Give this the col name of the model with optimal param value
df_temp_optimised_auc = pd.DataFrame(columns = [model_name])
df_temp_optimised_auc.loc[0] = optuna_log_reg_auc

# --------------------------------------------------

#append the roc auc for the optimised model to the summary of all models' roc auc values
combined_auc_df = pd.concat([combined_auc_df, df_temp_optimised_auc], axis=1)

# --------------------------------------------------

#Call function to add the summary performance metrics of this model run to the summary
df_summary_of_models = add_performance_metrics_to_summary(
    combined_performance_metrics_df,
    dict_metrics_summary,
    dict_mean_test_performance,
    model_name
    )
df_summary_of_models

# --------------------------------------------------
#STAGE TWO: ID model with max ROC AUC and print the features
#most predictive of high (and lower) DNA chance
# --------------------------------------------------

optimum_auc_model_name_raw = combined_auc_df.idxmax(axis=1)
optimum_auc_model_temp_name = str(optimum_auc_model_name_raw).split("_")[1:]

#optimum_auc_model_name_raw
optimum_auc_model_temp_name
#combined_auc_df

# --------------------------------------------------

"""UPTO HERE - NEED TO WORK OUT HOW TO BRING BACK THE VALUES AND COLUMN NAMES
FOR HIGHEST AUC (DONE)
THEN WITH RETURNED COLUMN NAME, FIGURE OUT HOW TO USE THE PARAM VALUE THAT OPTIMISED
THAT PARTICULAR MODEL IN A RERUN OF THAT MODEL TO THEN BE TWEAKED FURTHER 
WITH POLYNOMIAL ETC. (FUTURE DEVELOPMENT TO DO)"""

#Code to ID model name with max auc value.
#Models with an optimal parameter value include this parameter value in the name
#after an equals symbol. This code looks for the presence of the equals symbol
#and returns the relevant parameter value from the dictionary of optimal params
optimum_auc_model_name_raw = combined_auc_df.idxmax(axis=1)

#index_num_optimum_auc_model_name = int(str(optimum_auc_model_name_raw)[0])

optimum_auc_model_temp_name = str(optimum_auc_model_name_raw).split("_")[1:]
optimum_auc_model_shortened_name = str(optimum_auc_model_temp_name[0] + "_" + optimum_auc_model_temp_name[1])
print(f"The model producing the optimal Area Under the Curve is: {optimum_auc_model_shortened_name}.")

optimum_model_name_for_dict_key = str(optimum_auc_model_name_raw).split()[1]

# --------------------------------------------------

#Code to identify the second best performing model, by identifying the second highest auc value
second_highest_auc_model_value = combined_auc_df.apply(lambda row: row.nlargest(2).values[-1], axis=1)
second_highest_auc_model_name_raw = combined_auc_df.T.apply(lambda row: row.nlargest(2).idxmin())
print(second_highest_auc_model_name_raw)
print(second_highest_auc_model_value)

# --------------------------------------------------

#return the values (df) of coefficients associated with the optimum model name (optimum_model_name_for_dict_key) from above code
#save this to file / export into report - still to do
print(f"The top {top_x} features associated with a higher chance of DNA are:")
print(dict_coeff_higher_chance_dna_each_model[optimum_model_name_for_dict_key])

#save top X features associated with higher dna chance to csv to later call into code to produce report
dict_coeff_higher_chance_dna_each_model[optimum_model_name_for_dict_key].to_csv(f"{dna_assets_path}/df001_Top{max_features}HigherChanceDna.csv")

# --------------------------------------------------

#repeat above step, to show features associated with a lower chance of DNA
print(f"The top {top_x} features associated with a lower chance of DNA are:")
print(dict_coeff_lower_chance_dna_each_model[optimum_model_name_for_dict_key])

#save top X features associated with lower dna chance to csv to later call into code to produce report
dict_coeff_lower_chance_dna_each_model[optimum_model_name_for_dict_key].to_csv(f"{dna_assets_path}/df002_Top{max_features}LowerChanceDna.csv")

#save combined_auc_df to file 
combined_auc_df.round(decimals=2).to_csv(f"{dna_assets_path}/df003_AllModelsROCAUC.csv")
combined_auc_df.to_csv(f"{dna_assets_path}/df003_AllModelsROCAUCNotRounded.csv")

#save df_summary_of_models to file
df_summary_of_models.round(decimals=2).to_csv(f"{dna_assets_path}/df003_AllModelsMetrics.csv")


# --------------------------------------------------
#END OF FILE
# --------------------------------------------------

"""
Potential areas for further development:
1. UAT and remove all areas identified in the warnings
2. Add further classification algorithms / methods (Random Forests, Neural Net..) to check if they yield a better prediction that the current log regression variants.
"""