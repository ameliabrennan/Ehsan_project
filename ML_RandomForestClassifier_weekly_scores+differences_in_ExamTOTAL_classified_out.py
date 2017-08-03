import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from myFunctions import classify_WkTOTAL, final_score, convert_to_GPA, variation, convert_exam_to_pass_fail

# Load dataset
url = "/Users/e35596/RMITStudios/Ehsan_project/csv_files/Deidentified_data_fixedBBHITS_fixedTotalV.csv"

# Set the columns that I want, read in the dataset
cols = ['Exam_TOTAL','Wk2','Wk3','Wk4','Wk5','Wk6','Wk_TOTAL', 'bb_hours']
dataset = pandas.read_csv(url, sep=',', na_values=["", 'A', '-'], usecols=cols)

# set the NaNs to 0s everywhere
for col in cols:
    dataset[col].fillna(0, inplace=True)

# Add columns that flag whether the student mark has decreased or not, looking at all possible combinations of (later week) - (ealier week)
for i in range(2,7):
	for j in range(i+1, 7):
		#print 'i, j: ' + str(i) + ', ' + str(j)
		dataset['improvement_Wk{0}toWk{1}'.format(i,j)] = dataset.apply(lambda row: variation(row, 'Wk{0}'.format(i), 'Wk{0}'.format(j)), axis=1)

# create column of final exam score, with flag P if >= 20 and F if below
dataset['Exam_PassFail'] = dataset.apply(lambda row: convert_exam_to_pass_fail(row), axis=1)

#print(dataset.head(20))
#print(list(dataset))

# explicitly list the inputs here, these are all variables that will be fed in
input_list = ['bb_hours','Wk2','Wk3','Wk4','Wk5','Wk6','improvement_Wk2toWk3', 'improvement_Wk2toWk4', 'improvement_Wk2toWk5','improvement_Wk2toWk6','improvement_Wk3toWk4','improvement_Wk3toWk5','improvement_Wk3toWk6','improvement_Wk4toWk5','improvement_Wk4toWk6','improvement_Wk5toWk6']
#input_list = ['improvement_Wk2toWk3', 'improvement_Wk2toWk4', 'improvement_Wk2toWk5','improvement_Wk2toWk6','improvement_Wk3toWk4','improvement_Wk3toWk5','improvement_Wk3toWk6','improvement_Wk4toWk5','improvement_Wk4toWk6','improvement_Wk5toWk6']
#input_list.append('Exam_PassFail')
input_indices = []
for name in input_list:
	input_indices.append(dataset.columns.get_loc(name))
print input_indices
#input_indices = ['Wk6']

# this is the variable we want to try to predict
Y_variable = 'Exam_PassFail'
Y_index = dataset.columns.get_loc(Y_variable)
print Y_index

# Split-out validation dataset
array = dataset.values
X = array[:,input_indices]
Y = array[:,Y_index]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Make predictions on validation dataset
algo = RandomForestClassifier(n_estimators=500)
algo.fit(X_train, Y_train)
predictions = algo.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
print(algo.decision_path(X_validation))
print(algo.get_params())
print(algo)

# View a list of the features and their importance scores
print tabulate(list(zip(input_list, algo.feature_importances_)))
print sum(algo.feature_importances_)

'''
Results from one run of the above (changes each time):

[('Wk2', 0.16424656392250453),
('Wk3', 0.082613650537509326),
('Wk4', 0.14464458623985116),
('Wk5', 0.13995952900713424),
('Wk6', 0.23274843588080601),
('improvement_Wk2toWk3', 0.0097275771291813803),
('improvement_Wk2toWk4', 0.035330151811149499),
('improvement_Wk2toWk5', 0.033363198791580069),
('improvement_Wk2toWk6', 0.026172971126866014),
('improvement_Wk3toWk4', 0.023339517017819368),
('improvement_Wk3toWk5', 0.026803171039352557),
('improvement_Wk3toWk6', 0.013727872708418859),
('improvement_Wk4toWk5', 0.022628313446189922),
('improvement_Wk4toWk6', 0.012858217976203984),
('improvement_Wk5toWk6', 0.031836243365433149)]

'''
