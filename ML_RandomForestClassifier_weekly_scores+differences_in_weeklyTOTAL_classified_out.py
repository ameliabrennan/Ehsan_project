import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from tabulate import tabulate

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from myFunctions import classify_WkTOTAL, final_score, convert_to_GPA, variation

# Load dataset
url = "/Users/e35596/RMITStudios/Ehsan_project/csv_files/Deidentified_data_fixedBBHITS_fixedTotalV.csv"

# Set the columns that I want, read in the dataset
cols = ['Exam_TOTAL','Wk2','Wk3','Wk4','Wk5','Wk6','Wk_TOTAL','bb_hours']
dataset = pandas.read_csv(url, sep=',', na_values=["", 'A', '-'], usecols=cols)

# set the NaNs to 0s everywhere
for col in cols:
    dataset[col].fillna(0, inplace=True)

# Add columns that flag whether the student mark has decreased or not, looking at all possible combinations of (later week) - (ealier week)
for i in range(2,7):
	for j in range(i+1, 7):
		#print 'i, j: ' + str(i) + ', ' + str(j)
		dataset['improvement_Wk{0}toWk{1}'.format(i,j)] = dataset.apply(lambda row: variation(row, 'Wk{0}'.format(i), 'Wk{0}'.format(j)), axis=1)

# create column of final quiz score, with flag 1 if > 32 and 0 if below
dataset['Wk_TOTAL_flag'] = dataset.apply(lambda row: classify_WkTOTAL(row), axis=1)

#print(dataset.head(20))
#print(list(dataset))

# explicitly list the inputs here, these are all variables that will be fed in
input_list = ['bb_hours','Wk2','Wk3','Wk4','Wk5','Wk6','improvement_Wk2toWk3', 'improvement_Wk2toWk4', 'improvement_Wk2toWk5','improvement_Wk2toWk6','improvement_Wk3toWk4','improvement_Wk3toWk5','improvement_Wk3toWk6','improvement_Wk4toWk5','improvement_Wk4toWk6','improvement_Wk5toWk6']
input_indices = []
for name in input_list:
	input_indices.append(dataset.columns.get_loc(name))
print input_indices

# this is the variable we want to try to predict
Y_variable = 'Wk_TOTAL_flag'
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
algo = RandomForestClassifier()
algo.fit(X_train, Y_train)
predictions = algo.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#print(rfc.decision_path(X_validation))
#print(rfc.get_params())
#print(rfc)

# View a list of the features and their importance scores
print tabulate((list(zip(input_list, algo.feature_importances_))))

'''
Results from one run of the above (changes each time):

[('Wk2', 0.15024731339726918),
('Wk3', 0.1325848072780588),
('Wk4', 0.094414696725759362),
('Wk5', 0.12499159492028115),
('Wk6', 0.28028964268382739),
('improvement_Wk2toWk3', 0.0055124765464279512),
('improvement_Wk2toWk4', 0.012306189685765734),
('improvement_Wk2toWk5', 0.030250391144419748),
('improvement_Wk2toWk6', 0.0125006586902815),
('improvement_Wk3toWk4', 0.041533268850849944),
('improvement_Wk3toWk5', 0.013768554731255083),
('improvement_Wk3toWk6', 0.0035775044431426463),
('improvement_Wk4toWk5', 0.056576857647153887),
('improvement_Wk4toWk6', 0.014854235754451267),
('improvement_Wk5toWk6', 0.026591807501056462)]


dtc = DecisionTreeClassifier()
dtc.fit(X_train, Y_train)
predictions_dtc = dtc.predict(X_validation)
print(accuracy_score(Y_validation, predictions_dtc))
print(confusion_matrix(Y_validation, predictions_dtc))
print(classification_report(Y_validation, predictions_dtc))
print(dtc.decision_path(X_validation))
print(dtc)

with open("decision_tree.txt", "w") as f:
    f = tree.export_graphviz(dtc, out_file=f)
'''
