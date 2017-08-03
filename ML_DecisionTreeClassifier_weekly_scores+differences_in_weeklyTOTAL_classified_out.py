import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from tabulate import tabulate

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection, tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

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
# create column of final exam score, with flag P if >= 20 and F if below
dataset['Exam_PassFail'] = dataset.apply(lambda row: convert_exam_to_pass_fail(row), axis=1)

#print(dataset.head(20))
#print(list(dataset))

# explicitly list the inputs here, these are all variables that will be fed in
input_list = ['bb_hours','Wk2','Wk3','Wk4','Wk5','Wk6','improvement_Wk2toWk3', 'improvement_Wk2toWk4', 'improvement_Wk2toWk5','improvement_Wk2toWk6','improvement_Wk3toWk4','improvement_Wk3toWk5','improvement_Wk3toWk6','improvement_Wk4toWk5','improvement_Wk4toWk6','improvement_Wk5toWk6']
input_indices = []
for name in input_list:
	input_indices.append(dataset.columns.get_loc(name))
print input_indices

# this is the variable we want to try to predict
#Y_variable = 'Wk_TOTAL_flag'
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
algo = DecisionTreeClassifier()
algo.fit(X_train, Y_train)
predictions = algo.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))
#print(rfc.decision_path(X_validation))
#print(rfc.get_params())
#print(rfc)

print tabulate((list(zip(input_list, algo.feature_importances_))))

# Export the tree, can run this at http://webgraphviz.com/ (see http://dataaspirant.com/2017/04/21/visualize-decision-tree-python-graphviz/ for further info on saving as PDF instead)
with open("decision_tree.txt", "w") as f:
    f = tree.export_graphviz(algo, out_file=f)
