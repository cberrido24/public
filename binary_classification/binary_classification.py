#-----------------------------------------------------------------------------#
# Binary classification - based on kaggle competition
# https://www.kaggle.com/competitions/playground-series-s3e17/overview
#
#
# The goal of this competition is to predict the probability of failure for
# each machine (one row representing one machine) given technical variables
# such as temperature, speed, etc...
# First, we will try to explore the data and visualise the variables
# Then, we will try to create a model that we can use to predict the
# probability of failure for each machine on the test dataset
#-----------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


path = "D:/projects/binary_classification/"
inpath = str(path + "inputs/")
outpath = str(path + "outputs/")

train = pd.read_csv(str(inpath+"train.csv"))
test = pd.read_csv(str(inpath+"test.csv"))

#-----------------------------------------------------------------------------#
# Formatting data
#-----------------------------------------------------------------------------#

columns_to_rename = test.columns
columns_train = train.columns

dict = {'Product ID':'product_id',
        'Air temperature [K]':'air_temperature_k',
        'Process temperature [K]':'process_temperature_k',
        'Rotational speed [rpm]':'rotational_speed_rpm',
        'Torque [Nm]':'torque_nm',
        'Tool wear [min]':'tool_wear_min'
        }


test.rename(dict, axis=1, inplace=True)
train.rename(dict, axis=1, inplace=True)
train.rename({'Machine failure':'machine_failure'}, axis=1, inplace=True)

#-----------------------------------------------------------------------------#
# Exploration of data and visualizations
#-----------------------------------------------------------------------------#
# 1. Distribution of product types
train['Type'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Product Types')
plt.xlabel('Product Type')
plt.ylabel('Count')

# Here, we notice that we mainly have products with type "L", then "M" and few "H".

# 2. Air temperature vs. process temperature
plt.scatter(train['air_temperature_k'], train['process_temperature_k'], alpha=0.5, c='blue')
plt.title('Air Temperature vs. Process Temperature')
plt.xlabel('Air Temperature [K]')
plt.ylabel('Process Temperature [K]')

# We observe a positive relationship between air temperature and process temperature

# 3. Rotational speed distribution
train['rotational_speed_rpm'].plot(kind='hist', bins=30, color='green')
plt.title('Distribution of Rotational Speed')
plt.xlabel('Rotational Speed [rpm]')
plt.ylabel('Frequency')

# This histogram represents the distribution of rotational speed in revolutions per minute (rpm).
# It shows that the data is fairly spread out over a range of values, and that
# the most common rotational speeds appear to cluster around 1500 to 1700 rpm
# This indicates that the machines typically operate within this mid-range speed, though there are instances of both lower and higher speeds.

# 4. Machine failure count
train['machine_failure'].value_counts().plot(kind='bar', color='orange')
plt.title('Machine Failure Count')
plt.xlabel('Machine Failure')
plt.ylabel('Count')

# This bar chart shows that machine failures are relatively rare in the dataset.
# The vast majority of the records indicate no machine failure (value 0), with only a small fraction showing instances of machine failure (value 1).
# This suggests that machines don't frequently fail.


#-----------------------------------------------------------------------------#
# Logistic regression
#-----------------------------------------------------------------------------#

# Separate training set
variables_used = np.array(train[[#'id',
                          #'product_id',
                          'air_temperature_k',
                          'process_temperature_k',
                          'rotational_speed_rpm',
                          'torque_nm',
                          'tool_wear_min',
                          'TWF',
                          'HDF',
                          'PWF',
                          'OSF',
                          'RNF'
                          ]])
variable_to_predict = np.array(train['machine_failure'])

# Normalise values
from sklearn import preprocessing
variables_used = preprocessing.StandardScaler().fit(variables_used).transform(variables_used)

# Split dataset
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(variables_used, variable_to_predict, test_size=0.2, random_state=2)
print ('Train set:', train_x.shape,  train_y.shape)
print ('Test set:', test_x.shape,  test_y.shape)

# Modeling - Logistic Regression
    # **C** parameter indicates **inverse of regularization strength** which must be a positive float. Smaller values specify stronger regularization.
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(train_x, train_y)
LR

yhat = LR.predict(test_x)
yhat

yhat_prob = LR.predict_proba(test_x)
yhat_prob


#-----------------------------------------------------------------------------#
# Evaluating results
#-----------------------------------------------------------------------------#
# Jaccard index
    # Jaccard  index represents
    # the size of the intersection divided by the size of the union of the two label sets.
    # If the entire set of predicted labels for a sample strictly matches with the true set of labels,
    # then the subset accuracy is 1.0; otherwise it is 0.0.
from sklearn.metrics import jaccard_score
jaccard_score(test_y, yhat, pos_label=0)


# Confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print(confusion_matrix(test_y, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['failure=1','failure=0'],normalize= False,  title='Confusion matrix')

#-----------------------------------------------------------------------------#
# Apply LR to test set
#-----------------------------------------------------------------------------#
test_set = np.array(test[['air_temperature_k',
                          'process_temperature_k',
                          'rotational_speed_rpm',
                          'torque_nm',
                          'tool_wear_min',
                          'TWF',
                          'HDF',
                          'PWF',
                          'OSF',
                          'RNF'
                          ]])

test_set = preprocessing.StandardScaler().fit(test_set).transform(test_set)

yhat_final = LR.predict(test_set)
yhat_final

yhat_prob_final = LR.predict_proba(test_set)
yhat_prob_final = pd.DataFrame(yhat_prob_final)

output = pd.concat([test, yhat_prob_final], axis=1)
output = output[['id', 1]].rename({1:'Machine failure'}, axis=1)


#-----------------------------------------------------------------------------#
# Exporting outputs
#-----------------------------------------------------------------------------#
output.to_csv(str(outpath+'submission_logistic_regression.csv'), index=False)




