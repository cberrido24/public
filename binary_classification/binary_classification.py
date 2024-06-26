#-----------------------------------------------------------------------------#
# Binary classification - based on kaggle competition
# https://www.kaggle.com/competitions/playground-series-s3e17/overview
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




