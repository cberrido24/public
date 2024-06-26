#-----------------------------------------------------------------------------#
# Linear regression - based on kaggle competition
# https://www.kaggle.com/competitions/playground-series-s3e16/overview
#-----------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

#-----------------------------------------------------------------------------#
# Importing data
#-----------------------------------------------------------------------------#
path = "D:/projects/linear_reg_crabs/"
inpath = str(path+"inputs/")
outpath = str(path+"outputs/")

train = pd.read_csv(str(inpath+"train.csv"))
test = pd.read_csv(str(inpath+"test.csv"))

#-----------------------------------------------------------------------------#
# Feature selection and preparing data
#-----------------------------------------------------------------------------#
train.columns
test.columns
dict = {"Shucked Weight" : "shucked_weight",
        "Viscera Weight" : "viscera_weight",
        "Shell Weight" : "shell_weight"
        }

train.rename(dict, axis=1, inplace=True)
test.rename(dict, axis=1, inplace=True)

features = [#"Sex",
            "Length",
            "Diameter",
            "Height",
            "Weight",
            "shucked_weight",
            "viscera_weight",
            "shell_weight"
            ]

variable_to_predict = np.array(train["Age"])
variables_used = np.array(train[features])
variables_used_test_set = np.array(test[features])

def rename_sex(dataset):
    dataset["sex_I"] = np.where(dataset["Sex"]=="I", 1, 0)
    dataset["sex_M"] = np.where(dataset["Sex"]=="M", 1, 0)
    dataset["sex_F"] = np.where(dataset["Sex"]=="F", 1, 0)

rename_sex(train)
rename_sex(test)

# Normalise values
from sklearn import preprocessing
variables_used = preprocessing.StandardScaler().fit(variables_used).transform(variables_used)

# Split train and test datasets
from sklearn.model_selection import train_test_split
train_features, test_features, train_age, test_age = train_test_split(variables_used, variable_to_predict, test_size=0.2, random_state=123)

print ('Train set:', train_features.shape,  train_age.shape)
print ('Test set:', test_features.shape,  test_age.shape)

#-----------------------------------------------------------------------------#
# Modelling
#-----------------------------------------------------------------------------#
from sklearn.linear_model import LinearRegression
LR = LinearRegression().fit(train_features, train_age)

test_modelled_prediction = np.round(LR.predict(test_features), 0)
print(test_modelled_prediction)

#-----------------------------------------------------------------------------#
# # Evaluating model and plots
#-----------------------------------------------------------------------------#
# Evaluating model
MSE = np.mean((test_modelled_prediction - test_age) ** 2)
print("Mean Squared error(MSE): %.2f"
      % MSE)

# Plots
plt.scatter(train['viscera_weight'], train['Age'], color='blue')
plt.xlabel("Viscera weight")
plt.ylabel("Age")
plt.show()

#-----------------------------------------------------------------------------#
# Predicting
#-----------------------------------------------------------------------------#
# Normalise values
variables_used_test_set = preprocessing.StandardScaler().fit(variables_used_test_set).transform(variables_used_test_set)
modelled_predictions_test_set = np.round(LR.predict(variables_used_test_set), 0)

#-----------------------------------------------------------------------------#
# Exporting results
#-----------------------------------------------------------------------------#
predictions_to_submit = pd.concat([test["id"], pd.DataFrame(modelled_predictions_test_set)], axis=1)
predictions_to_submit = predictions_to_submit.rename({0:'Age'}, axis=1)
predictions_to_submit.to_csv(str(outpath+"submission.csv"), index=False)


