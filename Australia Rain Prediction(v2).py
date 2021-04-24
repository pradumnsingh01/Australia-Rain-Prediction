#################### Importing necessary libraries ########################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#################### Importing the dataset #################################

dataset = pd.read_csv("weatherAUS.csv")

############################# EDA ##################################

dataset.isnull().sum() / len(dataset) * 100

# Basis the above code, we will be droping the following columns:
    
dataset = dataset.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis = 1)
dataset = dataset.drop(["Date"], axis = 1)

# Dividing the data basis continuous and categorical:

dataset.dtypes
    
dataset_int = dataset.select_dtypes(include = np.number)
dataset_cat = dataset.select_dtypes(exclude = np.number)

# Filling up the null values:
    
int_cols = dataset_int.columns
cat_cols = dataset_cat.columns

# Checking the distribution of continuous features:
    
dataset_int.hist(bins = 100)
plt.show()

# Checking for the outliers in the contiuous features:
    

dataset_int.boxplot(column = ["MinTemp", "MaxTemp","Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am", "Temp3pm"])
plt.show()




# SInce it is a classification problem and not a regression problem
    # and since we will be using tree based algos and ensemble techniques
    # as well, the outliers will not create an issue, hence we will not 
    # remove the outliers for the categorical features.
    
    
# Now checking the value counts of categorical variables:
    
dataset_cat["Location"].value_counts()
dataset_cat["Location"].nunique()

dataset_cat["WindGustDir"].value_counts()
dataset_cat["WindGustDir"].nunique()

dataset_cat["WindDir9am"].value_counts()
dataset_cat["WindDir9am"].nunique()

dataset_cat["WindDir3pm"].value_counts()
dataset_cat["WindDir3pm"].nunique()

dataset_cat["RainToday"].value_counts() # unbalanced data (because there is low rainfall in Australia)
dataset_cat["RainToday"].nunique()

dataset_cat["RainTomorrow"].value_counts() #unbalanced
dataset_cat["RainTomorrow"].nunique()


###############################################################################

# Dividing into Independent and target:
    
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

###############################################################################

# Spliting into train and test sets:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)

###############################################################################

# Splitting the X_train into categorical and continuous features:
    
X_train_int = X_train.select_dtypes(include = np.number)
X_train_cat = X_train.select_dtypes(exclude = np.number)

from sklearn.impute import SimpleImputer
sim_num = SimpleImputer(strategy = "median")
X_train_int = sim_num.fit_transform(X_train_int)
X_train_int = pd.DataFrame(X_train_int, columns = int_cols)

X_train_int.isnull().sum()

X_train_cat.isnull().sum()

from sklearn.impute import SimpleImputer
sim_cat = SimpleImputer(strategy = "most_frequent")
X_train_cat = sim_cat.fit_transform(X_train_cat)
X_train_cat = pd.DataFrame(X_train_cat, columns = cat_cols[:-1])

X_train_cat.isnull().sum()

# Now we have treated all the null values of train set.

# Now we will scale the X_train_int:
    
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_int = sc.fit_transform(X_train_int)
X_train_int = pd.DataFrame(X_train_int, columns = int_cols)

# One hot encoding of X_train_cat:
    
X_train_cat = pd.get_dummies(X_train_cat)


# Creating final train datasset:
    
X_train_final = pd.concat([X_train_int, X_train_cat], axis = 1)

X_train_f = X_train_final.values



# Now considering y_train:
    
y_train.isnull().sum()

    # There are 2422 null values present in the y_train.
    
from sklearn.impute import SimpleImputer
sim_mode = SimpleImputer(strategy = "most_frequent")
y_train = sim_mode.fit_transform(y_train.values.reshape(-1,1))

# Now label encoding the y_train:
    
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

np.unique(y_train)

# Building Models:
    
# We will be using KFold and hcekc whether overfitting is being prevented or not.
    
    
from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier()

from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)    

# Decision Tree:

scores_dt = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    dtf.fit(X_train_idx, y_train_idx)
    scores_dt.append(dtf.score(X_test_idx, y_test_idx))
    
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


log_reg = LogisticRegression()
nb = GaussianNB()
rf = RandomForestClassifier()
ada = AdaBoostClassifier()
gb = GradientBoostingClassifier()


# Logistic Regression:

scores_lr = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    log_reg.fit(X_train_idx, y_train_idx)
    scores_lr.append(log_reg.score(X_test_idx, y_test_idx))
    

# GaussianNB:
    
scores_nb = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    nb.fit(X_train_idx, y_train_idx)
    scores_nb.append(nb.score(X_test_idx, y_test_idx))
    
scores_rf = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    rf.fit(X_train_idx, y_train_idx)
    scores_rf.append(rf.score(X_test_idx, y_test_idx))
    
scores_ada = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    ada.fit(X_train_idx, y_train_idx)
    scores_ada.append(ada.score(X_test_idx, y_test_idx))
    
scores_gb = []

for train_idx, test_idx in kf.split(X_train_f):
    X_train_idx, X_test_idx = X_train_f[train_idx], X_train_f[test_idx]
    y_train_idx, y_test_idx = y_train[train_idx], y_train[test_idx]
    gb.fit(X_train_idx, y_train_idx)
    scores_gb.append(gb.score(X_test_idx, y_test_idx))
    

# Since Logistic Regression and Random Forest are showing very close
    # scores, we will be comparing the through roc-auc curve, which
    # will help us in determining which model have better class
    # distinguishing ability.
    
from sklearn.metrics import roc_auc_score, roc_curve

y_score_lr = log_reg.decision_function(X_train_f)
y_score_lr

y_score_rf = rf.predict_proba(X_train_f)
y_score_rf = y_score_rf[:,1]
y_score_rf


print(roc_auc_score(y_train, y_score_lr))
print(roc_auc_score(y_train, y_score_rf))


fpr_lr, tpr_lr, th_lr = roc_curve(y_train, y_score_lr)
fpr_rf, tpr_rf, th_rf = roc_curve(y_train, y_score_rf)

plt.plot(fpr_lr, tpr_lr, label = "Logistic Regression")
plt.plot(fpr_rf, tpr_rf, label = "Random Forest")
plt.plot([0,1], [0,1])
plt.xlabel("False Posotive Rate (1-Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("ROC Curve for Rain Prediction")
plt.legend()
plt.show()


# It shows that Random Forest has better class distinguishing 
    # ability than Logistic Regression
    

# Now making necessary changes in the test data:
    

# Splitting the X_test into continuous and categorical columns:
    
X_test_int = X_test.select_dtypes(include = np.number)
X_test_cat = X_test.select_dtypes(exclude = np.number)


# Filling the null values through median in X_test_int:
    
X_test_int = sim_num.transform(X_test_int)

# Now scaling the X_test_int:
    
X_test_int = sc.transform(X_test_int)

X_test_int = pd.DataFrame(X_test_int, columns = int_cols)


# Now considering X_test_cat:
    
# imputing missing values with mode in X_test_cat:
    
X_test_cat = pd.DataFrame(sim_cat.transform(X_test_cat), columns = cat_cols[:-1])

X_test_cat = pd.get_dummies(X_test_cat)

X_test_final = pd.concat([X_test_int, X_test_cat], axis = 1)

y_test = sim_mode.transform(y_test.values.reshape(-1,1))

y_test = le.transform(y_test)



# Unsing Preformcance metrics to check the model:
    
y_pred_lr = log_reg.predict(X_test_final)
y_pred_rf = rf.predict(X_test_final)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix

for i in [accuracy_score, precision_score, recall_score]:
    print(i)
    print(i(y_test, y_pred_lr))
    print(i(y_test, y_pred_rf))
    
y_score_lr_test = log_reg.decision_function(X_test_final)
y_score_lr_test

y_score_rf_test = rf.predict_proba(X_test_final)
y_score_rf_test = y_score_rf_test[:,1]

print(roc_auc_score(y_test, y_score_lr_test))
print(roc_auc_score(y_test, y_score_rf_test))

fpr_lr_test, tpr_lr_test, th_lr_test = roc_curve(y_test, y_score_lr_test)
fpr_rf_test, tpr_rf_test, th_rf_test = roc_curve(y_test, y_score_rf_test)

plt.plot(fpr_lr, tpr_lr, label = "Logistic Regression on Train Set")
plt.plot(fpr_lr_test, tpr_lr_test, label = "Logistic Regression on Test Set")
plt.plot(fpr_rf, tpr_rf, label = "Random Forest on Train Set")
plt.plot(fpr_rf_test, tpr_rf_test, label = "Random Forest on Test Set")
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True POsitive Rate (Sensitivity)")
plt.title("ROC Curve for Rain Prediction")
plt.legend()
plt.show()