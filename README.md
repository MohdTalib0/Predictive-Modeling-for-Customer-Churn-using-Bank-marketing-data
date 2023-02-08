Customer Churn Prediction
Building a predictive model for customer churn
A Bank wants to take care of customer retention for its product: savings accounts. The bank wants you to identify customers likely to churn balances below the minimum balance. You have the customers information such as age, gender, demographics along with their transactions with the bank.
What is the problem?
The main problem is to predict if a customer would be credit defaulter or not depending upon the previous data of the customer.
Why is it important?
It is important from a bank’s perspective in order to maintain business and customer relationship/ Apart from that if someone could be predicted as a defaulter then primitive measures can be taken in order to ensure that such violations do not happen.
The ML Pipeline that I have followed is :
1.	Importing the necessary libraries and the dataset
2.	Performing Data Pre-processing (Exploratory Data Analysis and Data Manipulation)
3.	Modelling using Logistic Regression, XGBoost and Random Forest
4.	Performing Prediction
5.	Visualization in between Actual and predicted Values
The environment used was python 3.7 and the libraries such as NumPy, Pandas, Matplotlib ,Seaborn, Plotly, Standard Scaler and Scikit Learn module were used for Scientific computations.

Importing Libraries
In [1]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
In [2]:
df = pd.read_csv("C:/Users/hp/Downloads/bank.csv")
In [3]:
df.head()
Out[3]:
	age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"
0	30;"unemployed";"married";"primary";"no";1787;...
1	33;"services";"married";"secondary";"no";4789;...
2	35;"management";"single";"tertiary";"no";1350;...
3	30;"management";"married";"tertiary";"no";1476...
4	59;"blue-collar";"married";"secondary";"no";0;...
Introducing Seperators
In [4]:
df = pd.read_csv("C:/Users/hp/Downloads/bank.csv", sep = ";") 
# in this case ; is our seperater
In [5]:
#now checking the head of our data
df.head()
Out[5]:
Exploratory Data Analysis
In [9]:
df.dtypes
Out[9]:
age           int64
job          object
marital      object
education    object
default      object
balance       int64
housing      object
loan         object
contact      object
day           int64
month        object
duration      int64
campaign      int64
pdays         int64
previous      int64
poutcome     object
y            object
dtype: object
In [10]:
df.columns
Out[10]:
Index(['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
       'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
       'previous', 'poutcome', 'y'],
      dtype='object')
In [7]:
df.describe()
Out[7]:
	age	balance	day	duration	campaign	pdays	previous
count	4521.000000	4521.000000	4521.000000	4521.000000	4521.000000	4521.000000	4521.000000
mean	41.170095	1422.657819	15.915284	263.961292	2.793630	39.766645	0.542579
std	10.576211	3009.638142	8.247667	259.856633	3.109807	100.121124	1.693562
min	19.000000	-3313.000000	1.000000	4.000000	1.000000	-1.000000	0.000000
25%	33.000000	69.000000	9.000000	104.000000	1.000000	-1.000000	0.000000
50%	39.000000	444.000000	16.000000	185.000000	2.000000	-1.000000	0.000000
75%	49.000000	1480.000000	21.000000	329.000000	3.000000	-1.000000	0.000000
max	87.000000	71188.000000	31.000000	3025.000000	50.000000	871.000000	25.000000
In [8]:
#checking if our data has any null values in it
df.isna().sum()
Out[8]:
age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
contact      0
day          0
month        0
duration     0
campaign     0
pdays        0
previous     0
poutcome     0
y            0
dtype: int64
In [13]:
#No Null values in given dataset
In [14]:
df.duplicated().sum()
#No duplicate values in given dataset
Out[14]:
0
In [10]:
df['y'].value_counts()
Out[10]:
no     4000
yes     521
Name: y, dtype: int64
In [11]:
#looks like we have data imbalancing
In [12]:
df_his = px.histogram(df,x = 'age', color = 'y', marginal='box')
In [13]:
df_his.update_layout(bargap=0.2)
In [14]:
#it seems age do play a vital role in customer churn
In [15]:
sns.pairplot(df)
Out[15]:
<seaborn.axisgrid.PairGrid at 0x15309b3d9a0>
 
In [16]:
df.hist(bins = 50, figsize=(20,15))
Out[16]:
array([[<AxesSubplot:title={'center':'age'}>,
        <AxesSubplot:title={'center':'balance'}>,
        <AxesSubplot:title={'center':'day'}>],
       [<AxesSubplot:title={'center':'duration'}>,
        <AxesSubplot:title={'center':'campaign'}>,
        <AxesSubplot:title={'center':'pdays'}>],
       [<AxesSubplot:title={'center':'previous'}>, <AxesSubplot:>,
        <AxesSubplot:>]], dtype=object)
 
In [11]:
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4521 entries, 0 to 4520
Data columns (total 17 columns):
 #   Column     Non-Null Count  Dtype 
---  ------     --------------  ----- 
 0   age        4521 non-null   int64 
 1   job        4521 non-null   object
 2   marital    4521 non-null   object
 3   education  4521 non-null   object
 4   default    4521 non-null   object
 5   balance    4521 non-null   int64 
 6   housing    4521 non-null   object
 7   loan       4521 non-null   object
 8   contact    4521 non-null   object
 9   day        4521 non-null   int64 
 10  month      4521 non-null   object
 11  duration   4521 non-null   int64 
 12  campaign   4521 non-null   int64 
 13  pdays      4521 non-null   int64 
 14  previous   4521 non-null   int64 
 15  poutcome   4521 non-null   object
 16  y          4521 non-null   object
dtypes: int64(7), object(10)
memory usage: 600.6+ KB
In [12]:
#We have 7 features of int value attributes and 10 of object data types
#We'll copy the data in another dataframe to avoid data loss of orignal data
churn = df.copy()
In [18]:
churn.head()

Data Cleaning
In [15]:
#We can drop 'day' & 'month' columns as 'pdays' gives the number of days that passed by after the client was last contacted from a previous campaign
In [19]:
churn.drop(["day","month"],axis =1, inplace = True)
In [20]:
np.where(churn == 'unknown')
Out[20]:
(array([   0,    3,    3, ..., 4517, 4517, 4518], dtype=int64),
 array([13,  8, 13, ...,  8, 13, 13], dtype=int64))
In [21]:
churn.iloc[0:1,15:16]
In [22]:
#There are many unknown entries we need to replace them as null/Nan
#so we will be replacing "unknown" as nan
In [23]:
churn = churn.replace("unknown", np.nan)
In [24]:
churn.iloc[0:1,15:16]
In [25]:
#it will generate null values
pd.isna(churn).sum()
Out[25]:
age             0
job            38
marital         0
education     187
default         0
balance         0
housing         0
loan            0
contact      1324
duration        0
campaign        0
pdays           0
previous        0
poutcome     3705
y               0
dtype: int64
In [26]:
(3705/churn.poutcome.shape[0])*100
Out[26]:
81.95089581950896
After replacing unknown values as null we find 'poutcome' has 3705 null entries i.e. about 81.9% null values, this feature is of no need we can simply drop it.
In [27]:
churn.drop('poutcome', inplace=True, axis =1)
In [28]:
(1324/churn.contact.shape[0])*100
Out[29]:
29.285556292855563
In [30]:
#Similarly it doesnt matter if 'contact' was through cellular or telephone it doesnt affect the target variable, so we can drop this feature

churn.drop('contact', axis = 1, inplace = True)
In [31]:
Data Preprocessing
In [ ]:
#As 'job' & 'education' has lower null values we'll fill them using fillna method instead of dropping
In [32]:
churn["job"].fillna(method = "ffill",inplace=True)
In [33]:
churn["education"].fillna(method = "ffill",inplace=True)
In [34]:
churn.isna().sum()
Out[34]:
age          0
job          0
marital      0
education    0
default      0
balance      0
housing      0
loan         0
duration     0
campaign     0
pdays        0
previous     0
y            0
dtype: int64
In [35]:
#Then we will be replacing yes and no as 1 and 0
#Now we need to convert all categorical data to numerical data. This will allow us to perform calculations on our data
In [36]:
list = ['default', 'housing','loan', 'y']
In [37]:
def binary_map(q):
    return q.map({'yes':1,'no':0})
In [38]:
churn[list]=churn[list].apply(binary_map)
In [39]:
Dummy Variable
In [16]:
#we can use the One Hot Encoder and dummy variables
In [40]:
job = pd.get_dummies(churn['job'])
In [41]:
job.head()
In [42]:
churn = churn.drop('job', axis = 1)
In [43]:
churn.shape
Out[43]:
(4521, 12)
In [44]:
churn = pd.concat([churn,job], axis = 1)
In [45]:
churn = pd.get_dummies(churn)

#we can drop the primary column as it can be predicted with the help of other dummy variables

churn = churn.drop('education_primary', axis =1)
In [49]:
churn = churn.drop('marital_divorced', axis =1)
In [50]
churn.isna().sum()
age                    0
default                0
balance                0
housing                0
loan                   0
duration               0
campaign               0
pdays                  0
previous               0
y                      0
admin.                 0
blue-collar            0
entrepreneur           0
housemaid              0
management             0
retired                0
self-employed          0
services               0
student                0
technician             0
unemployed             0
marital_married        0
marital_single         0
education_secondary    0
education_tertiary     0
dtype: int64
Univariate analysis
In [61]:
for i, predictor in enumerate(churn.drop(columns=['y','age','campaign','previous', 'balance', 'duration','pdays'])):
    ax = sns.countplot(data =churn, x = predictor, hue='y')
    if predictor == "PaymentMethod": 
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
        plt.tight_layout()
        plt.show()
    else:
        plt.tight_layout()
        plt.show()
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
In [62]:
churn.shape
Out[62]:
(4521, 25)
In [63]:
corr_matrix = churn.corr()
In [64]:
corr_matrix['y'].sort_values(ascending=False)
Out[64]:
y                      1.000000
duration               0.401118
previous               0.116714
pdays                  0.104087
retired                0.088736
education_tertiary     0.054962
student                0.047809
marital_single         0.045815
age                    0.045092
management             0.034558
balance                0.017905
admin.                 0.007753
housemaid              0.004339
default                0.001303
unemployed             0.000078
self-employed         -0.004614
technician            -0.009555
entrepreneur          -0.017088
services              -0.024819
education_secondary   -0.029365
campaign              -0.061147
marital_married       -0.064643
blue-collar           -0.069502
loan                  -0.070517
housing               -0.104683
Name: y, dtype: float64
In [69]:
corr_matrix['y'].sort_values(ascending=False).plot(kind = 'bar', figsize = (15,10))
Out[69]:
<AxesSubplot:>
 
Building Model
In [101]:
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
In [83]:
x = churn.drop('y', axis =1)
In [84]:
x.shape
Out[84]:
(4521, 24)
In [85]:
y = churn['y']
In [86]:
y.shape
Out[86]:
(4521,)
As the data set were highly unbalanced, we will use upsampling in order to increase accuracy using SMOTEENN
In [88]:
sm = SMOTEENN(random_state = 0)
In [90]:
x_resampled_sm, y_resampled_sm = sm.fit_resample(x,y)
Decision Tree Classifier
In [91]:
dt = DecisionTreeClassifier(criterion = 'gini', random_state = 100)
In [93]:
x_train,x_test,y_train,y_test = train_test_split(x_resampled_sm, y_resampled_sm,test_size=0.2, random_state = 42)
In [94]:
dt.fit(x_train,y_train)
Out[94]:
 DecisionTreeClassifier
DecisionTreeClassifier(random_state=100)
In [95]:
dt_pred = dt.predict(x_test)
In [98]:
dt.score(x_test,y_test)
Out[98]:
0.9375
In [97]:
print(classification_report(y_test, dt_pred, labels=[0,1]))
              precision    recall  f1-score   support

           0       0.94      0.92      0.93       551
           1       0.93      0.95      0.94       633

    accuracy                           0.94      1184
   macro avg       0.94      0.94      0.94      1184
weighted avg       0.94      0.94      0.94      1184

In [103]:
print(metrics.confusion_matrix(y_test, dt_pred))
[[509  42]
 [ 32 601]]
In [19]:
# This method gave 93.75% accuracy score
Random Forest Classifier
In [142]:
rf = RandomForestClassifier(n_estimators = 100, random_state = 100, criterion='gini')
In [143]:
rf.fit(x_train,y_train)
Out[143]:
 RandomForestClassifier
RandomForestClassifier(random_state=100)
In [144]:
rf_pred = rf.predict(x_test)
In [147]:
rf.score(x_test,y_test)
#score achieved by entropy 0.9543918918918919
Out[147]:
0.9611486486486487
In [148]:
print(classification_report(y_test,rf_pred,labels = [0,1]))
              precision    recall  f1-score   support

           0       0.96      0.96      0.96       551
           1       0.96      0.96      0.96       633

    accuracy                           0.96      1184
   macro avg       0.96      0.96      0.96      1184
weighted avg       0.96      0.96      0.96      1184

In [149]:
print(metrics.confusion_matrix(y_test,rf_pred))
[[528  23]
 [ 23 610]]
In [ ]:
# 96.11% accuracy
XGBoost Classifier
In [150]:
xgb = XGBClassifier(n_estimators=100, random_state = 100)
In [151]:
xgb.fit(x_train,y_train)
Out[151]:
 XGBClassifier
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=None, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, num_parallel_tree=None,
              predictor=None, random_state=100, ...)
In [152]:
xgb_pred = xgb.predict(x_test)
In [153]:
xgb.score(x_test,y_test)
Out[153]:
0.964527027027027
In [155]:
print(classification_report(y_test, xgb_pred))
              precision    recall  f1-score   support

           0       0.96      0.96      0.96       551
           1       0.97      0.97      0.97       633

    accuracy                           0.96      1184
   macro avg       0.96      0.96      0.96      1184
weighted avg       0.96      0.96      0.96      1184

In [158]:
print(metrics.confusion_matrix(y_test,xgb_pred))
[[530  21]
 [ 21 612]]
In [20]:
# Acurracy achieved by this model is 96.45%
Logistic Regression
In [167]:
lr = LogisticRegression(max_iter = 300,random_state = 100)
In [168]:
lr.fit(x_train,y_train)
C:\Users\hp\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:458: ConvergenceWarning:

lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression

Out[168]:
 LogisticRegression
LogisticRegression(max_iter=300, random_state=100)
In [169]:
lr_pred = lr.predict(x_test)
In [171]:
lr.score(x_test,y_test)
Out[171]:
0.924831081081081
In [172]:
print(classification_report(y_test,lr_pred))
              precision    recall  f1-score   support

           0       0.91      0.93      0.92       551
           1       0.94      0.92      0.93       633

    accuracy                           0.92      1184
   macro avg       0.92      0.93      0.92      1184
weighted avg       0.93      0.92      0.92      1184

In [174]:
print(confusion_matrix(y_test,lr_pred))
[[514  37]
 [ 52 581]]
In [ ]:
# # Acurracy achieved by this model is just 92.48%
PCA
In [176]:
from sklearn.decomposition import PCA
In [192]:
pca = PCA(n_components=0.95)
In [193]:
x_pca_train = pca.fit_transform(x_train)
x_pca_test = pca.transform(x_test)
In [194]:
pca.explained_variance_ratio_
Out[194]:
array([0.97636099])
In [201]:
pca.explained_variance_ratio_
Out[201]:
array([0.97636099])
In [195]:
xgb_pca = xgb.fit(x_pca_train,y_train)
In [196]:
xgb_pca_pred = xgb.predict(x_pca_test)
In [197]:
xgb.score(x_pca_test,y_test)
Out[197]:
0.6300675675675675
In [198]:
print(classification_report(y_test,xgb_pca_pred))
              precision    recall  f1-score   support

           0       0.62      0.55      0.58       551
           1       0.64      0.70      0.67       633

    accuracy                           0.63      1184
   macro avg       0.63      0.62      0.62      1184
weighted avg       0.63      0.63      0.63      1184

In [199]:
print(confusion_matrix(y_test,xgb_pca_pred))
[[302 249]
 [189 444]]
With PCA, we couldn't see any better results, hence let's finalise the model which was created by XGBoost Classifier (96.45% accuracy), and fine tune it using various techniques
Cross Validation
In [202]:
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
In [210]:
kfold = KFold(n_splits=10, shuffle= True, random_state=42)
In [211]:
scores = cross_val_score(xgb,x_train,y_train, cv=kfold)
In [212]:
scores
Out[212]:
array([0.97046414, 0.97679325, 0.97040169, 0.96828753, 0.97463002,
       0.97463002, 0.96828753, 0.95983087, 0.97040169, 0.9577167 ])
In [213]:
print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
Accuracy: 96.91% (0.59%)
Accuracy: 96.91% (0.59%)
We achieved 97.05% accuracy after performing cross validation which is great , now lets try GBC on our model/XGBoost to find its best parameters
Finding Optimal Parameter
In [217]:
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import uniform, randint
In [218]:
GBC = GradientBoostingClassifier()
In [250]:
params = {
    "n_estimators": [100,500,1000],
    "max_depth": [4,6,8],
    "learning_rate": [0.01,0.02 ,0.3],
    "subsample": [0.9,0.5, 0.2],
}
In [251]:
grid_GBC = GridSearchCV(estimator=GBC,param_grid = params,cv = 2 ,n_jobs = 1)
grid_GBC.fit(x_train,y_train)
Out[251]:
 GridSearchCV
 estimator: GradientBoostingClassifier
 GradientBoostingClassifier
In [252]:
print("The best estimator across ALL searched params:\n",grid_GBC.best_estimator_)
 The best estimator across ALL searched params:
 GradientBoostingClassifier(learning_rate=0.02, max_depth=8, n_estimators=1000,
                           subsample=0.5)
In [253]:
print("The best score across ALL searched params: \n", grid_GBC.best_score_)
The best score across ALL searched params: 
 0.9587912087912087
In [254]:
print("The best parameters across ALL searched params: \n", grid_GBC.best_params_)
The best parameters across ALL searched params: 
 {'learning_rate': 0.02, 'max_depth': 8, 'n_estimators': 1000, 'subsample': 0.5}
Final Model
In [21]:
#Lets fine tune our model with these parameters and produce our final model
In [260]:
model_xg_smote=XGBClassifier(colsample_bytree= 0.3406585285177396, gamma= 0.4330880728874676, learning_rate= 0.2, max_depth= 8, n_estimators=1000, reg_lambda= 0.041168988591604894, subsample=0.5)
model_xg_smote.fit(x_train,y_train)
Out[260]:
 XGBClassifier
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=0.3406585285177396, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=0.4330880728874676, gpu_id=None, grow_policy=None,
              importance_type=None, interaction_constraints=None,
              learning_rate=0.2, max_bin=None, max_cat_threshold=None,
              max_cat_to_onehot=None, max_delta_step=None, max_depth=8,
              max_leaves=None, min_child_weight=None, missing=nan,
              monotone_constraints=None, n_estimators=1000, n_jobs=None,
              num_parallel_tree=None, predictor=None, random_state=None, ...)
In [261]:
final_y_pred = model_xg_smote.predict(x_test)
In [262]:
model_score = model_xg_smote.score(x_test, y_test)
In [263]:
print(model_score)
print(metrics.classification_report(y_test, final_y_pred))
print(metrics.confusion_matrix(y_test, final_y_pred))
0.9662162162162162
              precision    recall  f1-score   support

           0       0.96      0.97      0.96       551
           1       0.97      0.97      0.97       633

    accuracy                           0.97      1184
   macro avg       0.97      0.97      0.97      1184
weighted avg       0.97      0.97      0.97      1184

[[532  19]
 [ 21 612]]
In [264]:
kfold = KFold(n_splits=10, shuffle= True, random_state=42)
In [266]:
scores = cross_val_score(model_xg_smote,x_train,y_train, cv=kfold)
In [267]:
print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
Accuracy: 96.85% (0.40%)
We have achieved accuracy of 96.85% after cross validation and fine tuning our model with the best parameters
AUC-ROC
In [268]:
from sklearn.metrics import roc_auc_score,roc_curve
In [270]:
y_pred_prob = model_xg_smote.predict_proba(x_test)[:,1]
In [271]:
auc_roc = roc_auc_score(y_test, y_pred_prob)
print("AUC-ROC Score: ", auc_roc)
AUC-ROC Score:  0.9957710094815401
AUC-ROC score is 0.999547 which is almost 1 which indicates our model is almost perfect
In [273]:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr,label='AUC-ROC = %0.2f' % auc_roc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
 
Saving our model
In [275]:
import pickle
In [276]:
filename = 'churn_in_bank_XGB_Model.sav'
pickle.dump(model_xg_smote, open(filename, 'wb'))
load_model = pickle.load(open(filename, 'rb'))
model_score_r4 = load_model.score(x_test, y_test)
model_score_r4
Out[276]:
0.9662162162162162
In [ ]:
 

