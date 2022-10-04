import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
#creating a pipeline
prep = Pipeline([
    ('stand', StandardScaler()),
    ('norm', Normalizer())
])
#feature preparation function
def prep_features(csv):
    train = pd.read_csv(csv)
    num = train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].fillna(0).values
    txt = train[['Sex', 'Cabin', 'Embarked']].values
    le = LabelEncoder()
    for i in range(len(txt[0])):
        txt[:, i] = le.fit_transform(txt[:, i])
    x = np.concatenate((txt, num), axis = 1)
    return x
#labels preparation function
def prep_output(csv):
    y = pd.read_csv(csv)[['Survived']].values
    return y
#initialyzing the data
x_train = prep_features('train.csv')
y_train = prep_output('train.csv')
x_test = prep_features('test.csv')
y_test = prep_output('gender_submission.csv')
#training the model
x_train = prep.fit_transform(x_train)
clf = LogisticRegression(random_state = 0).fit(x_train, y_train.ravel())
p = clf.predict(x_test)
#creating a dataframe of the result
dataset = pd.DataFrame({'Survived': p[:]})
id = pd.read_csv("gender_submission.csv")['PassengerId']
df = pd.concat([id, dataset],axis = 1)
df.to_csv('submission.csv', sep=',', index = False)
#displaying the score
clf.score(x_test, y_test)
i = accuracy_score(y_test, p)
print("score : ", i)
