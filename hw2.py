import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn import tree, model_selection, ensemble

train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')
combined = pd.concat([train, test])


def preprocess_age(df):
    df['Age'] = SimpleImputer(strategy='mean').fit_transform(np.array(df['Age']).reshape(-1, 1))

    age = np.array(df['Age'])
    is_child = np.zeros_like(age, np.int)
    is_child[age <= 5] = 1
    df['Child'] = is_child


def preprocess_fare(df):
    df['Fare'] = SimpleImputer(strategy='mean').fit_transform(np.array(df['Fare']).reshape(-1, 1))

    fare = np.array(df['Fare'])
    fare_indicator = np.zeros_like(fare, np.int)
    fare_indicator[fare > 20] = 1
    fare_indicator[fare > 50] = 2
    df['FareLevel'] = fare_indicator


def preprocess_embarked(df):
    df['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(np.array(df['Embarked']).reshape(-1, 1))
    df['Embarked'] = train['Embarked'].astype('category').cat.codes


def preprocess_sex(df):
    df['Sex'] = train['Sex'].astype('category').cat.codes


def preprocess_ds(df):
    preprocess_age(df)
    preprocess_fare(df)
    preprocess_embarked(df)
    preprocess_sex(df)


preprocess_ds(train)
preprocess_ds(test)

print('Features Correlation')
print(train.corr())

selected_features = ['Pclass', 'Sex', 'Child', 'Embarked', 'FareLevel']
# selected_features = ['Pclass', 'Sex', 'Child', 'Embarked', 'Fare']

print('Selected Features')
print(selected_features)

y = train["Survived"]
X = train[selected_features]
dt = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
dt.fit(X, y)

plt.figure(figsize=(32, 24))
tree.plot_tree(dt, feature_names=selected_features, filled=True)
plt.show()

y_predicted = dt.predict(X)
print('Decision Tree')
print('Training accuracy: ', str(dt.score(X, y)))

cv = model_selection.KFold(n_splits=5)

dt_models = []
dt_train_scores = []
dt_valid_scores = []
for train_index, valid_index in cv.split(X):
    model = tree.DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    dt_train_scores.append(model.score(X.iloc[train_index], y.iloc[train_index]))
    dt_valid_scores.append(model.score(X.iloc[valid_index], y.iloc[valid_index]))
    dt_models.append(model)
print('K-Fold Decision Tree')
print('Training accuracies: ', dt_train_scores)
print('Validation accuracies: ', dt_valid_scores)
print('Average training accuracy: ', np.mean(dt_train_scores))
print('Average validation accuracy: ', np.mean(dt_valid_scores))


rf_models = []
rf_train_scores = []
rf_valid_scores = []
for train_index, valid_index in cv.split(X):
    model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=50, bootstrap=True, max_depth=5, min_samples_split=2)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    rf_train_scores.append(model.score(X.iloc[train_index], y.iloc[train_index]))
    rf_valid_scores.append(model.score(X.iloc[valid_index], y.iloc[valid_index]))
    rf_models.append(model)
print('K-Fold Random Forest')
print('Training accuracies: ', rf_train_scores)
print('Validation accuracies: ', rf_valid_scores)
print('Average training accuracy: ', np.mean(rf_train_scores))
print('Average validation accuracy: ', np.mean(rf_valid_scores))