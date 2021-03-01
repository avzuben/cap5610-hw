import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.impute import SimpleImputer
from sklearn import svm, model_selection


# Task 2
def task2_kernel(data):
    return np.concatenate([np.expand_dims(data[:, 0], -1), np.expand_dims(data[:, 0] * data[:, 1], -1)], axis=-1)


positive_points = np.array([(-1, 1), (1, -1)])
negative_points = np.array([(-1, -1), (1, 1)])
transformed_positive_points = task2_kernel(positive_points)
transformed_negative_points = task2_kernel(negative_points)

ifig = 0
plt.figure(ifig)
plt.scatter(transformed_positive_points[:, 0], transformed_positive_points[:, 1], label='positive')
plt.scatter(transformed_negative_points[:, 0], transformed_negative_points[:, 1], label='negative')
plt.plot(np.linspace(-2, 2, 11), np.zeros(11), color='black')
plt.plot(np.linspace(-2, 2, 11), np.ones(11), color='gray', linestyle='dashed')
plt.plot(np.linspace(-2, 2, 11), -np.ones(11), color='gray', linestyle='dashed')
plt.grid(which='both', axis='both')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.ylabel('x1x2')
plt.xlabel('x1')
plt.legend(loc='upper right')
plt.show()


# Task 5

positive_points = np.array([(1, 1), (2, 2), (2, 0)])
negative_points = np.array([(0, 0), (1, 0), (0, 1)])

# line equation
a = (0, 1.5)
b = (0.5, 1)
m = (a[1] - b[1]) / (a[0] - b[0])
c = a[1] - m * a[0]
x = np.linspace(-1, 3, 11)

ifig += 1
plt.figure(ifig)
plt.scatter(positive_points[:, 0], positive_points[:, 1], label='positive')
plt.scatter(negative_points[:, 0], negative_points[:, 1], label='negative')
plt.plot(x, m*x+c, color='black')
plt.grid(which='both', axis='both')
plt.xlim([-1, 3])
plt.ylim([-1, 3])
plt.ylabel('x2')
plt.xlabel('x1')
plt.legend(loc='upper right')
plt.show()


# Task 6

xx, yy = np.meshgrid(range(-5, 5), range(-5, 5))
z = np.ones_like(xx) * 0.5

ifig += 1
fig = plt.figure(ifig)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z, alpha=0.2)
ax.scatter(0, 1, 0, color='blue')
ax.scatter(np.sqrt(2), 1, 1, color='orange')
ax.scatter(-np.sqrt(2), 1, 1, color='orange')
ax.set_zlabel('x^2')
ax.set_xlabel('sqrt(2)x')
ax.set_ylabel('1')
plt.show()

# Task 7

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

print('Selected Features')
print(selected_features)

y = train["Survived"]
X = train[selected_features]

cv = model_selection.KFold(n_splits=5)

linear_train_scores = []
linear_valid_scores = []
quadratic_train_scores = []
quadratic_valid_scores = []
rbf_train_scores = []
rbf_valid_scores = []
for train_index, valid_index in cv.split(X):
    linear_model = svm.SVC(kernel='linear')
    linear_model.fit(X.iloc[train_index], y.iloc[train_index])
    linear_train_scores.append(linear_model.score(X.iloc[train_index], y.iloc[train_index]))
    linear_valid_scores.append(linear_model.score(X.iloc[valid_index], y.iloc[valid_index]))

    quadratic_model = svm.SVC(kernel='poly', degree=2)
    quadratic_model.fit(X.iloc[train_index], y.iloc[train_index])
    quadratic_train_scores.append(quadratic_model.score(X.iloc[train_index], y.iloc[train_index]))
    quadratic_valid_scores.append(quadratic_model.score(X.iloc[valid_index], y.iloc[valid_index]))

    rbf_model = svm.SVC(kernel='rbf')
    rbf_model.fit(X.iloc[train_index], y.iloc[train_index])
    rbf_train_scores.append(rbf_model.score(X.iloc[train_index], y.iloc[train_index]))
    rbf_valid_scores.append(rbf_model.score(X.iloc[valid_index], y.iloc[valid_index]))

print('K-Fold Linear SVM')
print('Training accuracies: ', linear_train_scores)
print('Validation accuracies: ', linear_valid_scores)
print('Average training accuracy: ', np.mean(linear_train_scores))
print('Average validation accuracy: ', np.mean(linear_valid_scores))

print('K-Fold Quadratic SVM')
print('Training accuracies: ', quadratic_train_scores)
print('Validation accuracies: ', quadratic_valid_scores)
print('Average training accuracy: ', np.mean(quadratic_train_scores))
print('Average validation accuracy: ', np.mean(quadratic_valid_scores))

print('K-Fold RBF SVM')
print('Training accuracies: ', rbf_train_scores)
print('Validation accuracies: ', rbf_valid_scores)
print('Average training accuracy: ', np.mean(rbf_train_scores))
print('Average validation accuracy: ', np.mean(rbf_valid_scores))

