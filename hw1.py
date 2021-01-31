import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')

# Q1
print(train.keys())

#  Q2, Q3, and Q4 - dataset observation
categorical_features = ['Survived', 'Pclass', 'Sex', 'Embarked']
numerical_features = ['PassengerId', 'Age', 'SibSp', 'Parch', 'Fare']
mixed_features = ['Ticket']

# Q5
print('Training set missing values:')
print(train.isnull().sum())
print('Test set missing values:')
print(test.isnull().sum())

# Q6
print(train.dtypes)

# Q7
for f in numerical_features:
    print(f)
    print('count: ', train[f].count())
    print('mean: ', train[f].mean())
    print('std: ', train[f].std())
    print('min: ', train[f].min())
    print('25%: ', train[f].quantile(0.25))
    print('50%: ', train[f].quantile(0.5))
    print('75%: ', train[f].quantile(0.75))
    print('max: ', train[f].max())

# Q8
for f in categorical_features:
    print(f)
    print('count: ', train[f].count())
    print('unique: ', len(train[f].unique()))
    print('value counts:')
    print(train[f].value_counts())

# Q9
pclass1 = train[train['Pclass'] == 1]
print(pclass1['Survived'].value_counts())
print('pclass 1 survival ratio: ', pclass1['Survived'].value_counts()[1] / pclass1['Survived'].count())

# Q10
male = train[train['Sex'] == 'male']
female = train[train['Sex'] == 'female']
print('male survival ratio: ', male['Survived'].value_counts()[1] / male['Survived'].count())
print('female survival ratio: ', female['Survived'].value_counts()[1] / female['Survived'].count())

# Q11
survived = train[train['Survived'] == 1]
not_survived = train[train['Survived'] == 0]

not_survived['Age'].plot.hist(bins=16)
plt.title('Survived = 0')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 80])
plt.xlim([0, 80])
plt.show()

survived['Age'].plot.hist(bins=16)
plt.title('Survived = 1')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 80])
plt.xlim([0, 80])
plt.show()

infants = train[train['Age'] <= 4]
print('infant survival ratio: ', infants['Survived'].value_counts()[1] / infants['Survived'].count())
oldest = train[train['Age'] == 80]
print('oldest survival ratio: ', oldest['Survived'].value_counts()[1] / oldest['Survived'].count())
over15 = train[train['Age'] >= 15]
age15_25 = over15[over15['Age'] <= 25]
print('15-25 y/o survival ratio: ', age15_25['Survived'].value_counts()[1] / age15_25['Survived'].count())

# Q12
pclass1_survived = survived[survived['Pclass'] == 1]
pclass1_not_survived = not_survived[not_survived['Pclass'] == 1]
pclass2_survived = survived[survived['Pclass'] == 2]
pclass2_not_survived = not_survived[not_survived['Pclass'] == 2]
pclass3_survived = survived[survived['Pclass'] == 3]
pclass3_not_survived = not_survived[not_survived['Pclass'] == 3]

pclass1_survived['Age'].plot.hist(bins=16)
plt.title('Pclass = 1 | Survived = 1')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 50])
plt.xlim([0, 80])
plt.show()

pclass1_not_survived['Age'].plot.hist(bins=16)
plt.title('Pclass = 1 | Survived = 0')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 50])
plt.xlim([0, 80])
plt.show()

pclass2_survived['Age'].plot.hist(bins=16)
plt.title('Pclass = 2 | Survived = 1')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 50])
plt.xlim([0, 80])
plt.show()

pclass2_not_survived['Age'].plot.hist(bins=16)
plt.title('Pclass = 2 | Survived = 0')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 50])
plt.xlim([0, 80])
plt.show()

pclass3_survived['Age'].plot.hist(bins=16)
plt.title('Pclass = 3 | Survived = 1')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 50])
plt.xlim([0, 80])
plt.show()

pclass3_not_survived['Age'].plot.hist(bins=16)
plt.title('Pclass = 3 | Survived = 0')
plt.xlabel('Age')
plt.ylabel('')
plt.ylim([0, 50])
plt.xlim([0, 80])
plt.show()

print(train['Pclass'].value_counts())
print(not_survived['Pclass'].value_counts())

infants_pclass2 = infants[infants['Pclass'] == 2]
print('infant pclass=2 survival ratio: ', infants_pclass2['Survived'].value_counts()[1] / infants_pclass2['Survived'].count())
infants_pclass3 = infants[infants['Pclass'] == 3]
print('infant pclass=3 survival ratio: ', infants_pclass3['Survived'].value_counts()[1] / infants_pclass3['Survived'].count())

pclass1 = train[train['Pclass'] == 1]
pclass2 = train[train['Pclass'] == 2]
pclass3 = train[train['Pclass'] == 3]

print('Pclass=1 survival ratio: ', pclass1['Survived'].value_counts()[1] / pclass1['Survived'].count())

print('Pclass=1 age distribution: mean ', pclass1['Age'].mean(), ' std ', pclass1['Age'].std())
print('Pclass=2 age distribution: mean ', pclass2['Age'].mean(), ' std ', pclass2['Age'].std())
print('Pclass=3 age distribution: mean ', pclass3['Age'].mean(), ' std ', pclass3['Age'].std())

# Q13
embarked_s_survived = survived[survived['Embarked'] == 'S']
embarked_s_not_survived = not_survived[not_survived['Embarked'] == 'S']
embarked_c_survived = survived[survived['Embarked'] == 'C']
embarked_c_not_survived = not_survived[not_survived['Embarked'] == 'C']
embarked_q_survived = survived[survived['Embarked'] == 'Q']
embarked_q_not_survived = not_survived[not_survived['Embarked'] == 'Q']

embarked_s_male_survived = embarked_s_survived[embarked_s_survived['Sex'] == 'male']
embarked_s_male_not_survived = embarked_s_not_survived[embarked_s_not_survived['Sex'] == 'male']
embarked_s_female_survived = embarked_s_survived[embarked_s_survived['Sex'] == 'female']
embarked_s_female_not_survived = embarked_s_not_survived[embarked_s_not_survived['Sex'] == 'female']
embarked_c_male_survived = embarked_c_survived[embarked_c_survived['Sex'] == 'male']
embarked_c_male_not_survived = embarked_c_not_survived[embarked_c_not_survived['Sex'] == 'male']
embarked_c_female_survived = embarked_c_survived[embarked_c_survived['Sex'] == 'female']
embarked_c_female_not_survived = embarked_c_not_survived[embarked_c_not_survived['Sex'] == 'female']
embarked_q_male_survived = embarked_q_survived[embarked_q_survived['Sex'] == 'male']
embarked_q_male_not_survived = embarked_q_not_survived[embarked_q_not_survived['Sex'] == 'male']
embarked_q_female_survived = embarked_q_survived[embarked_q_survived['Sex'] == 'female']
embarked_q_female_not_survived = embarked_q_not_survived[embarked_q_not_survived['Sex'] == 'female']

plt.title('Embarked = S | Survived = 1')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.bar(['male', 'female'], [embarked_s_male_survived['Fare'].mean(), embarked_s_female_survived['Fare'].mean()])
plt.ylim([0, 90])
plt.show()

plt.title('Embarked = S | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.bar(['male', 'female'], [embarked_s_male_not_survived['Fare'].mean(), embarked_s_female_not_survived['Fare'].mean()])
plt.ylim([0, 90])
plt.show()

plt.title('Embarked = C | Survived = 1')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.bar(['male', 'female'], [embarked_c_male_survived['Fare'].mean(), embarked_c_female_survived['Fare'].mean()])
plt.ylim([0, 90])
plt.show()

plt.title('Embarked = C | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.bar(['male', 'female'], [embarked_c_male_not_survived['Fare'].mean(), embarked_c_female_not_survived['Fare'].mean()])
plt.ylim([0, 90])
plt.show()

plt.title('Embarked = Q | Survived = 1')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.bar(['male', 'female'], [embarked_q_male_survived['Fare'].mean(), embarked_q_female_survived['Fare'].mean()])
plt.ylim([0, 90])
plt.show()

plt.title('Embarked = Q | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')
plt.bar(['male', 'female'], [embarked_q_male_not_survived['Fare'].mean(), embarked_q_female_not_survived['Fare'].mean()])
plt.ylim([0, 90])
plt.show()

# Q14
print('ticket duplicate rate: ', 1 - len(train['Ticket'].unique()) / train['Ticket'].count())

# Q15
print('cabin training missing values: ', train.isnull().sum()['Cabin'])
print('cabin training missing values %: ', train.isnull().sum()['Cabin'] / (train.isnull().sum()['Cabin'] + train['Cabin'].count()))
print('cabin test missing values: ', test.isnull().sum()['Cabin'])
print('cabin test missing values %: ', test.isnull().sum()['Cabin'] / (test.isnull().sum()['Cabin'] + test['Cabin'].count()))
print('cabin missing values: ', train.isnull().sum()['Cabin'] + test.isnull().sum()['Cabin'])
print('cabin missing values %: ', (train.isnull().sum()['Cabin'] + test.isnull().sum()['Cabin']) / (train.isnull().sum()['Cabin'] + test.isnull().sum()['Cabin'] + train['Cabin'].count() + test['Cabin'].count()))

# Q16
train['Gender'] = train['Sex'].astype('category').cat.codes

# Q17
train['Age'] = SimpleImputer(strategy='mean').fit_transform(np.array(train['Age']).reshape(-1, 1))

# Q18
train['Embarked'] = SimpleImputer(strategy='most_frequent').fit_transform(np.array(train['Embarked']).reshape(-1, 1))

# Q19
test['Fare'] = SimpleImputer(strategy='most_frequent').fit_transform(np.array(test['Fare']).reshape(-1, 1))

# Q20
fare = np.array(train['Fare'])
fare_indicator = np.zeros_like(fare, np.int)
fare_indicator[fare > 7.91] = 1
fare_indicator[fare > 14.454] = 2
fare_indicator[fare > 31] = 3
train['Fare'] = fare_indicator
