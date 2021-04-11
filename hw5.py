import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD
from surprise import KNNBasic
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate

ifig = 0

file_path = './archive/ratings_small.csv'
df = pd.read_csv(file_path)

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

pmf = SVD(biased=False)
pmf_cv = cross_validate(pmf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

sim_options = {'user_based': True}
user_cf = KNNBasic(sim_options=sim_options)
user_cf_cv = cross_validate(user_cf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

sim_options = {'user_based': False}
item_cf = KNNBasic(sim_options=sim_options)
item_cf_cv = cross_validate(item_cf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


user_cf_similarities = {}
item_cf_similarities = {}
similarities = ['cosine', 'msd', 'pearson']
for s in similarities:
    sim_options = {'name': s, 'user_based': True}
    user_cf = KNNBasic(sim_options=sim_options)
    user_cf_similarities[s] = cross_validate(user_cf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    sim_options = {'name': s, 'user_based': False}
    item_cf = KNNBasic(sim_options=sim_options)
    item_cf_similarities[s] = cross_validate(item_cf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


user_cf_k = {}
item_cf_k = {}
k_values = np.arange(10, 105, 5)
for k in k_values:
    sim_options = {'user_based': True}
    user_cf = KNNBasic(k=k, sim_options=sim_options)
    user_cf_k[k] = cross_validate(user_cf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    sim_options = {'user_based': False}
    item_cf = KNNBasic(k=k, sim_options=sim_options)
    item_cf_k[k] = cross_validate(item_cf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


user_cf_similarities_mae = [user_cf_similarities[s]['test_mae'].mean() for s in similarities]
user_cf_similarities_rmse = [user_cf_similarities[s]['test_rmse'].mean() for s in similarities]
item_cf_similarities_mae = [item_cf_similarities[s]['test_mae'].mean() for s in similarities]
item_cf_similarities_rmse = [item_cf_similarities[s]['test_rmse'].mean() for s in similarities]

user_cf_k_mae = [user_cf_k[k]['test_mae'].mean() for k in k_values]
user_cf_k_rmse = [user_cf_k[k]['test_rmse'].mean() for k in k_values]
item_cf_k_mae = [item_cf_k[k]['test_mae'].mean() for k in k_values]
item_cf_k_rmse = [item_cf_k[k]['test_rmse'].mean() for k in k_values]


print('5-fold cross-validation results:')

print('PMF - MAE')
print(pmf_cv['test_mae'])
print('Average: ', np.mean(pmf_cv['test_mae']))
print('PMF - RMSE')
print(pmf_cv['test_rmse'])
print('Average: ', np.mean(pmf_cv['test_rmse']))

print('User-based CF - MAE')
print(user_cf_cv['test_mae'])
print('Average: ', np.mean(user_cf_cv['test_mae']))
print('User-based CF - RMSE')
print(user_cf_cv['test_rmse'])
print('Average: ', np.mean(user_cf_cv['test_rmse']))

print('Item-based CF - MAE')
print(item_cf_cv['test_mae'])
print('Average: ', np.mean(item_cf_cv['test_mae']))
print('Item-based CF - RMSE')
print(item_cf_cv['test_rmse'])
print('Average: ', np.mean(item_cf_cv['test_rmse']))


print('Similarities results:')

print('User-based CF - MAE')
for i, s in enumerate(similarities):
    print(s + ' - ' + str(user_cf_similarities_mae[i]))

print('User-based CF - RMSE')
for i, s in enumerate(similarities):
    print(s + ' - ' + str(user_cf_similarities_rmse[i]))

print('Item-based CF - MAE')
for i, s in enumerate(similarities):
    print(s + ' - ' + str(item_cf_similarities_mae[i]))

print('Item-based CF - RMSE')
for i, s in enumerate(similarities):
    print(s + ' - ' + str(item_cf_similarities_rmse[i]))


ifig += 1
labels = ['user-based', 'item-based']
x = np.arange(0, len(labels) * 2.5, 2.5)
width = 0.25
plt.figure(ifig)
fig, ax = plt.subplots()
for i in range(len(similarities)):
    ax.bar(x - (width * len(similarities)) / 2 + i * width, [user_cf_similarities_mae[i], item_cf_similarities_mae[i]], width, label=similarities[i])
plt.xticks(x - width / 2, labels)
plt.title('collaborative filtering models - similarities vs. mae')
plt.ylabel('mae')
plt.grid(axis='y')
plt.legend()
plt.show()

ifig += 1
labels = ['user-based', 'item-based']
x = np.arange(0, len(labels) * 2.5, 2.5)
width = 0.25
plt.figure(ifig)
fig, ax = plt.subplots()
for i in range(len(similarities)):
    ax.bar(x - (width * len(similarities)) / 2 + i * width, [user_cf_similarities_rmse[i], item_cf_similarities_rmse[i]], width, label=similarities[i])
plt.xticks(x - width / 2, labels)
plt.title('collaborative filtering models - similarities vs. rmse')
plt.ylabel('rmse')
plt.grid(axis='y')
plt.legend()
plt.show()


print('Number of neighbors results:')

print('User-based CF - MAE')
print('Best K: ', k_values[np.argmin(user_cf_k_mae)])
print('Best K MAE: ', np.min(user_cf_k_mae))

print('User-based CF - RMSE')
print('Best K: ', k_values[np.argmin(user_cf_k_rmse)])
print('Best K RMSE: ', np.min(user_cf_k_rmse))

print('Item-based CF - MAE')
print('Best K: ', k_values[np.argmin(item_cf_k_mae)])
print('Best K MAE: ', np.min(item_cf_k_mae))

print('Item-based CF - RMSE')
print('Best K: ', k_values[np.argmin(item_cf_k_rmse)])
print('Best K RMSE: ', np.min(item_cf_k_rmse))


ifig += 1
plt.figure(ifig)
plt.title('collaborative filtering models - number of neighbors vs. rmse')
plt.plot(k_values, user_cf_k_rmse, marker='o', label='user-based')
plt.plot(k_values, item_cf_k_rmse, marker='o', label='item-based')
plt.xlabel('number of neighbors')
plt.ylabel('rmse')
plt.xlim([10, 100])
plt.grid(which='both', axis='both')
plt.legend()
plt.show()

ifig += 1
plt.figure(ifig)
plt.title('collaborative filtering models - number of neighbors vs. mae')
plt.plot(k_values, user_cf_k_mae, marker='o', label='user-based')
plt.plot(k_values, item_cf_k_mae, marker='o', label='item-based')
plt.xlabel('number of neighbors')
plt.ylabel('mae')
plt.xlim([10, 100])
plt.grid(which='both', axis='both')
plt.legend()
plt.show()
