import numpy as np

from kmeans import KMeans

data = np.array([
    [3., 5.],
    [3., 4.],
    [2., 8.],
    [2., 3.],
    [6., 2.],
    [6., 4.],
    [7., 3.],
    [7., 4.],
    [8., 5.],
    [7., 6.],
])

n_clusters = 2
x_label = '# wins in season 2016'
y_label = '# wins in season 2017'

kmeans = KMeans(n_clusters)
kmeans.fit(data, centroids=np.array([[4., 6.], [5., 4.]]), distance='manhattan', n_iter=1)
kmeans.plot_2d(title='initial centroids: (4, 6) and (5, 4) - manhattan distance', x_label=x_label, y_label=y_label)

kmeans.fit(data, centroids=np.array([[4., 6.], [5., 4.]]), distance='euclidean', n_iter=1)
kmeans.plot_2d(title='initial centroids: (4, 6) and (5, 4) - euclidean distance', x_label=x_label, y_label=y_label)

kmeans.fit(data, centroids=np.array([[3., 3.], [8., 3.]]), distance='manhattan', n_iter=1)
kmeans.plot_2d(title='initial centroids: (3, 3) and (8, 3) - manhattan distance', x_label=x_label, y_label=y_label)

kmeans.fit(data, centroids=np.array([[3., 2.], [4., 8.]]), distance='manhattan', n_iter=1)
kmeans.plot_2d(title='initial centroids: (3, 2) and (4, 8) - manhattan distance', x_label=x_label, y_label=y_label)



