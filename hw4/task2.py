import numpy as np
from sklearn import datasets

from kmeans import KMeans


def info(title, kmeans):
    print(title)
    print('Accuracy: ', kmeans.accuracy())
    print('SSE (Euclidean): ', kmeans.sse('euclidean'))
    print('SSE (Cosine): ', kmeans.sse('cosine'))
    print('SSE (Jaccard): ', kmeans.sse('jaccard'))
    print('Time: ', kmeans.time)
    print('Iterations: ', kmeans.n_iterations)
    print('Centroids: ', kmeans.centroids)


iris = datasets.load_iris()

n_clusters = len(iris['target_names'])
centroids = iris['data'][np.random.randint(0, iris['data'].shape[0], n_clusters)]
kmeans = KMeans(n_clusters)

print('Initial Centroids: ', centroids)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='euclidean', n_iter=100)
info('Euclidean - stop: 100 iterations', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='euclidean', stop_condition='sse')
info('Euclidean - stop: SSE', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='euclidean', stop_condition='centroid')
info('Euclidean - stop: centroid', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='cosine', n_iter=100)
info('Cosine - stop: 100 iterations', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='cosine', stop_condition='sse')
info('Cosine - stop: SSE', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='cosine', stop_condition='centroid')
info('Cosine - stop: centroid', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='jaccard', n_iter=100)
info('Jaccard - stop: 100 iterations', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='jaccard', stop_condition='sse')
info('Jaccard - stop: SSE', kmeans)

kmeans.fit(iris['data'], y=iris['target'], centroids=centroids.copy(), distance='jaccard', stop_condition='centroid')
info('Jaccard - stop: centroid', kmeans)