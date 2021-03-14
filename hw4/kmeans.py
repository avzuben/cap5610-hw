import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import mode

from distances import euclidean_distance, manhattan_distance, cosine_distance, generalized_jaccard_distance


class KMeans:
    def __init__(self, k):
        self.k = k
        self.predicted = None
        self.data = None
        self.labels = None
        self.centroids = None
        self.distance = 'euclidean'
        self.n_iterations = 0
        self.time = 0
        self.ifig = 0

    def _calculate_distance(self, a, b, distance=None):
        if distance is None:
            distance = self.distance
        if distance == 'manhattan':
            return manhattan_distance(a, b)
        if distance == 'cosine':
            return cosine_distance(a, b)
        if distance == 'jaccard':
            return generalized_jaccard_distance(a, b)
        return euclidean_distance(a, b)

    def _predict(self):
        for i, p in enumerate(self.data):
            distances = []
            for c in self.centroids:
                distances.append(self._calculate_distance(p, c))
            self.predicted[i] = np.argmin(distances)

    def _calculate_centroids(self):
        last_centroids = self.centroids.copy()
        for i in range(self.k):
            self.centroids[i] = np.mean(self.data[self.predicted == i], axis=0)
        return np.sum(self.centroids != last_centroids), last_centroids

    def fit(self, X, y=None, centroids=None, distance='euclidean', stop_condition='iter', n_iter=1):
        self.data = X
        self.labels = y
        self.distance = distance
        self.predicted = np.zeros(self.data.shape[0])

        if centroids is None:
            self.centroids = self.data[np.random.randint(0, self.data.shape[0], self.k)]
        else:
            self.centroids = np.array(centroids)

        fit_start = time.time()

        self._predict()
        self.n_iterations = 0

        last_sse = np.inf

        loop = True
        while loop:
            centroids_updated, last_centroids = self._calculate_centroids()
            self._predict()
            self.n_iterations += 1
            sse = self.sse()
            loop = (stop_condition == 'iter' and self.n_iterations < n_iter) or (
                    stop_condition == 'sse' and sse < last_sse) or (
                    stop_condition == 'centroid' and centroids_updated > 0)
            last_sse = sse

        if stop_condition == 'sse':
            self.centroids = last_centroids
            self._predict()

        self.time = time.time() - fit_start

    def sse(self, distance=None):
        sse = 0
        for i, c in enumerate(self.centroids):
            for p in self.data[self.predicted == i]:
                sse += self._calculate_distance(p, c, distance) ** 2
        return sse

    def accuracy(self):
        predicted_labels = np.zeros_like(self.predicted)
        for i in range(self.k):
            predicted_labels[self.predicted == i] = mode(self.labels[self.predicted == i])[0][0]
        return np.mean(predicted_labels == self.labels)

    def plot_2d(self, title='', x_label='', y_label='', feature_x=0, feature_y=1):
        markers = ['o', '^', 's', 'v', '<', '>', 'P', 'x', '+']
        self.ifig += 1
        plt.figure(self.ifig)
        plt.title(title)
        for i in range(self.k):
            plt.scatter(self.data[self.predicted == i][:, feature_x], self.data[self.predicted == i][:, feature_y],
                        alpha=0.8, marker=markers[i % len(markers)], label='cluster #' + str(i))
        plt.scatter(self.centroids[:, feature_x], self.centroids[:, feature_y], alpha=1, marker='+',
                    color='black', label='centroid')
        for c in self.centroids:
            plt.text(c[feature_x] + 0.1, c[feature_y] + 0.1, '({:.3f}, {:.3f})'.format(c[feature_x], c[feature_y]))
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.legend()
        plt.show()
