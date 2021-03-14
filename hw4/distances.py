#
# Code adapted from:
# https://dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
#
import math


def euclidean_distance(x, y):
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


def square_rooted(x):
    return round(math.sqrt(sum([a * a for a in x])), 5)


def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 5)


def cosine_distance(x, y):
    return 1 - cosine_similarity(x, y)


def generalized_jaccard_similarity(x, y):
    numerator = sum(min(a, b) for a, b in zip(x, y))
    denominator = sum(max(a, b) for a, b in zip(x, y))
    return round(numerator / float(denominator), 5)


def generalized_jaccard_distance(x, y):
    return 1 - generalized_jaccard_similarity(x, y)
