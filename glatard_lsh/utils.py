"""A series of various helper functions mostly for ease of use."""
from enum import Enum
import numpy as np


class dist_type(Enum):
    # __order__ lets you loop on all possible enum values.
    __order__ = 'SSD EUCLIDEAN COSINE'
    SSD = "SSD"
    EUCLIDEAN = "EUCLIDEAN"
    COSINE = "COSINE"


def unit_vector(v):
    return v / np.linalg.norm(v)


def euclidean_dist(v1, v2):
    return np.linalg.norm(v1 - v2)


def ssd_dist(v1, v2):
    return euclidean_dist(v1, v2)**2


def cosine_dist(v1, v2):
    return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def dist(v1, v2, dist):
    if dist == dist_type.EUCLIDEAN.value:
        return euclidean_dist(v1, v2)
    elif dist == dist_type.COSINE.value:
        return cosine_dist(v1, v2)
    if dist == dist_type.SSD.value:
        return ssd_dist(v1, v2)


def orthog_projection(v1, v2):
    # orthogonal projection of v1 onto a straight line parallel to v2
    return np.dot(v1, (v2 / np.linalg.norm(v2))) * v2


def generate_random_seeds(quant: int, seed=None):
    """Gives you a series of random seeds generated from one seed.
    Useful for generating many structures from one seed."""
    np.random.seed(seed)
    return np.random.randint(2**32 - 1, size=quant).tolist()


def muller_generate_vector(seed: int, dim: int):
    # The generation of random unit vectors from
    # 'A Note on a Method for Generating Points Uniformly
    # on N-Dimensional Spheres', by Muller et al., 1959

    np.random.seed(seed)
    deviates = np.random.normal(0, 1, size=[dim])
    radius = sum(deviates**2)**.5
    return (deviates / radius).T


def perturb_vector(v, euclidean_distance, maximum=None):
    """Perturbs a vector to a new vector with a particular euclidian
    distance to the original vector."""
    # TODO: fix this as it is bad right now
    if maximum is None:
        maximum = np.max(v)
    minimum = np.min(v)
    perturbed = v.copy()
    dist = 0
    while (dist < euclidean_distance - 2):
        dist = np.linalg.norm(v - perturbed)
        true_max = min(round(euclidean_distance - dist), maximum)
        i = np.random.randint(len(v))
        rand_val = np.random.randint(0, true_max + 1)
        sign = -np.sign(v[i])
        perturbed[i] += sign * rand_val

        if perturbed[i] < minimum:
            perturbed[i] = minimum
        elif perturbed[i] > maximum:
            perturbed[i] = maximum
        if dist > euclidean_distance:
            # if pertubed too much, restart.
            # Maybe this is bad for really big vectors, find better way of
            # handling this.
            perturbed = v.copy()

    return perturbed


def random_rot(dim: int, seed: int, dtype='d'):
    # https://github.com/mdp-toolkit/mdp-toolkit/blob/master/mdp/utils/routines.py
    # Code adapted from MDP-toolkit's random_rot(), but using numpy functions
    # instead.

    # The algorithm is described in the paper
    # Stewart, G.W., "The efficient generation of random orthogonal
    # matrices with an application to condition estimators", SIAM Journal
    # on Numerical Analysis, 17(3), pp. 403-409, 1980.
    # For more information see
    # http://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
    # TODO: make this faster. FFHT?
    np.random.seed(seed)

    H = np.identity(dim)
    D = np.ones(dim)
    for n in range(1, dim):
        x = np.random.normal(size=(dim - n + 1, )).astype(dtype)
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        Hx = (np.identity(dim - n + 1, dtype=dtype) - 2. * np.outer(x, x) /
              (x * x).sum())
        mat = np.identity(dim, dtype=dtype)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    D[-1] = (-1)**(1 - dim % 2) * D.prod()
    H = (D * H.T).T
    return H
