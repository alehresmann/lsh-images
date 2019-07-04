import numpy as np
import random
from glatard_lsh import utils
from glatard_lsh.LSH_Smart_HP_Engine import LSH_Smart_HP_Engine
from glatard_lsh.LSH_Random_Absolute_Engine import LSH_Random_Absolute_Engine
from glatard_lsh.LSH_Random_Diff_Engine import LSH_Random_Diff_Engine
# for graphing of vector acceptance / rejection
try:
    import matplotlib.pyplot as plt
    graphing = True

except ImportError:
    print("Couldn't find plotly, not making visual graphs. If you want them, "
          "install the plotly package with: pip install --user plotly")
    graphing = False

# def test_on_actual_image():
#    hash_accuracy(100, 100, "smart_hyperplane", 10, 5, 1000)


def verify_hash_correctness(engine, point):
    pass


def _test_engines(dim, dist, engines, k, l, candidates, q, max_val):
    for e in engines:
        if engines[0].config["dist_type"] != e.config["dist_type"]:
            raise ValueError(
                "Misconfigured tests: cannot test engines on different \
                distance types!")

    if q is None:
        print("Generating a query point...")
        q = np.random.randint(-max_val, max_val, size=dim)

    for e in engines:
        e.config["dim"] = dim
        e.generate_struct(q, dist, k)

    if candidates is None:
        candidates = 1000

    if isinstance(candidates, int):
        print("Generating", 2 * candidates, "candidates...")
        print(candidates, "vectors with euclidean distances of:", 0, "to",
              dist)
        under = _generate_vectors_helper(candidates, q, 0, dist)
        print(candidates, "vectors with euclidean distances of:", dist + 1,
              "to", 10 * dist)
        over = _generate_vectors_helper(candidates, q, dist + 1, 10 * dist)
        candidates = under + over
    dists = [
        utils.dist(c, q, engines[0].config["dist_type"]) for c in candidates
    ]
    variances = [np.var(c) for c in candidates]
    print("done. Hashing for each engine.")
    for e in engines:
        print("hashing for", e.config["engine_type"])
        hashes = [e.hash(c) for c in candidates]
        correct = [
            True if
            ((h == e.config["bucket_query"] and d <= e.config["dist"]) or
             (h != e.config["bucket_query"] and d > e.config["dist"])) else
            False for h, d in zip(hashes, dists)
        ]
        if graphing:
            x = list(range(len(correct)))
            colour = ['green' if c else 'red' for c in correct]
            plt.scatter(x, dists, c=colour)
            plt.xlabel("vectors")
            plt.ylabel("dist")
            plt.show()
            plt.scatter(x, variances, c=colour)
            plt.xlabel("vectors")
            plt.ylabel("variance")
            plt.show()
        else:
            print("No graphing package, could not graph.")


def _generate_vectors_helper(candidate_quant, q, min_dist, max_dist):
    vs = []
    for i in range(candidate_quant):
        random_dist = random.randint(min_dist, max_dist)
        vs.append(utils.perturb_vector(q, random_dist))
    return vs


def test_engines():
    e1 = LSH_Random_Absolute_Engine()
    e1.config["per_group"] = 20
    e1.config["seeds"] = utils.generate_random_seeds(100)
    e2 = LSH_Random_Absolute_Engine()
    e2.config["per_group"] = 100
    e2.config["seeds"] = utils.generate_random_seeds(100)
    e3 = LSH_Random_Absolute_Engine()
    e3.config["per_group"] = 500
    e3.config["seeds"] = utils.generate_random_seeds(100)

    e1 = LSH_Random_Diff_Engine()
    e1.config["per_group"] = 20
    e1.config["seeds"] = utils.generate_random_seeds(100)
    e2 = LSH_Random_Diff_Engine()
    e2.config["per_group"] = 100
    e2.config["seeds"] = utils.generate_random_seeds(100)
    e3 = LSH_Random_Diff_Engine()
    e3.config["per_group"] = 5000
    e3.config["seeds"] = utils.generate_random_seeds(100)

    _test_engines(320**3, 100, [e1, e2, e3], 10, 10, 100, None, 100)
