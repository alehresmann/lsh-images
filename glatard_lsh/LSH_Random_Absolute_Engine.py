"""Naive hyperplane LSH_Engine.
Can do hashing for both cosine dist and euclidean dist.
If you set segment_size to 0, it does binary hash,
aka with regards to cosine dist"""
import numpy as np
from glatard_lsh.LSH_Engine_Facade import LSH_Engine_Facade, engine_type
from glatard_lsh.utils import dist_type
from glatard_lsh import utils
# (additional) config contents:
# segment_size


class LSH_Random_Absolute_Engine(LSH_Engine_Facade):
    def _init_config(self):
        LSH_Engine_Facade._init_config(self)
        self.config["engine_type"] = engine_type.RANDOM_ABSOLUTE.value
        self.config["per_group"] = 10
        self.config["groups"] = [[0]]
        self.config["dist_type"] = dist_type.EUCLIDEAN.value

    def __init__(self, config=None):
        LSH_Engine_Facade.__init__(self, config)
        if config is None:
            self._init_config()

    def verify_config(self, config=None):
        if config is None:
            config = self.config
        assert config["per_group"] > 0

    def generate_struct(self, q, dist, k):
        """Segment size indicates the size by which each hyperplane
        normal will be segmented. Smaller sizes means smaller buckets.
        See chapter 3 of the book Mining of Massive Datasets for more
        information."""
        LSH_Engine_Facade.generate_struct(self, q, dist, k)
        if self.config["seeds"] is None:
            self.config["seeds"] = utils.generate_random_seeds(k)
        self.config["segment_size"] = dist
        for i in range(k):
            np.random.seed(self.config["seeds"][i])
            self.config["groups"].append(
                np.random.randint(0, len(q), self.config["per_group"]))

        self.config["bucket_query"] = self.actual_hash(q)

    def actual_hash(self, v):
        abs_avg = []
        for g in self.config["groups"]:
            abs_avg.append(sum([v[i] for i in g]) / len(g))
        return abs_avg

    def hash(self, v):
        h = np.array(self.actual_hash(v))
        q = np.array(self.config["bucket_query"])
        if (utils.euclidean_dist(h, q) <= self.config["dist"] /
            (len(v) / len(h))):
            return self.config["bucket_query"]
        else:
            return "no"
