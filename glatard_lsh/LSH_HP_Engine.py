"""Naive hyperplane LSH_Engine.
Can do hashing for both cosine dist and euclidean dist.
If you set segment_size to 0, it does binary hash,
aka with regards to cosine dist"""
import math

from glatard_lsh.LSH_Engine_Facade import LSH_Engine_Facade, engine_type
from glatard_lsh.utils import dist_type
from glatard_lsh import utils
# (additional) config contents:
# segment_size


class LSH_HP_Engine(LSH_Engine_Facade):
    def _init_config(self):
        LSH_Engine_Facade._init_config(self)
        self.config["engine_type"] = engine_type.HP.value
        self.config["segment_size"] = 10
        self.config["dist_type"] = dist_type.EUCLIDEAN.value

    def __init__(self, config=None):
        LSH_Engine_Facade.__init__(self, config)
        if config is None:
            self._init_config()

    def verify_config(self, config=None):
        if config is None:
            config = self.config
        assert ((config["segment_size"] == 0
                 and config["dist_type"] == dist_type.COSINE.value)
                or (config["segment_size"] > 0
                    and config["dist_type"] == dist_type.EUCLIDEAN.value))

    def generate_struct(self, q, dist, k):
        """Segment size indicates the size by which each hyperplane
        normal will be segmented. Smaller sizes means smaller buckets.
        See chapter 3 of the book Mining of Massive Datasets for more
        information."""
        LSH_Engine_Facade.generate_struct(self, q, dist, k)
        self.config["segment_size"] = dist
        if self.config["seeds"] is None:
            self.config["seeds"] = utils.generate_random_seeds(k)
        self.config["bucket_query"] = self.hash(q)

    def _project_onto_hyperplanes(self, v):
        # returns the scalar of multiplication by each hyperplane needs to be
        # multipled to obtain v"s projection onto them
        seeds = self.config["seeds"]
        scalars = [0] * len(seeds)
        for i, seed in enumerate(seeds):
            hp = utils.muller_generate_vector(seed, self.config["dim"])
            # extract the scalar of the projection of v onto hp
            scalars[i] = utils.orthog_projection(v, hp)[0] / hp[0]
        return scalars

    def binary_hash(self, v):
        # for cosine distance
        return "".join([
            str(int(scalar >= 0))
            for scalar in self._project_onto_hyperplanes(v)
        ])

    def segment_hash(self, v):
        # for euclidean distance
        assert isinstance(self.config["segment_size"], int) and \
            self.config["segment_size"] >= 0, "LSH HP engine tried hashing \
                        , but no segment size has been set."

        return "".join([
            str(math.floor(scalar / self.config["segment_size"]))
            for scalar in self._project_onto_hyperplanes(v)
        ])

    def hash(self, v):
        if len(v) != self.config["dim"]:
            raise ValueError(
                "You must use hyperplanes whose normals are of the" +
                " same dimension as the vector given!")

        if self.config["dist_type"] == dist_type.COSINE.value:
            return self.binary_hash(v)
        elif self.config["dist_type"] == dist_type.EUCLIDEAN.value:
            return self.segment_hash(v)
