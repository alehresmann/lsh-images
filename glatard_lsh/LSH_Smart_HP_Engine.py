import math

from glatard_lsh.LSH_Engine_Facade import LSH_Engine_Facade, engine_type
from glatard_lsh.LSH_HP_Engine import LSH_HP_Engine
from glatard_lsh.utils import dist_type
from glatard_lsh import utils


class LSH_Smart_HP_Engine(LSH_HP_Engine):
    # helper class for finding good hyperplanes.

    def _init_config(self):
        LSH_HP_Engine._init_config(self)
        self.config["engine_type"] = engine_type.HP.value
        self.config["bucket_scalars"] = None
        self.config["dist_type"] = dist_type.EUCLIDEAN.value

    def __init__(self, config=None):
        LSH_HP_Engine.__init__(self, config)
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
        if self.config["seeds"] is None:
            self.config["seeds"] = utils.generate_random_seeds(k)
        LSH_Engine_Facade.generate_struct(self, q, dist, k)
        self.config["bucket_scalars"] = self.generate_bucket_scalars(
            q, self.config["seeds"], dist)

        self.config["bucket_query"] = self.hash(q)

    def generate_bucket_scalars(self, q, seeds, dist):
        bucket_scalars = []
        for seed in seeds:
            hp = utils.muller_generate_vector(seed, self.config["dim"])

            x = utils.orthog_projection(
                (hp * math.sqrt(dist)) + q, hp)[0] / hp[0]
            y = utils.orthog_projection(
                (hp * -math.sqrt(dist)) + q, hp)[0] / hp[0]
            if x > y:
                x, y = y, x
            bucket_scalars.append([x, y])
        return bucket_scalars

    def _compare_scalar(self, scalar, i):
        if scalar >= self.config["bucket_scalars"][i][
                0] and scalar <= self.config["bucket_scalars"][i][1]:
            return '1'
        return '0'

    def hash(self, v):
        scalars = self._project_onto_hyperplanes(v)
        h = ''.join([
            self._compare_scalar(scalars[i], i)
            for i in range(len(self.config["seeds"]))
        ])
        return h
