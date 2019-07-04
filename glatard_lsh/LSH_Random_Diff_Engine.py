# pylint: disable=C0103, C0330
import numpy as np
from glatard_lsh import utils
from glatard_lsh.LSH_Random_Absolute_Engine import LSH_Random_Absolute_Engine


class LSH_Random_Diff_Engine(LSH_Random_Absolute_Engine):
    def actual_hash(self, v):
        ave_diff = []
        for g in self.config["groups"]:
            ave_diff.append(np.var([v[i] for i in g]))
        return ave_diff

    def hash(self, v):
        h = np.array(self.actual_hash(v))
        q = np.array(self.config["bucket_query"])
        if (utils.euclidean_dist(h, q) <= 200 * self.config["dist"] /
            (len(v) / len(h))):
            return self.config["bucket_query"]
        return "no"
