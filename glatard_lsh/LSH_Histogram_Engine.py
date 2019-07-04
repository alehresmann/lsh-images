# flake8: noqa E129
# noqa: E129
# pylint: disable=C0103, C0330
import numpy as np
import math
from glatard_lsh import utils
from glatard_lsh.LSH_Engine_Facade import LSH_Engine_Facade
from glatard_lsh.LSH_Random_Absolute_Engine import LSH_Random_Absolute_Engine


class LSH_Histogram_Engine(LSH_Random_Absolute_Engine):
    def self.init_config(self):
        LSH_Random_Absolute_Engine.init_config()
        self.config["histograms"] = []  # array of arrays
        self.config["bounds"] = []

    def generate_struct(self, q, dist, k):
        # forging the bounds
        min_q = min(q)
        max_q = max(q)
        bound_dist = math.floor((max_q - min_q) / self.config["per_group"])
        for g in k:
            self.config["bounds"].append(bound_dist)
            bound_dist += bound_dist
        self.config["bounds"][-1] += (max_q - min_q) % self.config["per_group"]

        LSH_Random_Absolute_Engine.generate_struct(self, q, dist, k)

    def place_in_histogram(p):
        # aka binary search but for a "bucket"
        lower = 0
        higher = len(self.config["bounds"])
        while higher > lower + 1:
            mid = math.floor((higher + lower)/2)
            if p < self.config["bounds"][mid]:
                lower = mid
            else:
                higher = mid
        assert higher = lower + 1
        return lower  # which bucket p is in, in the histogram

    def actual_hash(self, v):
        histograms = []
        for group in self.config["groups"]:  # for each histogram
            values = [v[i] for i in group]
            dist = (max(values) - min(values)) / self.config["per_group"]
            histogram = [0] * self.config["per_group"]
            for i in self.config["per_group"]:  # for each point in each group
                histogram[self.place_in_histogram(value)] += 1
            histograms.append(histogram)
        return histograms

    def hash(self, v):
        h = np.array(self.actual_hash(v))
        q = np.array(self.config["bucket_query"])
        dist = [utils.euclidean_dist(h_histo, q_histo)
                for h_histo, q_histo in zip(h, q)]
        dist = sum(dist) / len(dist)
        if dist < self.config["dist"]:
            return self.config["bucket_query"]
        return "no"
