""" Interface for an lsh engine, aka an LSH function family. """
import sys
import datetime
import warnings
import toml
import numpy as np
from enum import Enum
from glatard_lsh.utils import dist_type
# TODO: potentially switch from config to just attributes on the class

# config contents:
# engine_type
# dist_type
# dimension (dim)
# seeds
# python_version
# np_version

# after generating the struct:
# dist -> expected radius of sphere around q which some bucket
# should try and approximate
# bucket_query -> which bucket q fell into


class engine_type(Enum):
    # __order__ lets you loop on all possible enum values.
    # eg for val in enum, do...
    __order__ = "FACADE HP SMART_HP CP RANDOM_ABSOLUTE RANDOM_DIFF"
    FACADE = "FACADE"
    HP = "HP"
    SMART_HP = "SMART_HP"
    CP = "CP"
    RANDOM_ABSOLUTE = "RANDOM_ABSOLUTE"
    RANDOM_DIFF = "RANDOM_DIFF"


class LSH_Engine_Facade:
    """Interface for an lsh engine, aka an LSH function family."""

    def _init_config(self):
        """Initialises config such that it can successfully hash, with some
        default values"""
        self.config = dict()
        self.config["dim"] = 10
        self.config["dist_type"] = dist_type.EUCLIDEAN.value
        self.config["engine_type"] = engine_type.FACADE.value
        self.config["seeds"] = [1, 2, 3, 4, 5]
        self.config["python_version"] = str(sys.version_info[:2])
        self.config["np_version"] = str(np.__version__)

    def __init__(self, config=None):
        """Initialises engine, with a dim, some seeds and a config if wanted.
        If seeds aren't given, they'll be randomly generated."""
        if config is None:
            self._init_config()
        else:
            self.set_config(config)

    def __str__(self):
        return str(self.get_config())

    def generate_struct(self, q, dist: int, k):
        """Configure a structure with:
        q: Query (centre of bucket),
        dist: Radius of bucket around q,
        k: Number of hash functions."""
        self.config["bucket_query"] = ""
        self.config["dist"] = dist

    def hash(self, v):
        """Hashes the given point v, assuming the structure has been built."""
        return ""

    def estimate_distance(self, h):
        """Based on the obtained hash, guesses the distance between the
        point who has that hash, and the query point."""

    def get_config(self):
        "Returns current configuration as dictionary"
        assert self.config is not None, "Engine wasn't configured properly!"
        return self.config

    def set_config(self, config):
        """Sets the configuration dictionary either using a file, either
        using a dict."""
        if isinstance(config, str):
            import os
            print(os.getcwd())
            with open(config) as f:
                config = toml.loads(f.read())
            f.close()
        if isinstance(config, dict):
            self.config = config
        else:
            assert False, "config given wasn't a filename or a dict!"

    def verify_config(self, config=None):
        """Verifies that the config matches the current environment settings.
        In this interface, only the python and numpy versions are checked.
        This should only be run after setup."""
        if config is None:
            config = self.config

        assert config["dim"] > 1, "Engine dim must be greater than 1"

        # Given that we use string values in toml file, need to verify they
        # are valid enum vals
        valid_dim = False
        valid_eng = False
        for e in dist_type:
            valid_dim = True if (
                self.config["dist_type"] == e.value) else valid_dim
        for e in engine_type:
            valid_eng = True if (
                self.config["engine_type"] == e.value) else valid_dim
        assert (valid_dim and valid_eng)

        assert len(config["seeds"]) > 0
        assert all(
            isinstance(item, int)
            for item in config["seeds"]), "List of seeds in Engine invalid"

        assert isinstance(self.config["bucket_query"], str)

        if config["python_version"] != str(sys.version_info[:2]):
            warnings.warn("Warning: This LSH struct was generated in python " +
                          config["python_version"] + " whereas you're on " +
                          str(sys.version_info[:2]) + ", you probably want"
                          "to verify that nothing has changed that may affect"
                          " the random generator.")

        if config["np_version"] != str(np.__version__):
            warnings.warn(
                "Warning: This LSH struct was generated with numpy " +
                config["np_version"] + " whereas you're on " + np.__version__ +
                ", you probably want to verify that "
                "nothing has changed that may affect the random generator.")

    def save_config(self, filename=None):
        if filename is None:
            filename = ("lsh_config_" + str(datetime.datetime.now()) + ".txt")
        with open(filename, 'w') as f:
            toml.dump(self.get_config(), f)
        f.close()
