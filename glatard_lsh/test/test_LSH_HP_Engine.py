import os
from ..LSH_HP_Engine import LSH_HP_Engine
# from ..utils import *


def test_init_config_should_pass():
    e = LSH_HP_Engine()
    e.verify_config()


def test_save_and_load_settings_in_file_should_pass():
    e = LSH_HP_Engine()
    f = LSH_HP_Engine()

    e.verify_config()
    e.save_config("test_config.toml")

    f.set_config("test_config.toml")
    assert e.config == f.config

    os.remove('test_config.toml')

    # def test_verify_quality_of_naive_segment_hash():
    #    dim = 1000
    #    function_quant = 100 # aka k, the number of hyperplanes in this family
    #    vector_quant_to_test = 1000
    #    distance = 100
    #    segment_size = distance
    #    e = hyperplane_engine(dim, segment_size=segment_size)
    #    e.seeds = utils.generate_random_seeds(function_quant)
    #
    #    print('\nTesting naive hyperplanes, segment hash:\ndim:', dim, \
    #            '\neuclidean distance tested:', distance, '\nsegment_size:', \
    #            segment_size, '\nh:', function_quant)
    #    test_engine.lsh_quality_tester(e, dim, function_quant, vector_quant_to_test, distance)
    #
    #
    # def test_verify_quality_of_naive_binary_hash():
    #    # this is just an example, but obviously using binary hash should get terrible results.
    #    dim = 1000
    #    function_quant = 100
    #    vector_quant_to_test = 1000
    #    distance = 100
    #
    #    e = hyperplane_engine(dim)
    #    e.seeds = utils.generate_random_seeds(function_quant)
    #    print('\nTesting naive hyperplanes, binary hash:\ndim:', dim, '\nh = ', function_quant)
    #    test_engine.lsh_quality_tester(e, dim, function_quant, vector_quant_to_test, distance)
