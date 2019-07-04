import os
import warnings
import numpy as np
import sys

from ..LSH_Engine_Facade import LSH_Engine_Facade, engine_type, dist_type


def test_save_and_load_settings_in_file_should_pass():
    e = LSH_Engine_Facade()
    f = LSH_Engine_Facade()

    e.generate_struct(None, None, None)
    e.verify_config()
    e.save_config("test_config.toml")

    f.set_config("test_config.toml")
    assert e.config == f.config

    os.remove('test_config.toml')


def test_verify_config_should_pass():
    e = LSH_Engine_Facade(
        "glatard_lsh/test/test_resources/verify_config_config.toml")
    try:
        e.verify_config()
    except AssertionError:
        e.config["dim"] = 2
    try:
        e.verify_config()
    except AssertionError:
        e.config["engine_type"] = engine_type.FACADE.value
        e.config["dist_type"] = dist_type.EUCLIDEAN.value
    try:
        e.verify_config()
    except AssertionError:
        e.config["seeds"] = [1, "a"]
    try:
        e.verify_config()
    except AssertionError:
        e.config["seeds"] = [1, 2]
    try:
        e.verify_config()
    except KeyError:
        e.config["bucket_query"] = "mmm"

    # turn on ability to treat warnings as errors
    warnings.filterwarnings("error")
    try:
        e.verify_config()
    except Warning:
        e.config["python_version"] = str(sys.version_info[:2])
        e.config["np_version"] = str(np.__version__)
    e.verify_config()


# def lsh_quality_tester(e,
#                       dim,
#                       function_quantity,
#                       vec_quantity,
#                       max_dist,
#                       q=None):
#    # This shouldn't be used by the interface but by implementations of the engine.
#    max_val = 100
#
#    if q is None:
#        q = np.random.randint(max_val, size=dim)
#    # random vectors
#    rando_vs = np.random.randint(max_val, size=(vec_quantity, dim))
#    # vectors purposely close to but smaller than given euclidean distance
#    small_vs = []
#    for i in range(vec_quantity):
#        random_dist = random.randint(max_dist - round(max_dist / 100) - 5,
#                                     max_dist)
#        small_vs.append(utils.perturb_vector(q, random_dist))
#
#    # vectors purposely close to but larger than given euclidean distance
#    large_vs = []
#    for i in range(vec_quantity):
#        random_dist = random.randint(max_dist + 1,
#                                     max_dist + round(max_dist / 100) + 5)
#        large_vs.append(utils.perturb_vector(q, random_dist))
#    print('max dist is:', max_dist)
#    print('testing random vectors')
#    _helper_lsh_qt(e, q, rando_vs, max_dist)
#    print('testing vectors with euclidean distances of:',
#          max_dist - round(max_dist / 100) - 5, 'to', max_dist)
#    _helper_lsh_qt(e, q, small_vs, max_dist)
#    print('testing vectors with euclidean distances of:', max_dist + 1, 'to',
#          max_dist + round(max_dist / 100) + 5)
#    _helper_lsh_qt(e, q, large_vs, max_dist)
#
#
# def _helper_lsh_qt(e, q, vectors, max_dist):
#    # helper func for testing hash functions on specific vectors
#    q_hash = e.hash(q)
#    false_negatives = 0
#    false_positives = 0
#    for v in vectors:
#        v_hash = e.hash(v)
#        actual_dist = utils.euclidean_dist(v, q)
#        #print(actual_dist, '\t', v_hash)
#        if v_hash != q_hash and actual_dist < max_dist:
#            false_negatives += 1
#        if v_hash == q_hash and actual_dist > max_dist:
#            false_positives += 1
#
#    print('Out of', len(vectors), 'reported', false_negatives,
#          '\tfalse negatives and', false_positives, '\tfalse positives.')
