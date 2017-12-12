#!/usr/bin/env python3

"""
distances.py
============
Calculator for distances between rankings
-----------------------------------------

This script is used for calculating the distances between a
user-provided set of MML-formatted ranking files. Currently, by
default, only Kendall's Tau and Spearman's Footrule are calculated,
but adding new rankings functions is very simple, and consists of
adding them to the DISTANCE_FUNCTIONS list. They must have the
following signature::

    f(rankings: iterable) -> distance: double
    len(rankings) == 2
    0 <= distance <= 1

Calling the script itself is straight-forward. It takes an unlimited
amount (at least two) of positional arguments, which are paths to the
ranking files. By default, the results are output to "distances.csv".
This can be changed via ``-o <path>``. Other useful parameters can be
found through the "-h" flag. Here are some examples::

    python distances.py my_rankings/*.out --length 8 --processes 4
    python distances.py r1.out r2.out -o output_file.csv
"""

import collections
import itertools
import pandas as pd
import scipy.stats
import sys
import argparse
import numpy as np
import logging
import os.path
import multiprocessing as mp
import stats.metrics
from stats.file_input import rankings_dict

logger = logging.getLogger()


'''
This function returns a dictionary of dataframes. Each dataframe represents 
an algorithm and contains the distances, for each user, of the rankings 
recommend by this algorithm to the rankings recommended by the other algorithms.


The returned structure can be seen as an 3-dimensional triangular matrix. 
Therefore, if for a pair of algorithms (A,B) the distances can be stored in 
distances[A][B] or distances[B][A]
'''
def distance_matrix_users(algs, function,algs_to_compare = [], num_processes=1):
    """Generates pairwise distance matrix according to a distance function"""
    index_users = algs[list(algs.keys())[0]].keys()
    distances = {}

    if len(algs_to_compare) == 0:
        algs_names = sorted(algs.keys())
    else:
        algs_names = sorted(algs_to_compare)

    for i,alg1 in enumerate(algs_names):
        #contructing the triangular matrix
        distances[alg1] = pd.DataFrame(index=index_users,columns=algs_names[i+2:])
        for j in range((i+1),len(algs_names)):
            alg2 = algs_names[j]
            logger.warn("Comparing {} to {} via {}".format(alg1, alg2,
                                                           function.__name__))
            rankings1, rankings2 = algs[alg1], algs[alg2]
            user_rankings = [(rankings1[user], rankings2[user])
                    for user in rankings1.keys() & rankings2.keys()]        
            with mp.Pool(num_processes) as pool:
                user_results = pool.map(function, user_rankings)

            distances[alg1][alg2] = pd.Series(user_results, index = index_users)
                 
    return distances


def distance_matrix(algs, function, num_processes,users_to_use=[]):
    """Generates pairwise distance matrix according to a distance function"""
    alg_index = collections.OrderedDict((alg, i) for i, alg in enumerate(sorted(algs.keys())))
    distances = np.ndarray((len(algs), len(algs)), dtype=float)

    for alg1, alg2 in itertools.product(algs.keys(), repeat=2):
        logger.warn("Comparing {} to {} via {}".format(alg1, alg2,
                                                       function.__name__))
        rankings1, rankings2 = algs[alg1], algs[alg2]
        #checking if we will use the complete set of users of just a sub set of them
        if len(users_to_use) == 0:
            user_rankings = [(rankings1[user], rankings2[user])
                    for user in rankings1.keys() & rankings2.keys()]
        else:
            user_rankings = [(rankings1[user], rankings2[user])
                    for user in users_to_use]

        with mp.Pool(num_processes) as pool:
            user_results = pool.map(function, user_rankings)
        mean = sum(user_results) / len(user_results)
        distances[alg_index[alg1], alg_index[alg2]] = mean

    alg_names = list(alg_index.keys())
    return pd.DataFrame(distances, index=alg_names, columns=alg_names)


def kendall_samuel(t):
    return stats.metrics.kendall_samuel(*t,penalty=1)

def kendall(t):
    return stats.metrics.kendall(*t)

def footrule(t):
    return stats.metrics.footrule(*t)

DISTANCE_FUNCTIONS = [
    kendall,
    kendall_samuel,
    footrule
]

FUNCTION_BY_NAME = {f.__name__: f for f in DISTANCE_FUNCTIONS}

def compute_distance_frames(algs, functions, num_processes, users_to_use=[]):
    """Computes distance matrix dataframes for a list of algorithms."""
    logger.warn("Initiating distance matrix calculations")

    frames = {}

    for f in functions:
        function_name = f.__name__
        logger.warn("Computing {} matrix".format(function_name))
        m = distance_matrix(algs, f, num_processes,users_to_use)
        frames[function_name] = m

    return frames


def mean_distances(frames, algs):
    """Computes mean distances from computed distance matrices."""
    logger.warn("Computing mean distances from matrices.")
    means = pd.DataFrame()

    for name, f in frames.items():
        # The mean value is not computed directly because the matrix contains
        # the distance between an algorithm and itself. We must subtract 1 from
        # the number of algorithms.
        means[name] = f.sum(axis=0) / (f.shape[0] - 1)

    return means


def slice_rankings(d, length):
    return {user_id: items[:length] for user_id, items in d.items()}



'''returns the algorithm name given its path
'''
def get_name_from_path(fstr):
    #re_str = 'u[1-5]-[a-z A-Z]*\\.out'
    #re_name = re.compile(re_str)
    #alg_name = re_name.match(fstr).group(0)

    return fstr.split('/')[-1]


def load_algs(files, length):
    logger.warn("Loading files {}".format(files))
    return {get_name_from_path(path): slice_rankings(rankings_dict(path), length)
            for path in files}


def save_distances(distances, path):
    logger.warn("Saving distances to {}".format(path))
    distances.to_csv(path)


def save_matrix_frames(frames, output_dir):
    logger.warn('Saving distance matrices to %s', output_dir)

    for name, f in frames.items():
        path = os.path.join(output_dir, name + '.csv')
        logger.warn('Saving matrix for %s to %s', name, path)
        f.to_csv(path)


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+",
                        help="ranking files to be compared.")
    parser.add_argument("-o", "--output", default="distances.csv",
                        help="output csv to be written. (default: %(default)s)")
    parser.add_argument("-d", "--output_dir", default=".",
                        help="directory to which outputs should be written. "
                        "(default: %(default)s)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="number of processes for parallel execution. "
                        "(default: %(default)s)")
    parser.add_argument("-l", "--length", type=int, default=20,
                        help="length of the rankings to be considered")
    parser.add_argument('-f','--function', action='append', help='distance functions to be computed', choices=FUNCTION_BY_NAME.keys(), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    algs = load_algs(args.files, args.length)

    functions = [FUNCTION_BY_NAME[name] for name in args.function]
    frames = compute_distance_frames(algs, functions, args.processes)
    d = mean_distances(frames, algs)

    if not os.path.exists(args.output_dir):
        logger.warn('Creating directory %s', args.output_dir)
        os.makedirs(args.output_dir)

    save_distances(d, os.path.join(args.output_dir, args.output))
    save_matrix_frames(frames, args.output_dir)


if __name__ == "__main__":
    main()
