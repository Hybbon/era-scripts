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


def distance_matrix(algs, function, num_processes):
    """Generates pairwise distance matrix according to a distance function"""
    alg_index = {alg: i for i, alg in enumerate(sorted(algs.keys()))}
    distances = np.ndarray((len(algs), len(algs)), dtype=float)

    for alg1, alg2 in itertools.product(algs.keys(), repeat=2):
        logger.warn("Comparing {} to {} via {}".format(alg1, alg2,
                                                       function.__name__))
        rankings1, rankings2 = algs[alg1], algs[alg2]
        user_rankings = [(rankings1[user], rankings2[user])
                for user in rankings1.keys() & rankings2.keys()]
        with mp.Pool(num_processes) as pool:
            user_results = pool.map(function, user_rankings)
        mean = sum(user_results) / len(user_results)
        distances[alg_index[alg1], alg_index[alg2]] = mean

    return distances


def kendall(t):
    return stats.metrics.kendall(*t)

def footrule(t):
    return stats.metrics.footrule(*t)

DISTANCE_FUNCTIONS = [
    kendall,
    footrule
]


def distances(algs, num_processes):
    """Computes mean distances for an algorithm via defined functions"""
    logger.warn("Initiating distance frame calculations")
    means = pd.DataFrame()

    for f in DISTANCE_FUNCTIONS:
        logger.warn("Computing {} matrix".format(f.__name__))
        m = distance_matrix(algs, f, num_processes)
        # The mean value is not computed directly because the matrix contains
        # the distance between an algorithm and itself. We must subtract 1 from
        # the number of algorithms.
        means[f.__name__] = m.sum(axis=0) / (len(algs) - 1)

    alg_names = sorted([os.path.basename(path) for path in algs.keys()])
    means['alg_names'] = alg_names
    return means.set_index('alg_names')


def slice_rankings(d, length):
    return {user_id: items[:length] for user_id, items in d.items()}


def load_algs(files, length):
    logger.warn("Loading files {}".format(files))
    return {path: slice_rankings(rankings_dict(path), length)
            for path in files}


def save_distances(distances, path):
    logger.warn("Saving distances to {}".format(path))
    distances.to_csv(path)


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+",
                        help="ranking files to be compared.")
    parser.add_argument("-o", "--output", default="distances.csv",
                        help="output csv to be written. (default: %(default)s)")
    parser.add_argument("-p", "--processes", type=int, default=1,
                        help="number of processes for parallel execution. "
                        "(default: %(default)s)")
    parser.add_argument("-l", "--length", type=int, default=20,
                        help="length of the rankings to be considered")
    return parser.parse_args()


def main():
    args = parse_args()
    algs = load_algs(args.files, args.length)
    d = distances(algs, args.processes)
    save_distances(d, args.output)


if __name__ == "__main__":
    main()
