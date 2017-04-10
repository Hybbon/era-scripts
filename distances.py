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
import ipdb

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

#TODO continuar a modificacao para calcular distancis dos from_algs
def distance_matrix(algs, function, num_processes,users_to_use=[], from_algs=None):
    """Generates pairwise distance matrix according to a distance function
        Algorithms order in the distance matrix is alphabetically ordered

        We can use from_algs when we want to compute the distance between two 
        distinct sets of algorithms
"""
    ipdb.set_trace()
    alg_index = {alg: i for i, alg in enumerate(sorted(algs.keys()))}
    if from_algs:
        alg_index.update({alg: i for i, alg in enumerate(sorted(from_algs.keys()))})
        distances = np.ndarray((len(from_algs), len(algs)), dtype=float)
        algs_iterator = itertools.product(from_algs.keys(),algs.keys())
    else:
        distances = np.ndarray((len(algs), len(algs)), dtype=float)       
        algs_iterator = itertools.product(algs.keys(), repeat=2)

    ipdb.set_trace()
    for alg1, alg2 in algs_iterator:
        logger.warn("Comparing {} to {} via {}".format(alg1, alg2,
                                                       function.__name__))

        if from_algs:
            rankings1, rankings2 = from_algs[alg1], algs[alg2]
        else:        
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

    return distances


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


def distances(algs, num_processes,users_to_use=[]):
    """Computes mean distances for an algorithm via defined functions"""
    logger.warn("Initiating distance frame calculations")
    means = pd.DataFrame()

    for f in DISTANCE_FUNCTIONS:
        logger.warn("Computing {} matrix".format(f.__name__))
        m = distance_matrix(algs, f, num_processes,users_to_use)
        # The mean value is not computed directly because the matrix contains
        # the distance between an algorithm and itself. We must subtract 1 from
        # the number of algorithms.
        means[f.__name__] = m.sum(axis=0) / (len(algs) - 1)

    alg_names = sorted([os.path.basename(path) for path in algs.keys()])
    means['alg_names'] = alg_names
    return means.set_index('alg_names')


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
