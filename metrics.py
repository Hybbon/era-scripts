#!/usr/bin/env python3

"""
metrics.py
==========
Recommender metrics calculator
------------------------------

This script is used for generating metrics for recommender systems'
results for a dataset. Currently, this script generates:

- NDCG
- MAP

It does it for the different ranking lengths specified in the config
files (which can be overridden if necessary).

This script runs according to RankingStats config files. (Read more
about them in the README file at the top directory of this repo.)

Examples::

    python metrics.py bases/ml100k/

    python metrics.py bases/ml100k/ -c override.json

"""

import stats
import os.path
import argparse
import statx.aux
import numpy as np
from stats.metrics import gen_metrics
import scipy.stats as st


def save_single_table(output_dir, basename, matrix, headers):
    np.savetxt(
        os.path.join(output_dir, basename + ".csv"),
        matrix,
        fmt="%.4f",
        delimiter=";",
        newline="\n",
        comments="# ",
        header=";".join(headers)
    )


def save_tables(res, algs_labels, output_dir, num_clusters=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for metric_name, lengths in res.items():
        for l, matrix in lengths.items():
            basename = "{0}@{1}".format(metric_name, l)
            if num_clusters:
                for cl_i in range(num_clusters):
                    cl_dir = os.path.join(output_dir, str(cl_i))
                    if not os.path.exists(cl_dir):
                        os.makedirs(cl_dir)
                    save_single_table(cl_dir, basename,
                                      matrix[:, :, cl_i], algs_labels)
            else:
                save_single_table(output_dir, basename, matrix, algs_labels)


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="folder containing the config file and the"
                        " ranking data to be evaluated.")
    parser.add_argument("-o", "--output", type=str, default="csv",
                        help="folder where the results' csvs should be saved, "
                        "relative to data folder. (default: %(default)s)")
    parser.add_argument("-c", "--config", type=str, default="",
                        help="name of the config file. (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    conf = aux.load_configs(aux.CONF_DEFAULT, os.path.join(args.data,
                            aux.BASE_CONF), args.config)

    dsr = stats.DataSetResults(conf, args.data)

    lengths = (1, 3, 5, 10, 100)
    res = gen_metrics(dsr, conf['parts'], conf['algs'], lengths)

    output_dir = os.path.join(args.data, args.output)
    save_tables(res, conf['algs'], output_dir)


if __name__ == "__main__":
    main()
