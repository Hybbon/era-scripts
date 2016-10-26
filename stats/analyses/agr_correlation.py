"""Correlation test between agreement and hit rate"""

ARGPARSE_PARAMS = ("-R", "--agr_correlation",
                   "Correlation test between agreement and hit rate")
MODULE_NAME = "agr_correlation"

import numpy as np
import os
from collections import namedtuple
from .agreement import gen_agreement_matrix
from ..plotting import plot_scatter
from scipy.stats import pearsonr

TOLERANCE = 2

AgrCorrelation = namedtuple('AgrCorrelation', ["user_points", "alg_points"])

def generate(dsr, results, conf):
    res = {}
    tol = conf['tolerance'] if conf and 'tolerance' in conf else TOLERANCE

    for p, part in dsr.parts.items():
        algs, hits = part.algs, part.hits
        user_points = {}
        alg_points = []
        total_hits = 0
        total_agr = 0
        for alg, alg_res in algs.items():
            m, user_ids = gen_agreement_matrix(algs, hits, alg_res,
                                               dsr.len_ranking, tol)

            plot_points = []

            for u_i, user_id in enumerate(user_ids):
                user_hits = np.count_nonzero(m[u_i, :])
                plot_points.append((m[u_i, :].sum(), user_hits))
                total_hits += user_hits
                total_agr += m[u_i, :].sum()

            user_points[alg] = plot_points
            alg_points.append((total_agr, total_hits))

        res[p] = AgrCorrelation(user_points, alg_points)
    return res


def save_scatter_plot(path, points):
    agreements, hits = zip(*points)
    corr, p_val = pearsonr(hits, agreements)
    plot_scatter(path, "Pearson: {} (p-value = {})".format(corr, p_val),
                 hits, "Hit count", agreements, "Agreement")



def plot(res, dsr, output_dir, conf, ext='pdf'):
    for p, (user_points, alg_points) in res.items():
        p_dir = os.path.join(output_dir, p)
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)
        for alg, alg_res in user_points.items():
            path = os.path.join(p_dir, alg + "-scatter.pdf")
            save_scatter_plot(path, alg_res)

        path = os.path.join(p_dir, "alg_comparison.pdf")
        save_scatter_plot(path, alg_points)



