"""Agreement within tolerance window matrices"""

ARGPARSE_PARAMS = ("-A", "--agreement",
                   "Agreement within tolerance window matrices")
MODULE_NAME = "agreement"

import numpy as np
import os
from collections import namedtuple
from ..plotting import plot_matrix_heatmap
from sklearn.cluster import KMeans

Agreement = namedtuple('Agreement', ['m', 'user_ids', 'centroids', 'labels'])
TOLERANCE = 2
DEF_N_CLUSTERS = 8


def within_tol(ranking, item_id, pos, tol):
    begin = 0 if pos - tol < 0 else pos - tol
    end = pos + tol + 1
    return item_id in ranking[begin:end]


def gen_agreement_matrix(algs, hits, alg_res, len_ranking, tol):
    # m[u_i][ranking_pos]
    m = np.zeros((len(alg_res.lists), len_ranking), dtype=int)
    user_ids = []
    for u_i, (user_id, ranking) in enumerate(alg_res.lists.items()):
        user_ids.append(user_id)
        if user_id in hits.lists:
            for pos, item_id in enumerate(ranking):
                if item_id in hits.lists[user_id]:
                    total = 0
                    for other_alg_res in algs.values():
                        total += within_tol(other_alg_res.lists[user_id],
                                            item_id, pos, tol)
                    m[u_i, pos] = total
    return m, user_ids


def cluster_counts(res):
    for p, algs in res.items():
        for alg, agmnt in algs.items():
            l = list(agmnt.labels)
            counts = {k: l.count(k) for k in set(l)}
            print("{0}/{1}: {2}".format(p, alg, str(counts)))


def generate(dsr, results, conf):
    res = {}
    tol = conf['tolerance'] if 'tolerance' in conf else TOLERANCE
    n_clusters = conf['n_clusters'] if 'n_clusters' in conf else DEF_N_CLUSTERS
    k_means = KMeans(n_clusters=n_clusters)
    # default: 8 clusters, 300 iterations max, 10 runs

    for p, part in dsr.parts.items():
        algs, hits = part.algs, part.hits
        res[p] = {}
        for alg, alg_res in algs.items():
            m, user_ids = gen_agreement_matrix(algs, hits, alg_res,
                                               dsr.len_ranking, tol)
            labels = k_means.fit_predict(m)
            centroids = k_means.cluster_centers_
            res[p][alg] = Agreement(m, user_ids, centroids, labels)
    return res


def plot(res, dsr, output_dir, conf, ext='pdf'):
    for p, algs in res.items():

        x_ticks = list(range(dsr.len_ranking))

        p_dir = os.path.join(output_dir, p)
        for alg, (m, user_ids, centroids, labels) in algs.items():
            alg_dir = os.path.join(p_dir, alg)
            if not os.path.exists(alg_dir):
                os.makedirs(alg_dir)

            cl_user_rows = [[] for c in centroids]
            cl_y_ticks = [[] for c in centroids]

            for u_i, l in enumerate(labels):
                cl_user_rows[l].append(m[u_i, :])
                cl_y_ticks[l].append(user_ids[u_i])

            cl_matrices = [np.array(rows_list) for rows_list in cl_user_rows]

            for i, (cl_m, y_ticks) in enumerate(zip(cl_matrices, cl_y_ticks)):
                figsize = (45, max(cl_m.shape[0] // 3, 2))
                path = os.path.join(alg_dir, "Cluster {0}.{1}".format(i, ext))
                plot_matrix_heatmap(path, "Concordância por posição por"
                                    " usuário", "Posição no ranking", x_ticks,
                                    "ID do usuário", y_ticks, cl_m, text=True,
                                    color_min=0, color_max=len(dsr.algs),
                                    rows_per_plot=len(y_ticks), cmap='RdPu',
                                    figsize=figsize)
