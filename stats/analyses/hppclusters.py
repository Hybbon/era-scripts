import os
from ..plotting import plot_histogram
import numpy as np

def hits_per_position(alg_res, hits, items_per_ranking, user_list):
    """Returns a list with the hits per position in a AlgResults object.

    alg_res -- AlgResults object to be evaluated.
    hits -- AlgResults object containing the hits for each user from the test
            base.
    items_per_ranking -- number of items in each user's ranking."""
    hpp = [0] * items_per_ranking
    for user_id in alg_res.lists:
        if user_id in user_list and user_id in hits.lists:
            for key, value in enumerate(alg_res.lists[user_id]):
                if value in hits.lists[user_id]:
                    hpp[key] += 1
    return hpp

def gen_cluster_hpp(dsr, cluster_users):
    res = {}
    for p, (algs, hits) in dsr.parts.items():
        for alg, alg_res in algs.items():
            if alg not in res:
                res[alg] = {}
            res[alg][p] = hits_per_position(alg_res, hits, dsr.len_ranking,
                cluster_users)

    res_avg = {alg: [sum(z) for z in zip(*parts.values())]
               for alg, parts in res.items()}

    sum_all = [sum(z) for z in zip(*res_avg.values())]
    res_avg['all_algs'] = sum_all

    return res_avg

def generate(dsr, agr_res, num_clusters=8):
    res = {}
    for p, algs in agr_res.items():
        res[p] = {}
        for alg, (m, user_ids, centroids, labels) in algs.items():
            res[p][alg] = {}
            for cl_i in range(num_clusters):
                cluster_users = [user_ids[u_i] for u_i in range(len(labels))
                    if labels[u_i] == cl_i]
                res[p][alg][cl_i] = (gen_cluster_hpp(dsr, cluster_users),
                    len(cluster_users)) # Cluster's # of users
    return res

def plot_cluster_hpp(clusters, dsr, output_dir, ext='pdf'):
    positions = list(range(dsr.len_ranking))

    for cl_i, (algs, cl_num_users) in clusters.items():
        cl_dir = os.path.join(output_dir, str(cl_i))
        if not os.path.exists(cl_dir):
            os.makedirs(cl_dir)
        for alg, hpp in algs.items():

            plot_histogram(os.path.join(cl_dir, alg + '-hpp.{0}'.format(ext)),
                           "{0} - Hits por índice".format(alg),
                           "Índice no ranking", positions,
                           "Número de hits", hpp)

            plot_histogram(os.path.join(cl_dir, alg + '-hn.{0}'.format(ext)),
                           "{0} - Hits por índice / nº de usuários do cluster"
                           " ({1} usuários)".format(alg, cl_num_users),
                           "Índice no ranking", positions,
                           "Número de hits", np.array(hpp) / cl_num_users)

            plot_histogram(os.path.join(cl_dir, alg + '-pdf.{0}'.format(ext)),
                           "{0} - Hits por índice (normalizados)".format(alg),
                           "Índice no ranking", positions,
                           "Número de hits / total de hits", hpp, normed=1)

            plot_histogram(os.path.join(cl_dir, alg + '-cdf.{0}'.format(ext)),
                           "{0} - Função de densidade cumulativa".format(alg),
                           "Índice no ranking", positions,
                           "Probabilidade cumulativa de hit", hpp, normed=1,
                           cumul=1, y_range=(0,1))

def plot(res, dsr, output_dir, ext='pdf'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for p, algs in res.items():
        p_dir = os.path.join(output_dir, p)
        for alg, clusters in algs.items():
            alg_dir = os.path.join(p_dir, alg)
            plot_cluster_hpp(clusters, dsr, alg_dir, ext)
