import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple
from ..plotting import plot_matrix_heatmap

HitMatrix = namedtuple('HitMatrix', ['matrix', 'user_ids'])

def generate(dsr):
    res = {}

    for p, (algs, hits) in dsr.parts.items():
        res[p] = {}
        for alg, alg_res in algs.items():
            # m[user_id][ranking_pos]
            m = np.zeros((len(alg_res.lists), dsr.len_ranking), dtype=bool)
            user_ids = []
            for u_i, (user_id, ranking) in enumerate(alg_res.lists.items()):
                user_ids.append(user_id)
                if user_id in hits.lists:
                    for pos, item_id in enumerate(ranking):
                        m[u_i, pos] = item_id in hits.lists[user_id]
            res[p][alg] = HitMatrix(m, user_ids)

    return res


def plot(res, dsr, output_dir, ext='pdf'):
    for p, algs in res.items():

        x_ticks = list(range(dsr.len_ranking))

        p_dir = os.path.join(output_dir, p)
        for alg, (matrix, user_ids) in algs.items():
            alg_dir = os.path.join(p_dir, alg)
            if not os.path.exists(alg_dir):
                os.makedirs(alg_dir)

            for begin in range(0, len(user_ids), 100):
                end = len(user_ids) if len(user_ids) < begin + 100 else begin + 100

                y_ticks = user_ids[begin:end]

                path = os.path.join(alg_dir, "{0}-{1}.{2}".format(begin, end, ext))
                plot_matrix_heatmap(path, "Hits por posição por usuário",
                    "Posição no ranking", x_ticks, "ID do usuário", y_ticks,
                    matrix[begin:end, :], text=False, color_min=0, color_max=1,
                    rows_per_plot=len(y_ticks), cmap='RdPu')
