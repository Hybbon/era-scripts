import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple
from ..plotting import plot_matrix_heatmap, plot_scatter
from ..metrics import precision
from operator import itemgetter

VictoryMatrix = namedtuple('VictoryMatrix', ['matrix', 'item_ids'])
VictDefScatter = namedtuple('VictDefScatter', ['x', 'y', 'item_ids'])
TOLERANCE = 5


def user_rankings_and_items(algs, user_id):
    rankings = []
    for alg_res in algs.values():
        if user_id in alg_res.lists:
            rankings.append(alg_res.lists[user_id])
    items = set()
    for ranking in rankings:
        items |= set(ranking)
    return rankings, sorted(items)


def victories(algs, tol):
    res = {}
    for alg, alg_res in algs.items():
        for user_id in alg_res.lists:
            if user_id not in res:
                rankings, items = user_rankings_and_items(algs, user_id)
                itm_idx = {}
                for i, item_id in enumerate(items):
                    itm_idx[item_id] = i
                m = np.zeros((len(items), len(items)), dtype=int)
                for ranking in rankings:
                    for i, w_id in enumerate(ranking):
                        for l_id in ranking[i + tol:]:
                            m[itm_idx[w_id], itm_idx[l_id]] += 1
                res[user_id] = VictoryMatrix(m, items)
    return res


def vict_def_scatter(victories):
    res = {}
    for user_id, (vict_m, item_ids) in victories.items():
        x = vict_m.sum(0)
        y = vict_m.sum(1)
        res[user_id] = VictDefScatter(x, y, item_ids)
    return res


def user_precisions(algs, hits, num_hits=None):
    prelim = {}
    for alg, alg_res in algs.items():
        # TODO: calculate weights for each algorithm's results based on MAP
        for user_id, ranking in alg_res.lists.items():
            if user_id not in prelim:
                prelim[user_id] = []
            h = [] if user_id not in hits.lists else hits.lists[user_id]
            prelim[user_id].append(precision(ranking, h, num_hits))
    return {user_id: sum(prelims) for user_id, prelims in prelim.items()}


def generate(dsr, num_users=8):
    res = {}

    tol = TOLERANCE

    for p, (algs, hits) in dsr.parts.items():
        v = victories(algs, tol)
        s = vict_def_scatter(v)
        u_precisions = user_precisions(algs, hits)
        u_p = sorted(u_precisions.items(), key=itemgetter(1))

        mid = ((len(u_p) - num_users) // 2, (len(u_p) + num_users) // 2)

        user_ids = [user_id for user_id, prec in u_p[:num_users] + u_p[mid[0]:mid[1]] + u_p[-1 * num_users:]]

        v_filtered = {user_id: v[user_id] for user_id in user_ids}
        s_filtered = {user_id: s[user_id] for user_id in user_ids}

        res[p] = (v_filtered, s_filtered, u_precisions)

    return res


def plot(res, dsr, output_dir, ext='pdf'):
    for p, (victories, scatters, user_precisions) in res.items():
        p_dir = os.path.join(output_dir, p)
        if not os.path.exists(p_dir):
            os.makedirs(p_dir)

        for (user_id, v), s in zip(victories.items(), scatters.values()):
            v_filename = "{0}-{1}-victories.{2}".format(user_id, user_precisions[user_id], ext)
            v_path = os.path.join(p_dir, v_filename)
            plot_matrix_heatmap(v_path, "Vitórias de um item sobre outro item",
                                "ID do vencedor", v.item_ids, "ID do perdedor", v.item_ids,
                                v.matrix, text=False, color_min=0, color_max=max(v.matrix.flat),
                                rows_per_plot=len(v.item_ids), cmap='RdPu', text_size='small')
            s_filename = "{0}-{1}-scatter.{2}".format(user_id, user_precisions[user_id], ext)
            s_path = os.path.join(p_dir, s_filename)
            plot_scatter(s_path, "Vitórias x Derrotas de cada item do usuário",
                         s.x, "Vitórias", s.y, "Derrotas")
