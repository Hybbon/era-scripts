import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple
from ..plotting import plot_matrix_heatmap

Heatmap = namedtuple('Heatmap', ['matrix', 'id_y'])

DEFAULT_MATRIX_VAL = 120

# m[alg_i][item][pos]

def generate(dsr):
    res = {}

    heat_maps = {}
    for p, (algs, hits) in dsr.parts.items():
        alg_dicts = []

        for alg_i, alg in enumerate(dsr.algs):
            alg_res = algs[alg]
            alg_dicts.append({})
            for user in alg_res.lists.values():
                for pos, item in enumerate(user):
                    if item not in alg_dicts[alg_i]:
                        alg_dicts[alg_i][item] = [pos]
                    else:
                        alg_dicts[alg_i][item].append(pos)

            avg = lambda iterable: sum(iterable) / len(iterable)
            for item, pos_list in alg_dicts[alg_i].items():
                alg_dicts[alg_i][item] = avg(pos_list)

        alg_dict_keys = set(alg_dicts[0].keys())
        for i in range(1, len(alg_dicts)):
            alg_dict_keys |= set(alg_dicts[i].keys())

        rows = len(alg_dict_keys)
        columns = len(dsr.algs)

        id_y = {}

        matrix = np.full((columns, rows), DEFAULT_MATRIX_VAL)
        for alg_i, alg_dict in enumerate(alg_dicts):
            for item, pos in alg_dict.items():
                if item not in id_y:
                    id_y[item] = len(id_y)
                matrix[alg_i, id_y[item]] = pos

        res[p] = Heatmap(matrix, id_y)

    return res

def plot(res, dsr, output_dir, ext='pdf'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for p, heatmap in res.items():

        x_ticks = dsr.algs
        y_ticks = [None] * len(heatmap.id_y)
        for item, y in heatmap.id_y.items():
            y_ticks[y] = item

        plot_matrix_heatmap(os.path.join(output_dir, p + "-heatmap.{0}".format(ext)),
                            "Posição em que ocorrem hits de cada filme por algoritmo",
                            "Algoritmo", x_ticks, "ID do Filme", y_ticks,
                            heatmap.matrix.T, rows_per_plot=75, color_min=0,
                            color_max=dsr.len_ranking)




