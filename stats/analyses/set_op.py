"""Ranking intersection analyses module"""

ARGPARSE_PARAMS = ("-S", "--set_op", "Ranking intersection analyses module")
MODULE_NAME = "set_op"

import os
import numpy as np
import pandas as pd
from itertools import combinations
from ..plotting import plot_bar_chart, plot_all_and_hits, plot_frame_histogram


def set_operation_stats(algs, begin=0, end=None, hits=None):
    """Computes avg. intersection length for Rankings.

    Returns two lists, indexed by intersection size - 1:

    avg_isect_per_r: average length of the intersection between r AlgResults.

    If hits is defined, then the final AlgResults object is the intersection
    between the intersection between the r algorithms and the hits AlgResults
    object.

    algs -- dictionary of AlgResults indexed by algorithm name.
    hits (optional) -- AlgResults object containing the hits for each user from
    the test base.
    begin, end (optional) -- range for which the stats should be generated in
                             each user's item list.
    """
    if begin or end:
        new_algs = {}
        for alg, alg_res in algs.items():
            new_algs[alg] = alg_res.slice(begin, end)
        algs = new_algs

    tuples = []

    alg_res_list = list(algs.values())
    comb_lengths = range(1, len(algs) + 1)
    avg_isect_len_per_r = []
    for r in comb_lengths:
        isect_lenghts = []
        # Iterates through combinations of size r
        for comb in combinations(range(len(alg_res_list)), r):
            # Performs the intersection between the items in the combination
            isect = alg_res_list[comb[0]]
            for i in comb[1:]:
                isect &= alg_res_list[i]
            if hits:
                isect &= hits

            for user_id, ranking in isect.lists.items():
                tuples.append((
                    r,
                    user_id,
                    len(ranking),
                    comb
                ))

            avg_isect_len = isect.avg_len()

            isect_lenghts.append(avg_isect_len)
        avg_isect_len_per_r.append(sum(isect_lenghts) / len(isect_lenghts))

    col_names = ("comb_length", "user_id", "isect_size", "comb")
    frame = pd.DataFrame.from_records(tuples, columns=col_names)

    return avg_isect_len_per_r, frame


def print_hits_per_alg(algs, hits):
    for alg, alg_res in algs.items():
        print("{0}: {1} hits".format(alg, (alg_res & hits).avg_len()))


def count_by_size(frame):
    summed = frame.groupby('isect_size').apply(lambda f: f.count().user_id)
    return summed


def find_best_combinations(frame):
    each_comb_len = frame.groupby('comb_length')
    best_comb = {}
    for comb_length, cl_frame in each_comb_len:
        comb_sum = cl_frame.groupby('comb').apply(lambda f: f.sum().isect_size)
        best_comb[comb_length] = comb_sum.sort_values(ascending=False).index[0]
    return best_comb


def filter_by_best_comb(frame, best_comb):
    if best_comb:
        each_comb_len = frame.groupby('comb_length')
        out = pd.DataFrame()
        for comb_len, cl_frame in each_comb_len:
            out = out.append(cl_frame[cl_frame.comb == best_comb[comb_len]])
        del out['comb']
        return out
    else:
        del frame['comb']
        return frame


def isect_histogram_frame(frame, best_comb=None):
    frame = filter_by_best_comb(frame, best_comb)
    counts = frame.groupby('comb_length').apply(count_by_size)
    counts = counts.reset_index()
    import pdb
    pdb.set_trace()
    counts.columns = ['comb_length', 'isect_size', 'count']
    counts = counts[counts.comb_length <= 5]
    counts = counts.sort_values(['comb_length', 'isect_size'])
    return counts


def generate(dsr, results, conf):
    labels = ['isect_all', 'isect_hits']
    slices = [tuple(l) for l in conf['slices']]

    # Results are stored, for each kind of result, in a 3D numpy matrix. The
    # dimensions represent slices, partitionings and combination sizes, respec-
    # tively.
    res = {
        label: np.ndarray([len(slices), len(dsr.parts), len(dsr.algs)])
        for label in labels
    }

    frames_all, frames_hits = {}, {}

    for p_i, (p, part) in enumerate(dsr.parts.items()):
        algs, hits = part.algs, part.hits

        print("Fold {0}".format(p))
        print_hits_per_alg(algs, hits)
        print()

        for s_i, s in enumerate(slices):
            res['isect_all'][s_i, p_i, :], frame_all = (
                set_operation_stats(algs, *s))
            frames_all[s] = frame_all
            res['isect_hits'][s_i, p_i, :], frame_hits = (
                set_operation_stats(algs, *s, hits=hits))
            frames_hits[s] = frame_hits

    def avg(iterable):
        return sum(iterable) / len(iterable)

    res_avg = {
        label: {
            s: [
                avg(res[label][s_i, :, r])
                for r in range(len(dsr.algs))
            ]
            for s_i, s in enumerate(slices)
        }
        for label in labels
    }

    best_comb = find_best_combinations(frame_hits)

    frames_all = {s: (isect_histogram_frame(f, best_comb),
                      isect_histogram_frame(f))
                  for s, f in frames_all.items()}
    frames_hits = {s: (isect_histogram_frame(f, best_comb),
                       isect_histogram_frame(f))
                   for s, f in frames_hits.items()}

    return res_avg, frames_all, frames_hits


def normalize_by_num_users(frame):
    def normalize(cl_frame):
        num_users = cl_frame.sum()['count']
        cl_frame['count'] = cl_frame['count'] / (num_users * 0.01)
        return cl_frame
    return frame.groupby('comb_length').apply(normalize).reset_index()


def plot(res_tuple, dsr, output_dir, conf, ext='pdf'):
    (res_avg, frames_all, frames_hits) = res_tuple
    slices = [tuple(l) for l in conf['slices']]

    comb_lengths = list(range(1, len(dsr.algs) + 1))  # x axis label

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for s in slices:
        slice_dir = os.path.join(output_dir, str(s))
        if not os.path.exists(slice_dir):
            os.makedirs(slice_dir)

        isect_all = res_avg['isect_all'][s]
        isect_hits = res_avg['isect_hits'][s]

        frame_all_best, frame_all_mean = frames_all[s]
        frame_hits_best, frame_hits_mean = frames_hits[s]

        frame_all_best = frame_all_best[frame_all_best.comb_length > 1]
        frame_all_mean = frame_all_mean[frame_all_mean.comb_length > 1]

        frame_all_best = normalize_by_num_users(frame_all_best)
        frame_all_mean = normalize_by_num_users(frame_all_mean)
        frame_hits_best = normalize_by_num_users(frame_hits_best)
        frame_hits_mean = normalize_by_num_users(frame_hits_mean)

        # plot_bar_chart(os.path.join(slice_dir, 'isect_all.{0}'.format(ext)),
        #                "Interseção entre k algoritmos (todos os itens)",
        #                "Número de algoritmos combinados (k)", comb_lengths,
        #                "Itens em comum", isect_all)

        # plot_bar_chart(os.path.join(slice_dir, 'isect_hits.{0}'.format(ext)),
        #                "Interseção entre k algoritmos (apenas hits)",
        #                "Número de algoritmos combinados (k)", comb_lengths,
        #                "Itens em comum", isect_hits)

        # plot_all_and_hits(os.path.join(slice_dir,
        #                                'isect_both.{0}'.format(ext)),
        #                   "Intersection between k algorithms",
        #                   "Combination size (k)", comb_lengths,
        #                   "Avg. items in common", isect_all, isect_hits)

        plot_frame_histogram(os.path.join(slice_dir,
                                          'hist_all_best.{0}'.format(ext)),
                             "Number\nof items",
                             "Number of methods in agreement",
                             "Users (%)", frame_all_best)
        # plot_frame_histogram(os.path.join(slice_dir,
        #                                   'hist_all_mean.{0}'.format(ext)),
        #                      "Number\nof items",
        #                      "Number of methods in agreement",
        #                      "Users (%)", frame_all_mean)
        plot_frame_histogram(os.path.join(slice_dir,
                                          'hist_hits_best.{0}'.format(ext)),
                             "Number\nof items",
                             "Number of methods in agreement",
                             "Users (%)", frame_hits_best)
        # plot_frame_histogram(os.path.join(slice_dir,
        #                                   'hist_hits_mean.{0}'.format(ext)),
        #                      "Number\nof items",
        #                      "Number of methods in agreement",
        #                      "Users (%)", frame_hits_mean)
