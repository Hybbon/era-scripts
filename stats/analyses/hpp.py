"""Hit per position histograms"""

ARGPARSE_PARAMS = ("-H", "--hpp", "Hit per position histograms")
MODULE_NAME = "hpp"
ACRONYMS = {"BPRSLIM" : "BPRSLIM",
        	"CofiRank" : "Cofi",
	        "FISM_librec" : "FISM",
        	"Hybrid_librec" : "Hybrid",
        	"ItemKNN" : "ItemKNN",
        	"LDA_librec" : "LDA",
        	"LeastSquareSLIM" : "LSSLIM",
        	"libfm" : "libFM",
        	"MostPopular" : "MP",
        	"MultiCoreBPRMF" : "BPRMF",
        	"RankALS_librec" : "RALS",
        	"SoftMarginRankingMF" : "SMRMF",
        	"WRMF" :  "WRMF",
        	"Poisson" : "PF",
            "CoFactor" : "CF"}


import os
import numpy as np
from ..plotting import plot_histogram, plot_multiple_cumul


def hits_per_position(alg_res, hits, items_per_ranking):
    """Returns a list with the hits per position in a AlgResults object.

    alg_res -- AlgResults object to be evaluated.
    hits -- AlgResults object containing the hits for each user from the test
            base.
    items_per_ranking -- number of items in each user's ranking."""
    hpp = [0] * items_per_ranking
    for user_id in alg_res.lists:
        if user_id in hits.lists:
            for key, value in enumerate(alg_res.lists[user_id]):
                if value in hits.lists[user_id]:
                    hpp[key] += 1
    return hpp


def generate(dsr, results, conf):
    res = {}
    for p, part in dsr.parts.items():
        for alg, alg_res in part.algs.items():
            if alg not in res:
                res[alg] = {}
            res[alg][p] = hits_per_position(alg_res, part.hits,
                                            dsr.len_ranking)

    res_avg = {alg: [sum(z) for z in zip(*parts.values())]
               for alg, parts in res.items()}

    sum_all = [sum(z) for z in zip(*res_avg.values())]
    res_avg['all_algs'] = sum_all

    return res_avg, res


def alg_name_by_avg_map(dsr):
    map_lists = []
    for part in dsr.parts.values():
        map_lists.append(np.array(part.metrics['map']))
    map_sums = sum(map_lists)
    avg_map_and_name = zip(map_sums, dsr.algs)
    sorted_tuples = sorted(avg_map_and_name)
    return [t[1] for t in sorted_tuples]

def plot(res, dsr, output_dir, conf, ext='pdf'):
    res_avg = res[0]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    positions = list(range(dsr.len_ranking))

    sorted_alg_names = alg_name_by_avg_map(dsr)
    best_key = sorted_alg_names[-1]
    worst_key = sorted_alg_names[0]
    mp_key = 'MostPopular'

    data = []
    data.append(('all (mean)', res_avg['all_algs']))

    data.append((ACRONYMS[best_key]+' (best)', res_avg[best_key]))
    data.append((ACRONYMS[worst_key]+' (worst)', res_avg[worst_key]))

    data_mp = data[:]

    if mp_key != best_key and mp_key != worst_key:
        data_mp.append((mp_key, res_avg[mp_key]))

    plot_multiple_cumul(os.path.join(output_dir, 'cumul.{0}'.format(ext)),
                        "", "Position in the ranking", positions,
                        "Hit cumulative probability", data,
                        y_range=(0, 1))

    plot_multiple_cumul(os.path.join(output_dir, 'cumul_mp.{0}'.format(ext)),
                        "", "Position in the ranking", positions,
                        "Hit cumulative probability", data_mp,
                        y_range=(0, 1))

    # plot_multiple_cumul(os.path.join(output_dir, 'cumul_all.{0}'.format(ext)),
    #                     "", "Position in the ranking", positions,
    #                     "Hit cumulative probability", res_avg,
    #                     y_range=(0, 1))
