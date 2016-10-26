import math
import numpy as np
import itertools


def precision(ranking, hits, num_items=None):
    """Calculates a ranking's precision, based on a list of hit items.

    This function returns a normalized value (in [0,1]) which gauges the preci-
    sion of a ranking to a user's tastes.

    ranking -- list of item ids ordered by descending relevance.
    hits -- unordered container of item ids which are relevant to the user for
    which the rankings in the previous parameter were generated.
    num_items (optional) -- if specified, only the first num_items items of the
    ranking shall be considered."""
    ranking = ranking[:num_items]
    num_hits = 0
    precision = 0

    for i, item_id in enumerate(ranking):
        if item_id in hits:
            num_hits += 1
            precision += num_hits / (i + 1)

    # Checar: dividir pelo menor tamanho entre treino e teste p/ usuario?
    return precision / min(len(ranking), len(hits))


def dcg(rel_list):
    """Calculates the DCG, based on a list of relevance values.

    rel_list -- list of relevance values (floating point numbers"""
    if len(rel_list) == 0:
        raise ValueError("container must not be empty.")
    dcg = rel_list[0]
    for i, rel in enumerate(rel_list[1:]):
        i += 2  # Compensates for 0-indexing and processing the first
        # item separately
        dcg += rel/math.log(i,2)
    return dcg


def ndcg(ranking, hits, num_items=None):
    """Calculates a ranking's NDCG, based on a list of hit items.

    ranking -- list of item ids ordered by descending relevance.
    hits -- unordered container of item ids which are relevant to the user for
    which the rankings in the previous parameter were generated.
    num_items (optional) -- if specified, only the first num_items items of the
    ranking shall be considered."""
    ranking = ranking[:num_items]

    rel_list = [1 if item_id in hits else 0 for item_id in ranking]

    optimal_res_list = sorted(rel_list, reverse=True)

    optimal_dcg = dcg(optimal_res_list)

    return 0 if optimal_dcg == 0 else dcg(rel_list) / optimal_dcg


def _reverse_index(ranking):
    return {item: index for index, item in enumerate(ranking)}


def kendall(a, b, penalty=0.5):
    """Calculates Kendall's Tau distance between two rankings.

    It is necessary and enforced that both rankings have the same
    length.

    a, b -- rankings to be compared."""
    assert len(a) == len(b)
    length = len(a)
    pos_a, pos_b = _reverse_index(a), _reverse_index(b)
    item_list = list(set(a) | set(b))

    inversion_count = 0

    for i, x in enumerate(item_list):
        for y in item_list[i + 1:]:
            # First two ifs: if both items are present in a list but not in
            # the other, a penalty p (0 < p <= 1) is added to the distance
            if x not in pos_a and y not in pos_a:
                if x not in pos_b and y not in pos_b:
                    inversion_count += penalty
            elif x not in pos_b and y not in pos_b:
                if x in pos_a != y in pos_a:
                    inversion_count += penalty
            else:
                a_greater = pos_a.get(x, length) > pos_a.get(y, length)
                b_greater = pos_b.get(x, length) > pos_b.get(y, length)
                inversion_count += a_greater != b_greater

    num_items = len(item_list)

    return inversion_count / (num_items * (num_items - 1) / 2)


def footrule(a, b):
    """Calculates Spearman's Footrule distance between two rankings.

    It is necessary and enforced that both rankings have the same
    length.

    a, b -- rankings to be compared."""
    assert len(a) == len(b)
    length = len(a)
    pos_a, pos_b = _reverse_index(a), _reverse_index(b)
    item_list = list(set(a) | set(b))

    distance = sum(
        abs(pos_a.get(item, length) - pos_b.get(item, length))
        for item in item_list
    )

    # length: maximum distance for a given item
    return distance / (length * len(item_list))



def gen_metrics(dsr, parts_labels, algs_labels, lengths):
    res = {}
    for key in ['ndcg', 'map', 'kendall', 'spearman']:
        res[key] = {}
        for l in lengths:
            res['ndcg'][l] = np.ndarray((len(parts_labels), len(algs_labels)))

    for p_i, p in enumerate(parts_labels):
        algs, hits = dsr.parts[p].algs, dsr.parts[p].hits
        for a_i, alg in enumerate(algs_labels):
            alg_res = algs[alg]
            for l in lengths:
                res['ndcg'][l][p_i, a_i] = alg_res.ndcg_metrics(hits, l)
                res['map'][l][p_i, a_i] = alg_res.map_metrics(hits, l)

    return res


def gen_part_metrics(algs, algs_keys, hits, length=10):
    ndcgs, maps = [], []  # Indexed in parallel with the algs keys
    for alg in algs_keys:
        alg_res = algs[alg]
        ndcgs.append(alg_res.ndcg_metrics(hits, length))
        maps.append(alg_res.map_metrics(hits, length))
    return {'ndcg': ndcgs, 'map': maps}

