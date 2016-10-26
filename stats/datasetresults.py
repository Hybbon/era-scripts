import os
import pandas as pd
from collections import namedtuple
from .file_input import ratings_dict, rankings_dict
from .metrics import gen_part_metrics
from .algresults import AlgResults

Part = namedtuple('Part', ['algs', 'hits', 'gpra', 'metrics'])


class DataSetResults:
    """Results and hits for many recommendation algorithms for a dataset."""

    def __init__(self, conf, folder):
        """Initializes a new object from a config dictionary and a directory
        which contains results.

        conf -- dictionary which contains the following settings:
            'len_ranking': length of each user's ranking
            'algs': list of algorithm names
            'parts': list of partitioning prefixes
            'alg_form': format specification for each algorithm's results' file
                        name
            'hits_form': format specification for the hits file's file name
        folder -- relative address to the directory which contains the results
                  and hits files.
        """
        self.len_ranking = conf['base']['len_ranking']
        self.algs = conf['algs']
        self.parts = {}
        self.raw = self._load_raw(os.path.join(folder, conf['base']['raw']))
        for p in conf['parts']:
            p_part = self._load_part(
                folder,
                conf['algs'],
                conf['base']['alg_form'],
                conf['base']['hits_form'],
                conf['base']['gpra_form'],
                p)
            self.parts[p] = p_part

    def _load_part(self, folder, algs, alg_form, hits_form, gpra_form, p=""):
        """Loads all rankings and the test base from a partition.

        This function returns two objects: a dictionary indexed by algorithm
        name which contains AlgResults objects composed by each algorithm's
        rankings and a AlgResults object (unordered) which contains the hits
        for each user, as defined by the test base for this partition.

        folder -- relative address to the directory which contains the results
                  and hits files.
        algs -- list of algorithm names
        alg_form --  format specification for each algorithm's results' file
                     name
        hits_form -- format specification for the hits file's file name
        p (optional) -- prefix string for the partition to be loaded
        """
        rankings = {}
        for alg in algs:
            addr = os.path.join(folder, alg_form.format(p, alg))
            rankings[alg] = AlgResults.from_dict(rankings_dict(addr))
        hits_addr = os.path.join(folder, hits_form.format(p))
        hits = AlgResults.from_dict(ratings_dict(hits_addr))
        gpra_addr = os.path.join(folder, gpra_form.format(p))
        try:
            gpra = AlgResults(rankings_dict(gpra_addr))
        except:
            gpra = None

        metrics = gen_part_metrics(rankings, self.algs, hits)

        return Part(rankings, hits, gpra, metrics)

    def _load_raw(self, raw_addr):
        headers = ("user_id", "item_id", "rating")
        return pd.read_csv(raw_addr, "\t", names=headers)
