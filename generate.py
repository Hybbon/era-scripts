#!/usr/bin/env python3

"""
generate.py
===========
Dataset analyzer - calculation and processing
---------------------------------------------

This script is, together with plot.py, responsible for generating
dataset analyses, such as hit-per-position and agreement charts.

generate.py should be run first, because it is the one which does the
actual processing and number crunching. After execution, it saves the
result objects to a .pickle file, which is, then, loaded by plot.py
in order to make any required output files and visualizations.

This script runs according to RankingStats config files. (Read more
about them in the README file at the top directory of this repo.)

The analyses to be performed are specified via command-line uppercase
flags. The complete list of analysis modules can be accessed through
the default help flag, ``-h``.

New runs will overwrite old runs of the same analysis, but any
other results will be kept (unless the ``-d`` flag is set).

Examples::

    python generate.py -SHA bases/ml100k/

The call above will run three analysis modules, specified by the flags
``-S``, ``-H`` and ``-A`` with the dataset at ``bases/ml100k/``.

    python generate.py -d -S bases/dataset/

This call will, before running the analysis, delete any old results at
the ``results.pickle`` file.

Implementing new analysis modules
=================================

New analysis modules can be added by creating a new module inside the
stats.analyses package and implementing two functions at the root of
these modules::

    generate(dsr: DataSetResults,
             results: dict[str, object],
             conf: dict) -> result: object

Does any required processing before output and stores results in
the output object. The provided ``results`` dictionary contains
results for other modules, which allows a module to use another
module's results. In this case, extra care must be taken in order
to make sure the required module is executed before the new one.

    plot(res: object, dsr: DataSetResults, output_dir: str,
         conf: dict, ext: str) -> None

Outputs any required files at output_dir. ``res`` refers to the
object created by the ``generate`` call.

Then, in order to enable the new module and add it to the list of
arguments of *generate.py* and *plot.py*, two more things must be
done. First, at the top of the module, two global variables must
be defined::

    ARGPARSE_PARAMS = ("-H", "--hpp", "Hit per position histograms")
    MODULE_NAME = "hpp"

``ARGPARSE_PARAMS`` contains three parameters for *argparse*: the
short-form parameter, the long, double-dash parameter and the help
text. ``MODULE_NAME`` is quite self-explanatory, and must be a valid
Python identifier.

Finally, one must modify stats/__init__.py and add the new module to
the ``modules`` list. Note that the order specified here is the order
in which the modules will be executed.
"""

import stats
import os.path
import argparse
import stats.aux as aux
import logging
import ipdb
from datetime import datetime



def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="folder containing the config file and the"
                        " ranking data to be evaluated.")
    parser.add_argument("-o", "--output", type=str, default="results.pickle",
                        help="file where the pickled data should be saved, "
                        "relative to data folder. (default: %(default)s)")
    parser.add_argument("-c", "--config", type=str, default="",
                        help="name of the config file. (default: %(default)s)")
    parser.add_argument("-d", "--delete", action='store_true',
                        default=False, help="If set, previous results will be"
                        " deleted. Otherwise, they'll be overwritten on a mo"
                        "dule basis, if their arguments are set or kept.")
    parser.add_argument("-l", "--list", action='store_true', default=False,
                        help="If set, results already generated will be "
                        "listed, together will the date of the latest update.")
    parser.add_argument("-s", "--skip_dsr", action='store_true', default=False,
                        help="If set, the dataset's results will not be "
                        "loaded. This can be useful, but most analyses require"
                        " that these results be loaded. Proceed with care.")
    stats.parse_modules(parser)
    return parser.parse_args()


def list_results(results):
    print("List of generated modules:")
    for key in results.keys():
        print("- " + key)
    print("Last update: {0}".format(results['datetime']))


def gen_modules(modules, args, dsr, conf, results):
    args_dict = vars(args)

    print("Generating new analyses for the following modules:")
    for module in modules:
        name = module.MODULE_NAME
        if args_dict[name]:
            print("- {}".format(name))
            module_conf = conf[name] if name in conf else None
            results[name] = module.generate(dsr, results, module_conf)

    results['datetime'] = datetime.now()
    return results


def main():
    args = parse_args()
    conf = aux.load_configs(aux.CONF_DEFAULT,
                            os.path.join(args.data, aux.BASE_CONF),
                            args.config)

    # ratings = aux.load_ratings(os.path.join(args.data, conf['ratings']))
    pickle_path = os.path.join(args.data, args.output)

    dsr = None if args.skip_dsr else stats.DataSetResults(conf, args.data)

    if args.delete:
        results = {}
    else:
        try:
            results = aux.load_results(pickle_path)
        except FileNotFoundError:
            results = {}

    if args.list:
        list_results(results)
        return True
    results = gen_modules(stats.modules, args, dsr, conf, results)

    aux.save_results(pickle_path, results)


if __name__ == "__main__":
    main()
