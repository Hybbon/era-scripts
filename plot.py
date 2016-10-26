#!/usr/bin/env python3

"""
plot.py
===========
Dataset analyzer - output generator
-----------------------------------

This script is, together with generate.py, responsible for generating
dataset analyses, such as hit-per-position and agreement charts.

*plot.py* must only be run after calling *generate.py* for the same
analysis modules. It will, then, generate the output files for the
modules, such as charts and csv files. The calling syntax is very
similar to *generate.py*'s.'

This script runs according to RankingStats config files. (Read more
about them in the README file at the top directory of this repo.)

The analyses to be performed are specified via command-line uppercase
flags. The complete list of analysis modules can be accessed through
the default help flag, ``-h``.

Examples::

    python plot.py -SHA bases/ml100k/

The call above will create three modules' results, specified by the
flags ``-S``, ``-H`` and ``-A`` with the dataset at ``bases/ml100k/``.
"""


import stats
import os
import argparse
import stats.aux as aux
import matplotlib.pyplot as plt


def parse_args():
    """Parses command line parameters through argparse and returns parsed args.
    """

    try:
        file_formats = list(plt.gcf().canvas.get_supported_filetypes().keys())
    except:
        file_formats = ["pdf", "jpg"]

    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str,
                        help="folder containing the config file.")
    parser.add_argument("-e", "--ext", type=str, default="pdf",
                        choices=file_formats,
                        help="format of the files to be generated.")
    parser.add_argument("-i", "--input", type=str, default="results.pickle",
                        help="file from which the pickled data should be loa"
                        "ded, relative to data. (default: %(default)s)")
    parser.add_argument("-o", "--output", type=str, default="pdf",
                        help="folder where the generated pdfs should be saved."
                        " (default: %(default)s)")
    parser.add_argument("-c", "--config", type=str, default="",
                        help="name of the config file. (default: %(default)s)")
    stats.parse_modules(parser)
    return parser.parse_args()


def plot_modules(modules, args, dsr, conf, results):
    output_dir = os.path.join(args.data, args.output)
    args_dict = vars(args)
    print("Outputting data for the following modules:")
    for module in modules:
        name = module.MODULE_NAME
        if args_dict[name]:
            print("- {}".format(name))
            module_conf = conf[name] if name in conf else None
            module_dir = os.path.join(output_dir, name)
            module.plot(results[name], dsr, module_dir, module_conf,
                        ext=args.ext)


def main():
    args = parse_args()
    conf = aux.load_configs(aux.CONF_DEFAULT, os.path.join(args.data,
                            aux.BASE_CONF), args.config)

    pickle_path = os.path.join(args.data, args.input)
    results = aux.load_results(pickle_path)

    dsr = stats.DataSetResults(conf, args.data)

    plot_modules(stats.modules, args, dsr, conf, results)


if __name__ == "__main__":
    main()
