RankingStats
============

This repository is a collection of assorted scripts for dealing with
rating datasets and recommender systems. All Python files at the top
level directory are executable scripts. Detailed documentation on how
to run them and what each of them does is available at the docstrings
in the beginning of each script, but here's a quick summary:

- distances.py: Calculator for distances between rankings
- filter_yelp.py: Filters the yelp dataset to a single location
- generate.py: Dataset analyzer - calculation and processing. The
  docstring for this script also contains instructions on how to
  implement new analysis modules.
- plot.py: Dataset analyzer - output generator
- metrics.py: Recommender metrics calculator
- mmltohyper.py: Converts a MML-format dataset to Hyper's format and
  structure. Also contains information about HyPER execution and the
  conversion process in its docstring.
- pre_process.py: Dataset pre-processor and cross-validation fold
  generator
- run_recommenders.py: Recommender algorithm multiprocess batch
  runner

All scripts support command-line parameters, which are implemented
with *argparse*. Help text for them can be found through the ``-h``
flag.

Requirements
------------

All provided scripts were written with support for Python 3.3+ in
mind. Newer versions are also supported. The following packages are
required:

- *numpy*
- *scipy*
- *scikit-learn*
- *matplotlib*
- *seaborn*
- *pandas*

All of these can be installed through *pip*. A few other requirements
are needed for running recommenders - refer to *run_recommenders.py*
for more information.

Configuration files
-------------------

Many scripts support a unified configuration file format. These config
files follow a hierarchy. First, the default configuration is loaded
from ``config/config_default.json``. Then, an override for the dataset
is loaded from a ``config.json`` file in its directory, if there is
such a file. Finally, if specified, a final override is loaded from
the address set via the ``-c <path>`` parameter.


To fully understand the configuration options, read the default
configs at ``config/config_default.json``. The most important options
are explained below:

- *parts*: names of the cross-validation folds (note: not yet used in
  the pre-processing script, which generates u1 to uN regardless of
  these settings as hard-coded behavior.)
- *algs*: recommender algorithms to be used by *run_recommenders.py*
  and whose results should be read by the analysis scripts.
- *base*: contains format strings and settings regarding the files to
  be read from the data folder by the analysis scripts.
- *pre_process*: contains the type of the dataset, its field separator
  and the source filename, to be used by *pre_process.py*.


Some useful override files are also provided at the ``config/``
directory, most notably *baseval*, which loads the reevaluation files
from a dataset instead of the ones in the three-way train-val.-test
folds.
